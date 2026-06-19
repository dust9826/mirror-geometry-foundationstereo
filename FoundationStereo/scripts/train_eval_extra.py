"""train_eval_extra.py — 추가 모델 실험 2종 (view_split 의 split 을 재사용해 비교 가능).

  ① --mode single_scene : 씬 1개의 train 뷰(view_split 의 70%)만으로 모델 학습 ×12개
                           → 공통 test 뷰(464)로 평가 → 12x12 매트릭스
                           (대각 = 자기 씬 unseen 뷰, 비대각 = 본 적 없는 씬)
  ② --mode rep_view     : 각 씬의 대표뷰 v00(기준 카메라, seed0) 12장만으로 단일 모델
                           학습(뷰 전체 픽셀) → 공통 test 뷰(v00 제외)로 평가

공통 test = experiments/_splits/view_split/model.pt 의 test_views (재현성).

사용:
  python scripts/train_eval_extra.py --mode single_scene
  python scripts/train_eval_extra.py --mode rep_view
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import submodel_head as SM
from train_eval_splits import subsample, VIEW_RE

KEYS = ["auc", "iou", "f1", "accuracy", "precision", "recall"]


def scene_of(view_name):
    """'ds_v1_seed0/bedroom_v07' -> 'bedroom'"""
    base = view_name.split("/")[-1]
    m = VIEW_RE.match(base)
    return m.group(1) if m else base


def build_train_matrix(view_names, px_per_view, rng):
    Xs, ys, F, names = [], [], None, None
    for name in view_names:
        d = SM.build_features(name)
        if d is None:
            continue
        if F is None:
            F, names = d["X"].shape[1], d["feature_names"]
        if d["X"].shape[1] != F:
            continue
        if px_per_view and px_per_view > 0:
            Xi, yi = subsample(d["X"], d["y"], px_per_view, rng)
        else:
            Xi, yi = d["X"], d["y"]
        Xs.append(Xi.astype(np.float32)); ys.append(yi.astype(np.float32))
    return np.concatenate(Xs), np.concatenate(ys), F, names


def train_one(X, y, F, args, device):
    mean, std = SM.standardize_fit(X)
    Xn = SM.standardize_apply(X, mean, std)
    head = SM.train_head(Xn, y, F, linear=args.linear, hidden=args.hidden,
                         epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)
    return head, mean, std


def eval_views(head, mean, std, F, views, device, cache=None):
    rows = []
    for name in views:
        d = cache.get(name) if cache is not None else SM.build_features(name)
        if d is None or d["X"].shape[1] != F:
            continue
        p = SM.predict_proba(head, SM.standardize_apply(d["X"], mean, std), device=device)
        m = SM._eval_block(d["y"], p)
        rows.append(dict(view=name, scene=scene_of(name), **{k: m[k] for k in KEYS}))
    return rows


def agg(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["single_scene", "rep_view"])
    ap.add_argument("--px_per_view", type=int, default=4000, help="single_scene 용")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    rng = np.random.default_rng(args.seed)

    base_ck = torch.load(os.path.join(C.EXPERIMENTS_DIR, "_splits", "view_split", "model.pt"),
                         map_location="cpu", weights_only=False)
    train_views, test_views = base_ck["train_views"], base_ck["test_views"]
    scenes = sorted({scene_of(v) for v in train_views})
    out_dir = C.ensure_dir(os.path.join(C.EXPERIMENTS_DIR, "_splits", args.mode))
    print(f"[base] view_split 재사용: train {len(train_views)} / test {len(test_views)}, "
          f"{len(scenes)} scenes", flush=True)

    # test 뷰 feature 캐시 (여러 모델이 같은 test 셋 평가 → 반복 로드 방지, 메모리 ~15GB
    # 는 곤란하니 X 만 float16 으로 들고 있기엔 also big — 그냥 매번 로드(단순/안전).
    cache = None

    if args.mode == "single_scene":
        matrix = {}     # model_scene -> eval_scene -> metrics
        own = {}
        rows_all = []
        for i, sc in enumerate(scenes, 1):
            tv = [v for v in train_views if scene_of(v) == sc]
            X, y, F, names = build_train_matrix(tv, args.px_per_view, rng)
            head, mean, std = train_one(X, y, F, args, device)
            del X, y
            torch.save(dict(state_dict=head.state_dict(), mean=mean, std=std,
                            linear=bool(args.linear), hidden=int(args.hidden),
                            in_dim=int(F), feature_names=names,
                            train_scene=sc, train_views=tv),
                       os.path.join(out_dir, f"model_{sc}.pt"))
            rows = eval_views(head, mean, std, F, test_views, device, cache)
            for r in rows:
                r["model_scene"] = sc
            rows_all += rows
            per = {esc: agg([r for r in rows if r["scene"] == esc])
                   for esc in scenes}
            matrix[sc] = per
            own[sc] = per[sc]
            others = agg([r for r in rows if r["scene"] != sc])
            print(f"[{i:2d}/12] {sc:22s} own AUC {per[sc]['auc']:.3f}/IoU {per[sc]['iou']:.3f}"
                  f"  | others AUC {others['auc']:.3f}/IoU {others['iou']:.3f}", flush=True)

        # 매트릭스 CSV (행=모델씬, 열=평가씬)
        for key in ("auc", "iou"):
            with open(os.path.join(out_dir, f"matrix_{key}.csv"), "w", newline="",
                      encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["model_scene"] + scenes)
                for sc in scenes:
                    w.writerow([sc] + [f"{matrix[sc][esc][key]:.4f}" for esc in scenes])
        with open(os.path.join(out_dir, "per_view_metrics.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
            w.writeheader(); w.writerows(rows_all)
        summary = dict(
            mode="single_scene", px_per_view=args.px_per_view, scenes=scenes,
            own_mean=agg([r for r in rows_all if r["scene"] == r["model_scene"]]),
            cross_mean=agg([r for r in rows_all if r["scene"] != r["model_scene"]]),
            matrix=matrix,
        )
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        o, x = summary["own_mean"], summary["cross_mean"]
        print(f"\n[single_scene 종합] own(자기씬) AUC {o['auc']:.4f}/IoU {o['iou']:.4f}"
              f"  vs cross(타씬) AUC {x['auc']:.4f}/IoU {x['iou']:.4f}")

    else:   # rep_view
        rep = [f"ds_v1_seed0/{sc}_v00" for sc in scenes]
        rep = [v for v in rep
               if os.path.isfile(os.path.join(C.EXPERIMENTS_DIR, v, "fs", "features.npz"))]
        print(f"[rep_view] 대표뷰 {len(rep)}개 (각 씬 v00, seed0, 뷰 전체 픽셀)")
        X, y, F, names = build_train_matrix(rep, 0, rng)   # 전체 픽셀
        print(f"[train] pixels={X.shape[0]:,} pos={int(y.sum()):,}", flush=True)
        head, mean, std = train_one(X, y, F, args, device)
        del X, y
        torch.save(dict(state_dict=head.state_dict(), mean=mean, std=std,
                        linear=bool(args.linear), hidden=int(args.hidden),
                        in_dim=int(F), feature_names=names, train_views=rep),
                   os.path.join(out_dir, "model.pt"))

        ev = [v for v in test_views if not v.split("/")[-1].endswith("_v00")]
        rows = eval_views(head, mean, std, F, ev, device, cache)
        per_scene = {sc: agg([r for r in rows if r["scene"] == sc])
                     for sc in sorted({r["scene"] for r in rows})}
        with open(os.path.join(out_dir, "per_view_metrics.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        summary = dict(mode="rep_view", train_views=rep, n_test=len(rows),
                       overall=agg(rows), per_scene=per_scene)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        o = summary["overall"]
        print(f"\n[rep_view 종합] test {len(rows)}뷰  AUC {o['auc']:.4f}  "
              f"IoU {o['iou']:.4f}  F1 {o['f1']:.4f}")
        print(f"{'scene':22s} {'AUC':>7s} {'IoU':>7s}")
        for sc, m in per_scene.items():
            print(f"{sc:22s} {m['auc']:7.4f} {m['iou']:7.4f}")

    print(f"\n[save] {out_dir}")


if __name__ == "__main__":
    main()
