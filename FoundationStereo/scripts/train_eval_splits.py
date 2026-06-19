"""train_eval_splits.py — 거울검출 head 의 train/test split 실험 2종.

  A) --mode view_split  : 씬마다 뷰를 7:3 으로 나눈 뒤, **12개 씬의 70% 뷰를 전부
                          합쳐 단일 모델 1개를 학습**한다. 평가는 그 단일 모델로
                          각 씬의 나머지 30% 뷰(unseen 시점)에 대해 수행.
                          (씬별 모델 12개를 만드는 것이 아님 — within-scene 일반화)
  B) --mode scene_split : 씬 자체를 train/test 로 나눠, train 씬들의 전체 뷰를 합쳐
                          단일 모델 학습 → 한 번도 본 적 없는 test 씬 뷰로 평가
                          (cross-scene 일반화)

데이터: experiments/<group>/<scene>_<view>/ (run_fs_server + 후처리 완료분, ds 양 seed 풀링)
학습은 뷰당 픽셀 서브샘플링(거울/비거울 균형)으로 메모리 한도 내 처리.
평가는 뷰 전체 픽셀로 뷰별 지표 산출 → 씬별/전체 집계 + 픽셀 풀링 지표.

사용:
  python scripts/train_eval_splits.py --mode view_split
  python scripts/train_eval_splits.py --mode scene_split \
      --test_scenes blue_bathroom,gym,minimal_interior,terrazzo
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

VIEW_RE = re.compile(r"^(.+)_v(\d+)$")


def list_views(groups):
    """[(scene, group, exp_name)] — fs/features.npz 있는 뷰만."""
    out = []
    for g in groups:
        gd = os.path.join(C.EXPERIMENTS_DIR, g)
        if not os.path.isdir(gd):
            continue
        for name in sorted(os.listdir(gd)):
            m = VIEW_RE.match(name)
            if not m:
                continue
            if os.path.isfile(os.path.join(gd, name, "fs", "features.npz")):
                out.append((m.group(1), g, f"{g}/{name}"))
    return out


def subsample(X, y, n_px, rng):
    """거울/비거울 균형 서브샘플 (양성이 모자라면 전부 + 음성으로 채움)."""
    pos = np.flatnonzero(y > 0.5)
    neg = np.flatnonzero(y <= 0.5)
    n_pos = min(len(pos), n_px // 2)
    n_neg = min(len(neg), n_px - n_pos)
    sel = np.concatenate([
        rng.choice(pos, size=n_pos, replace=False) if n_pos else np.empty(0, np.int64),
        rng.choice(neg, size=n_neg, replace=False) if n_neg else np.empty(0, np.int64),
    ]).astype(np.int64)
    return X[sel], y[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["view_split", "scene_split"])
    ap.add_argument("--groups", default="ds_v1_seed0,ds_v2_seed1")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="view_split 용")
    ap.add_argument("--test_scenes", default="blue_bathroom,gym,minimal_interior,terrazzo",
                    help="scene_split 용 (쉼표목록)")
    ap.add_argument("--px_per_view", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None, help="기본 experiments/_splits/<mode>")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    rng = np.random.default_rng(args.seed)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    out_dir = C.ensure_dir(args.out or os.path.join(C.EXPERIMENTS_DIR, "_splits", args.mode))

    views = list_views(groups)
    scenes = sorted({s for s, _, _ in views})
    print(f"[data] {len(views)} views, {len(scenes)} scenes, groups={groups}")

    # ---- split ----
    if args.mode == "view_split":
        train_v, test_v = [], []
        for sc in scenes:
            vs = [v for v in views if v[0] == sc]
            idx = rng.permutation(len(vs))
            n_tr = int(round(len(vs) * args.train_ratio))
            train_v += [vs[i] for i in idx[:n_tr]]
            test_v += [vs[i] for i in idx[n_tr:]]
        split_info = dict(mode="view_split", train_ratio=args.train_ratio)
    else:
        test_sc = {s.strip() for s in args.test_scenes.split(",") if s.strip()}
        unknown = test_sc - set(scenes)
        if unknown:
            raise SystemExit(f"없는 test scene: {unknown} (있는 씬: {scenes})")
        train_v = [v for v in views if v[0] not in test_sc]
        test_v = [v for v in views if v[0] in test_sc]
        split_info = dict(mode="scene_split", test_scenes=sorted(test_sc),
                          train_scenes=sorted(set(scenes) - test_sc))
    print(f"[split] train {len(train_v)} views / test {len(test_v)} views")

    # ---- train set (뷰당 균형 서브샘플) ----
    Xs, ys, F, names = [], [], None, None
    for i, (sc, g, name) in enumerate(train_v, 1):
        d = SM.build_features(name)
        if d is None:
            continue
        if F is None:
            F, names = d["X"].shape[1], d["feature_names"]
        if d["X"].shape[1] != F:
            print(f"[skip] {name}: feature 차원 불일치")
            continue
        Xi, yi = subsample(d["X"], d["y"], args.px_per_view, rng)
        Xs.append(Xi.astype(np.float32)); ys.append(yi.astype(np.float32))
        if i % 100 == 0:
            print(f"  [load] {i}/{len(train_v)}", flush=True)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    del Xs, ys
    mean, std = SM.standardize_fit(X)
    Xn = SM.standardize_apply(X, mean, std)
    print(f"[train] pixels={X.shape[0]:,}, feat={F}, pos={int(y.sum()):,} "
          f"({y.mean()*100:.1f}%)", flush=True)
    head = SM.train_head(Xn, y, F, linear=args.linear, hidden=args.hidden,
                         epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)
    del X, Xn, y

    torch.save(dict(state_dict=head.state_dict(), mean=mean, std=std,
                    linear=bool(args.linear), hidden=int(args.hidden),
                    in_dim=int(F), feature_names=names,
                    split=split_info, seed=args.seed,
                    px_per_view=args.px_per_view,
                    train_views=[v[2] for v in train_v],
                    test_views=[v[2] for v in test_v]),
               os.path.join(out_dir, "model.pt"))
    print(f"[save] {os.path.join(out_dir, 'model.pt')}")

    # ---- eval (뷰 전체 픽셀) ----
    rows = []
    pool_y, pool_p = [], []
    for i, (sc, g, name) in enumerate(test_v, 1):
        d = SM.build_features(name)
        if d is None or d["X"].shape[1] != F:
            continue
        p = SM.predict_proba(head, SM.standardize_apply(d["X"], mean, std), device=device)
        m = SM._eval_block(d["y"], p)
        rows.append(dict(scene=sc, group=g, view=name, **m))
        # 픽셀 풀링용 (다운샘플해서 메모리 절약)
        sub = rng.choice(len(p), size=min(len(p), 12000), replace=False)
        pool_y.append(d["y"][sub]); pool_p.append(p[sub])
        if i % 100 == 0:
            print(f"  [eval] {i}/{len(test_v)}", flush=True)

    # ---- 집계/저장 ----
    keys = ["auc", "iou", "f1", "accuracy", "precision", "recall"]
    with open(os.path.join(out_dir, "per_view_metrics.csv"), "w", newline="",
              encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)

    def agg(rs):
        return {k: float(np.nanmean([r[k] for r in rs])) for k in keys}

    per_scene = {sc: agg([r for r in rows if r["scene"] == sc])
                 for sc in sorted({r["scene"] for r in rows})}
    pooled = SM._eval_block(np.concatenate(pool_y), np.concatenate(pool_p))
    summary = dict(
        split=split_info, groups=groups, seed=args.seed,
        n_train_views=len(train_v), n_test_views=len(rows),
        px_per_view=args.px_per_view,
        overall_view_mean=agg(rows),
        pooled_pixel=({k: float(pooled[k]) for k in keys}),
        per_scene=per_scene,
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n===== {args.mode} =====")
    print(f"train {len(train_v)} views / test {len(rows)} views")
    o = summary["overall_view_mean"]
    print(f"[전체-뷰평균] AUC {o['auc']:.4f}  IoU {o['iou']:.4f}  F1 {o['f1']:.4f}")
    p = summary["pooled_pixel"]
    print(f"[픽셀풀링]   AUC {p['auc']:.4f}  IoU {p['iou']:.4f}  F1 {p['f1']:.4f}")
    print(f"\n{'scene':22s} {'AUC':>7s} {'IoU':>7s} {'F1':>7s}")
    for sc, m in per_scene.items():
        print(f"{sc:22s} {m['auc']:7.4f} {m['iou']:7.4f} {m['f1']:7.4f}")
    print(f"\n[save] {out_dir} (model.pt, per_view_metrics.csv, summary.json)")


if __name__ == "__main__":
    main()
