"""apply_split_model.py — train_eval_splits 로 만든 split 모델을 전 뷰에 적용해
submodel_applied 스타일 산출물(pred_mask/panel/metrics)을 train/test 구분 저장.

출력: experiments/_splits/<mode>/applied/{train,test}/<group>__<scene>_<view>/
        pred_mask.png, panel.png, metrics.json
      + metrics_train.csv / metrics_test.csv (집계)

사용:
  python scripts/apply_split_model.py --mode view_split
  python scripts/apply_split_model.py --mode scene_split --only test
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import submodel_head as SM


def _scene_of(name):
    import re
    m = re.match(r"^(.+)_v(\d+)$", name.split("/")[-1])
    return m.group(1) if m else name


def _sample_per_scene(views, n):
    """씬별로 균등 간격 n개 샘플 (n<=0 이면 전체)."""
    if n <= 0:
        return views
    by = {}
    for v in views:
        by.setdefault(_scene_of(v), []).append(v)
    out = []
    for sc in sorted(by):
        vs = sorted(by[sc])
        if len(vs) <= n:
            out += vs
        else:
            idx = np.linspace(0, len(vs) - 1, n).round().astype(int)
            out += [vs[i] for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["view_split", "scene_split", "rep_view"])
    ap.add_argument("--only", default="both", choices=["train", "test", "both"])
    ap.add_argument("--per_scene", type=int, default=0,
                    help="씬당 N개만 샘플(균등간격, 0=전체)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    split_dir = os.path.join(C.EXPERIMENTS_DIR, "_splits", args.mode)
    ck = torch.load(os.path.join(split_dir, "model.pt"),
                    map_location=device, weights_only=False)
    head = SM.make_head(ck["in_dim"], linear=ck["linear"], hidden=ck["hidden"]).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()
    mean, std = ck["mean"], ck["std"]

    if args.mode == "rep_view":
        # test 목록은 view_split 의 공통 test 셋에서 v00 제외
        base = torch.load(os.path.join(C.EXPERIMENTS_DIR, "_splits", "view_split", "model.pt"),
                          map_location="cpu", weights_only=False)
        test_views = [v for v in base["test_views"]
                      if not v.split("/")[-1].endswith("_v00")]
        ck["test_views"] = test_views
    print(f"[load] {args.mode} model  (train {len(ck['train_views'])} / "
          f"test {len(ck['test_views'])} views)")

    splits = []
    if args.only in ("train", "both"):
        splits.append(("train", _sample_per_scene(ck["train_views"], args.per_scene)))
    if args.only in ("test", "both"):
        splits.append(("test", _sample_per_scene(ck["test_views"], args.per_scene)))

    for split_name, views in splits:
        rows = []
        base = C.ensure_dir(os.path.join(split_dir, "applied", split_name))
        for i, name in enumerate(views, 1):
            d = SM.build_features(name)
            if d is None or d["X"].shape[1] != ck["in_dim"]:
                print(f"[skip] {name}")
                continue
            p = SM.predict_proba(head, SM.standardize_apply(d["X"], mean, std),
                                 device=device)
            m = SM._eval_block(d["y"], p)

            safe = name.replace("/", "__").replace("\\", "__")
            out_dir = C.ensure_dir(os.path.join(base, safe))
            hw = d["hw"]
            full_hw = SM._full_hw_for(name, hw, C.EXPERIMENTS_DIR)
            left = os.path.join(C.EXPERIMENTS_DIR, name, "input", "left.png")
            SM.save_pred_mask(os.path.join(out_dir, "pred_mask.png"), p, hw, full_hw)
            SM.save_panel(os.path.join(out_dir, "panel.png"), name, p, d["y"], hw, left)
            with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dict(metrics=m, view=name, split=split_name,
                               mode=args.mode), f, indent=2, ensure_ascii=False)
            rows.append(dict(view=name, split=split_name, **m))
            if i % 100 == 0:
                print(f"  [{split_name}] {i}/{len(views)}", flush=True)

        csv_p = os.path.join(split_dir, "applied", f"metrics_{split_name}.csv")
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        ks = ["auc", "iou", "f1"]
        ag = {k: float(np.nanmean([r[k] for r in rows])) for k in ks}
        print(f"[{args.mode}/{split_name}] {len(rows)} views  "
              f"AUC {ag['auc']:.4f}  IoU {ag['iou']:.4f}  F1 {ag['f1']:.4f}")
        print(f"[save] {csv_p}")


if __name__ == "__main__":
    main()
