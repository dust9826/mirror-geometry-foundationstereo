"""verify_paper_numbers.py — 논문용 보고서 수치를 원본 결과 파일에서 직접 재집계/검증.

출력: 콘솔 + experiments/_paper_numbers.json
포함: 데이터셋 규모 / RQ1 GT오차 / RQ2 AHCF 신호(1536뷰 집계) /
      검출(view_split·scene_split·rep_view) / 정공법(GT mask 1536뷰 + e2e 15뷰)
제외: single_scene(씬특화), loft(데이터버그), legacy.
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C

EXP = C.EXPERIMENTS_DIR
GROUPS = ["ds_v1_seed0", "ds_v2_seed1"]
OUT = {}


def jload(p):
    return json.load(open(p, encoding="utf-8"))


# ---- 0) 데이터셋 규모 ----
views = []
for g in GROUPS:
    for d in sorted(os.listdir(os.path.join(EXP, g))):
        if os.path.isfile(os.path.join(EXP, g, d, "fs", "points.npy")):
            views.append(f"{g}/{d}")
scenes = sorted({v.split("/")[-1].rsplit("_v", 1)[0] for v in views})
OUT["dataset"] = dict(n_views=len(views), n_scenes=len(scenes), scenes=scenes,
                      per_group={g: sum(v.startswith(g) for v in views) for g in GROUPS})
print(f"[데이터] {len(views)} views, {len(scenes)} scenes, "
      f"{OUT['dataset']['per_group']}")

# ---- 1) RQ1: GT depth/disp 오차 ----
OUT["rq1_gt_depth"] = {}
for g in GROUPS:
    j = jload(os.path.join(EXP, g, "_eval_gt_depth.json"))
    OUT["rq1_gt_depth"][g] = j
    # 구조 출력(키만)
    print(f"[RQ1 {g}] keys: {list(j.keys())[:8]}")

# ---- 2) RQ2: AHCF 신호 AUC (전 뷰 집계) ----
sig_keys = ["pre_entropy", "post_entropy", "pre_conf", "post_conf",
            "delta_entropy", "delta_conf"]
acc = {k: [] for k in sig_keys}
best_post = []
n_rq2 = 0
for v in views:
    p = os.path.join(EXP, v, "features", "report.json")
    if not os.path.isfile(p):
        continue
    r = jload(p)
    rank = {x["signal"]: x["auc"] for x in r.get("ranked", [])}
    if not rank:
        continue
    n_rq2 += 1
    for k in sig_keys:
        if k in rank:
            acc[k].append(rank[k])
    # post 계열 최고
    posts = [rank[k] for k in ("post_entropy", "post_conf") if k in rank]
    if posts:
        best_post.append(max(posts))
OUT["rq2_ahcf"] = dict(
    n_views=n_rq2,
    median={k: float(np.median(a)) for k, a in acc.items() if a},
    mean={k: float(np.mean(a)) for k, a in acc.items() if a},
    best_post_median=float(np.median(best_post)),
)
print(f"[RQ2] n={n_rq2}")
for k in sig_keys:
    if acc[k]:
        print(f"  {k:14s} AUC median {np.median(acc[k]):.4f}  mean {np.mean(acc[k]):.4f}")
print(f"  best(post*)   median {np.median(best_post):.4f}")

# ---- 3) 검출 모델 (일반화 가능한 3종) ----
OUT["detect"] = {}
for mode in ("view_split", "scene_split", "rep_view"):
    s = jload(os.path.join(EXP, "_splits", mode, "summary.json"))
    entry = dict(n_train_views=s.get("n_train_views", len(s.get("train_views", []))),
                 n_test_views=s.get("n_test_views", s.get("n_test")),
                 overall=s.get("overall_view_mean", s.get("overall")),
                 pooled=s.get("pooled_pixel"),
                 per_scene=s.get("per_scene"), split=s.get("split"))
    # applied train 집계 (있으면)
    ptr = os.path.join(EXP, "_splits", mode, "applied", "metrics_train.csv")
    if os.path.isfile(ptr):
        rows = list(csv.DictReader(open(ptr, encoding="utf-8")))
        entry["train_applied"] = {k: float(np.mean([float(r[k]) for r in rows]))
                                  for k in ("auc", "iou", "f1")}
        entry["n_train_applied"] = len(rows)
    OUT["detect"][mode] = entry
    o = entry["overall"]
    print(f"[검출 {mode}] train {entry['n_train_views']} → test {entry['n_test_views']}: "
          f"AUC {o['auc']:.4f} IoU {o['iou']:.4f} F1 {o['f1']:.4f}")

# ---- 4) 정공법 ----
OUT["sym_gt_mask"] = jload(os.path.join(EXP, "_eval_sym.json"))
o = OUT["sym_gt_mask"]["overall"]
print(f"[정공법 GTmask] n={o['n']} angle_med {o['angle_med']:.2f}° "
      f"perp_med {o['perp_med']:.3f}m ok_both {o['ok_both']*100:.1f}%")

# e2e 15뷰
e2e = []
for p in glob.glob(os.path.join(EXP, "ds_v1_seed0", "*",
                                "mirror_geometry_sym_submodel", "result.json")):
    r = jload(p)
    ev = r.get("eval_vs_gt") or {}
    if ev:
        e2e.append(dict(view=r["scene"], angle=ev["angle_deg"], perp=ev["perp_err_m"]))
a = np.array([x["angle"] for x in e2e]); d = np.array([x["perp"] for x in e2e])
OUT["sym_e2e_pilot"] = dict(
    n=len(e2e), views=[x["view"] for x in e2e],
    angle_med=float(np.median(a)), perp_med=float(np.median(d)),
    ok_both=float(((a < 10) & (d < 0.5)).mean()),
    ok_angle10=float((a < 10).mean()), ok_perp05=float((d < 0.5).mean()),
)
print(f"[정공법 e2e n={len(e2e)}] angle_med {np.median(a):.2f}° "
      f"perp_med {np.median(d):.3f}m ok_both {OUT['sym_e2e_pilot']['ok_both']*100:.0f}%")

json.dump(OUT, open(os.path.join(EXP, "_paper_numbers.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\n저장: experiments/_paper_numbers.json")
