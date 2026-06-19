"""build_seeds_report_figs.py — ds_v1_seed0/ds_v2_seed1 보고서용 figure 수집/생성.

report/figures_seeds/ 에:
  - 각 split 실험(test)의 좋은 panel 3장 + 나쁜 panel 1장 복사
  - single_scene: 12x12 매트릭스 히트맵 생성 + own 최고/cross 최악 panel 생성
  - loft 렌더버그 시각화(L vs R), GT eval 요약 png 복사, 파이프라인 예시 복사
"""
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import submodel_head as SM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"        # 한글
plt.rcParams["axes.unicode_minus"] = False

EXP = C.EXPERIMENTS_DIR
SPLITS = os.path.join(EXP, "_splits")
OUT = C.ensure_dir(os.path.join(C.PROJECT_ROOT, "report", "figures_seeds"))
picked = {}   # 보고서 본문에서 쓸 (라벨 -> 파일명/메타)


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_good_bad(rows, n_good=3):
    """IoU 기준 — 서로 다른 씬에서 top n_good + 최저 1."""
    rows = [r for r in rows if r.get("iou") not in ("", "nan")]
    rows.sort(key=lambda r: -float(r["iou"]))
    good, seen = [], set()
    for r in rows:
        sc = r["view"].split("/")[-1].rsplit("_v", 1)[0]
        if sc in seen:
            continue
        seen.add(sc); good.append(r)
        if len(good) >= n_good:
            break
    bad = rows[-1]
    return good, bad


def copy_panel(mode, view, tag):
    safe = view.replace("/", "__")
    src = os.path.join(SPLITS, mode, "applied", "test", safe, "panel.png")
    dst_name = f"{mode}_{tag}_{safe}.png"
    shutil.copyfile(src, os.path.join(OUT, dst_name))
    return dst_name


# ---- 1) view_split / scene_split / rep_view: test panel 선별 ----
for mode in ("view_split", "scene_split", "rep_view"):
    rows = read_csv(os.path.join(SPLITS, mode, "applied", "metrics_test.csv"))
    good, bad = pick_good_bad(rows)
    items = []
    for i, r in enumerate(good, 1):
        fn = copy_panel(mode, r["view"], f"good{i}")
        items.append(dict(file=fn, view=r["view"], iou=float(r["iou"]),
                          auc=float(r["auc"]), kind="good"))
    fn = copy_panel(mode, bad["view"], "bad")
    items.append(dict(file=fn, view=bad["view"], iou=float(bad["iou"]),
                      auc=float(bad["auc"]), kind="bad"))
    picked[mode] = items
    print(f"[{mode}] good {[r['view'] for r in good]} / bad {bad['view']}")

# ---- 2) single_scene: 매트릭스 히트맵 + own최고/cross최악 panel ----
def load_matrix(key):
    p = os.path.join(SPLITS, "single_scene", f"matrix_{key}.csv")
    with open(p, encoding="utf-8") as f:
        rd = list(csv.reader(f))
    scenes = rd[0][1:]
    M = np.array([[float(x) for x in row[1:]] for row in rd[1:]])
    return scenes, M

for key in ("auc", "iou"):
    scenes, M = load_matrix(key)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M, cmap="viridis", vmin=(0.5 if key == "auc" else 0.0), vmax=1.0)
    ax.set_xticks(range(len(scenes))); ax.set_xticklabels(scenes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenes))); ax.set_yticklabels(scenes, fontsize=8)
    ax.set_xlabel("eval scene"); ax.set_ylabel("model (train scene)")
    ax.set_title(f"single-scene models: {key.upper()} matrix (diag=own)")
    for i in range(len(scenes)):
        for j in range(len(scenes)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.2, color="w" if M[i, j] < 0.75 else "k")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"single_scene_matrix_{key}.png"), dpi=130)
    plt.close(fig)
print("[single_scene] matrix heatmaps 저장")

# own 최고 / cross 최악 panel 생성 (모델 로드해서 렌더)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
pv = read_csv(os.path.join(SPLITS, "single_scene", "per_view_metrics.csv"))
own = [r for r in pv if r["scene"] == r["model_scene"]]
cross = [r for r in pv if r["scene"] != r["model_scene"]]
own_best = max(own, key=lambda r: float(r["iou"]))
cross_worst = min(cross, key=lambda r: float(r["iou"]))

def render_single(r, tag):
    ck = torch.load(os.path.join(SPLITS, "single_scene", f"model_{r['model_scene']}.pt"),
                    map_location=device, weights_only=False)
    head = SM.make_head(ck["in_dim"], linear=ck["linear"], hidden=ck["hidden"]).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()
    d = SM.build_features(r["view"])
    p = SM.predict_proba(head, SM.standardize_apply(d["X"], ck["mean"], ck["std"]),
                         device=device)
    safe = r["view"].replace("/", "__")
    fn = f"single_scene_{tag}_{r['model_scene']}_on_{safe}.png"
    left = os.path.join(EXP, r["view"], "input", "left.png")
    SM.save_panel(os.path.join(OUT, fn),
                  f"model[{r['model_scene']}] -> {r['view']}", p, d["y"], d["hw"], left)
    return dict(file=fn, view=r["view"], model=r["model_scene"],
                iou=float(r["iou"]), auc=float(r["auc"]))

picked["single_scene"] = [
    dict(**render_single(own_best, "own_best"), kind="good"),
    dict(**render_single(cross_worst, "cross_worst"), kind="bad"),
]
print(f"[single_scene] own_best {own_best['view']} / cross_worst {cross_worst['view']}")

# ---- 3) loft 렌더버그 시각화 ----
import imageio.v2 as iio
root0 = os.path.normpath(os.path.join(C.PROJECT_ROOT, "..", "Archive", "ds_v1_seed0"))
L = iio.imread(os.path.join(root0, "loft", "v00", "L.png")).astype(float)
R = iio.imread(os.path.join(root0, "loft", "v00", "R.png")).astype(float)
L2 = iio.imread(os.path.join(root0, "terrazzo", "v00", "L.png")).astype(float)
R2 = iio.imread(os.path.join(root0, "terrazzo", "v00", "R.png")).astype(float)
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, img, t in zip(axes[0], [L / 255, R / 255, np.abs(L - R).mean(-1)],
                      ["loft L", "loft R", "|L-R| (mean 0.02 → BROKEN)"]):
    ax.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "inferno",
              vmin=None if img.ndim == 3 else 0, vmax=None if img.ndim == 3 else 60)
    ax.set_title(t, fontsize=10); ax.axis("off")
for ax, img, t in zip(axes[1], [L2 / 255, R2 / 255, np.abs(L2 - R2).mean(-1)],
                      ["terrazzo L (정상)", "terrazzo R", "|L-R| (mean 22.1 → OK)"]):
    ax.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "inferno",
              vmin=None if img.ndim == 3 else 0, vmax=None if img.ndim == 3 else 60)
    ax.set_title(t, fontsize=10); ax.axis("off")
fig.suptitle("loft 렌더버그: 우측 카메라 미이동 (R=L)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "loft_bug.png"), dpi=110)
plt.close(fig)
print("[loft] bug figure 저장")

# ---- 4) GT eval png + 파이프라인 예시 복사 ----
for g in ("ds_v1_seed0", "ds_v2_seed1"):
    shutil.copyfile(os.path.join(EXP, g, "_eval_gt_depth.png"),
                    os.path.join(OUT, f"eval_gt_depth_{g}.png"))
ex = os.path.join(EXP, "ds_v1_seed0", "computer_room_v00")
shutil.copyfile(os.path.join(ex, "fs", "vis_disp.png"),
                os.path.join(OUT, "pipeline_vis_disp_computer_room_v00.png"))
shutil.copyfile(os.path.join(ex, "mirror_geometry", "ransac_inliers.png"),
                os.path.join(OUT, "pipeline_ransac_computer_room_v00.png"))
print("[copy] GT eval / pipeline 예시 저장")

with open(os.path.join(OUT, "_picked.json"), "w", encoding="utf-8") as f:
    json.dump(picked, f, indent=2, ensure_ascii=False)
print(f"\nDONE → {OUT}")
