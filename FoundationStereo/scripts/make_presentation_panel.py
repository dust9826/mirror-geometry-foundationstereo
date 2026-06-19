"""
make_presentation_panel.py — 발표용 AHCF pre/post 비교 패널 생성.

analyze_features.py 가 저장한 features.npz / report.json 을 재사용해,
발표 슬라이드에 바로 넣을 수 있는 가로형(2행 x 3열) 패널을 만든다:

  1열: 원본 좌영상(left.png) / GT 거울 마스크
  2열: AHCF 이전(pre)  — pre_conf / pre_entropy
  3열: AHCF 이후(post) — post_conf / post_entropy

각 신호 제목에 report.json 의 ROC AUC 를 표기해 "AHCF 이후에 거울 분리
신호가 강화된다"는 메시지가 행 비교만으로 읽히게 한다.

사용:
  python scripts/make_presentation_panel.py --scene ds_v1_seed0/computer_room_v00
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import common
import common as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def norm01(a):
    """디스플레이용 [0,1] 정규화 (유한값 1–99 퍼센타일 기준)."""
    a = np.asarray(a, dtype=np.float32)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(fin, 1), np.percentile(fin, 99)
    if hi <= lo:
        lo, hi = float(fin.min()), float(fin.max())
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser(description="발표용 AHCF pre/post 패널")
    ap.add_argument("--scene", required=True, help="experiments/<scene>")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    ap.add_argument("--out", default=None,
                    help="저장 경로 (기본: features/panel_presentation.png)")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.scene)
    feat_dir = os.path.join(run_dir, "features")
    npz = np.load(os.path.join(run_dir, "fs", "features.npz"))

    # --- 입력: 원본 좌영상 + GT 마스크 (full-res) ---
    left = plt.imread(os.path.join(run_dir, "input", "left.png"))
    mask_full = C.load_mask(os.path.join(run_dir, "input", "mirror_mask.png"))

    # --- 신호 4종 (feature 해상도) ---
    sig = {k: npz[k].astype(np.float32)
           for k in ("pre_conf", "pre_entropy", "post_conf", "post_entropy")}

    # --- report.json 의 AUC 를 제목에 표기 ---
    auc = {}
    rep_path = os.path.join(feat_dir, "report.json")
    if os.path.isfile(rep_path):
        with open(rep_path, encoding="utf-8") as f:
            auc = {r["signal"]: r["auc"] for r in json.load(f)["ranked"]}

    def title(name):
        a = auc.get(name)
        return f"{name}  (AUC={a:.3f})" if a is not None else name

    # --- 2행 x 3열 가로형 레이아웃 ---
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 7.2))

    axes[0, 0].imshow(left)
    axes[0, 0].set_title("Input (left image)", fontsize=12)
    axes[1, 0].imshow(mask_full, cmap="gray")
    axes[1, 0].set_title("GT mirror mask", fontsize=12)

    layout = [
        ("pre_conf", axes[0, 1]), ("post_conf", axes[0, 2]),
        ("pre_entropy", axes[1, 1]), ("post_entropy", axes[1, 2]),
    ]
    for name, ax in layout:
        im = ax.imshow(norm01(sig[name]), cmap="magma")
        ax.set_title(title(name), fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(f"{args.scene}  |  AHCF pre vs. post mirror-signal",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.88])

    # --- 열 헤더 (각 열 중앙 상단, 굵게) ---
    headers = ["Input / GT", "before AHCF (pre)", "after AHCF (post)"]
    for col, header in enumerate(headers):
        p = axes[0, col].get_position()
        fig.text((p.x0 + p.x1) / 2, 0.92, header, ha="center",
                 fontsize=14, fontweight="bold")

    out = args.out or os.path.join(feat_dir, "panel_presentation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("저장:", out)


if __name__ == "__main__":
    main()
