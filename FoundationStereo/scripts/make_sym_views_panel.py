"""
make_sym_views_panel.py — 한 장면의 여러 뷰에 대한 반사대칭 평면 복원 결과 패널.

estimate_mirror_plane_sym.py 가 각 뷰에 저장한 mirror_geometry_sym/result.json 을
모아, 뷰별 입력 영상 + 추정 법선/수직거리 vs GT 를 한 장에 정리한다.
성공(<10° & <0.5m)은 초록, 실패는 빨강 테두리로 표시한다.

사용:
  python scripts/make_sym_views_panel.py --scene_prefix ds_v1_seed0/scandinavian --views 0-9
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_views(spec):
    """'0-9' 또는 '0,3,7' -> [0, 1, ...]"""
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description="반사대칭 복원 다중 뷰 결과 패널")
    ap.add_argument("--scene_prefix", required=True,
                    help="experiments/<prefix>_vNN 의 prefix (예: ds_v1_seed0/scandinavian)")
    ap.add_argument("--views", default="0-9")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    views = parse_views(args.views)
    rows = []
    for v in views:
        scene = f"{args.scene_prefix}_v{v:02d}"
        run_dir = os.path.join(args.out_root, scene)
        res_path = os.path.join(run_dir, "mirror_geometry_sym", "result.json")
        if not os.path.isfile(res_path):
            print(f"WARN: {res_path} 없음 — 건너뜀")
            continue
        with open(res_path, encoding="utf-8") as f:
            r = json.load(f)
        left = plt.imread(os.path.join(run_dir, "input", "left.png"))
        rows.append((f"v{v:02d}", left, r))

    n = len(rows)
    cols = 5
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4.0 * cols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    n_ok = 0
    for ax, (vname, left, r) in zip(axes, rows):
        ev = r.get("eval_vs_gt") or {}
        ang = ev.get("angle_deg")
        derr = ev.get("perp_err_m")
        ok = ang is not None and ang < 10 and derr < 0.5
        n_ok += int(ok)

        ax.imshow(left)
        ax.set_xticks([])
        ax.set_yticks([])
        color = "tab:green" if ok else "tab:red"
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(4)

        nx, ny, nz = r["normal"]
        txt = (f"n=({nx:+.2f},{ny:+.2f},{nz:+.2f})\n"
               f"d={r['perp']:.2f}m (GT {ev.get('gt_perp', float('nan')):.2f}m)\n"
               f"angle={ang:.1f}°  d_err={derr:.2f}m")
        ax.set_title(f"{vname}  {'OK' if ok else 'FAIL'}",
                     fontsize=11, color=color, fontweight="bold")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.5,
                va="bottom", ha="left", family="monospace",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle(
        f"{args.scene_prefix}  |  reflection-symmetry plane recovery "
        f"({n_ok}/{n} success, <10° & <0.5m)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = args.out or os.path.join(
        args.out_root, os.path.dirname(args.scene_prefix),
        f"sym_views_{os.path.basename(args.scene_prefix)}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("저장:", out)


if __name__ == "__main__":
    main()
