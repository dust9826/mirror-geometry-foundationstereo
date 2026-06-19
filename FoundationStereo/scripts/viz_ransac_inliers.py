"""
viz_ransac_inliers.py — RANSAC 거울 평면 추정의 inlier/outlier 를 그림으로 보여준다.

거울 마스크 영역의 FoundationStereo 3D 점에 RANSAC 평면을 맞춘 뒤,
  - inlier(평면에 가까운 진짜 거울 점) = 초록
  - outlier(반사로 엉뚱하게 찍힌 점)   = 빨강
으로 (1) 좌영상 위 오버레이, (2) top-down(X-Z) 산점도 + 적합 평면 직선,
(3) 점-평면 잔차 히스토그램(+ inlier 임계선) 을 한 장에 그린다.

사용:
  python scripts/viz_ransac_inliers.py --scene computer_room_baseline080
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows 한글 폰트(맑은 고딕)로 제목/라벨이 네모로 깨지지 않게 설정.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
import imageio.v2 as imageio

import common as C
import mirror_geometry as MG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    ap.add_argument("--thresh", type=float, default=None, help="RANSAC inlier 거리(m)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--max_depth", type=float, default=50.0)
    args = ap.parse_args()

    run = os.path.join(args.out_root, args.scene)
    fs, ind = os.path.join(run, "fs"), os.path.join(run, "input")
    points = np.load(os.path.join(fs, "points.npy"))
    valid = np.load(os.path.join(fs, "valid.npy"))
    depth = np.load(os.path.join(fs, "depth_meter.npy"))
    mask = C.load_mask(os.path.join(ind, "mirror_mask.png"))
    left = imageio.imread(os.path.join(ind, "left.png"))[..., :3]

    sel = mask & valid & (depth > 0) & (depth <= args.max_depth)
    rows, cols = np.where(sel)
    pts = points[sel].astype(np.float64)
    if len(pts) < 3:
        raise SystemExit(f"유효 거울 점 부족({len(pts)}). 다른 scene 선택.")

    res = MG.fit_plane_ransac(pts, thresh=args.thresh, iters=args.iters)
    inl = res["inlier_mask"].astype(bool)
    n, d = np.asarray(res["normal"], float), float(res["d"])
    resid = pts @ n + d                      # 점-평면 부호 거리
    thr = args.thresh
    if thr is None:                          # 자동 임계(추정에 쓰인 값 근사): inlier 잔차 최대
        thr = float(np.abs(resid[inl]).max()) if inl.any() else float(np.abs(resid).std())

    print(f"[{args.scene}] 거울 점 {len(pts)}개 중 inlier {int(inl.sum())} "
          f"({100*inl.mean():.1f}%) / outlier {int((~inl).sum())}")
    print(f"  normal={np.round(n,3)} d={d:.3f}  rms(inlier)={res['rms_residual']:.4f} m  thr≈{thr:.3f} m")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(18, 5.5))

    # (1) 좌영상 오버레이
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(left)
    ax1.scatter(cols[~inl], rows[~inl], s=1, c="red", alpha=0.35, label=f"outlier ({int((~inl).sum())})")
    ax1.scatter(cols[inl], rows[inl], s=1, c="lime", alpha=0.35, label=f"inlier ({int(inl.sum())})")
    ax1.set_title("(1) 좌영상 위 거울 점\ninlier=초록 / outlier=빨강")
    ax1.axis("off")
    lg = ax1.legend(loc="lower right", markerscale=8, framealpha=0.8, fontsize=9)

    # (2) top-down X-Z 산점도 + 평면 직선
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(pts[~inl, 0], pts[~inl, 2], s=2, c="red", alpha=0.4, label="outlier")
    ax2.scatter(pts[inl, 0], pts[inl, 2], s=2, c="green", alpha=0.4, label="inlier")
    # 평면을 X-Z 평면에 투영한 직선: n_x*X + n_z*Z + d = 0 (Y 평균에서)
    ybar = float(pts[inl, 1].mean()) if inl.any() else float(pts[:, 1].mean())
    xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 50)
    if abs(n[2]) > 1e-6:
        zs = -(n[0] * xs + n[1] * ybar + d) / n[2]
        ax2.plot(xs, zs, "b-", lw=2, label="RANSAC 평면")
    ax2.set_xlabel("X (m, 카메라 우측)")
    ax2.set_ylabel("Z (m, 깊이)")
    ax2.set_title("(2) 위에서 본 거울 점 (X-Z)\n반사 outlier 는 평면에서 멀리 흩어짐")
    ax2.legend(markerscale=4, fontsize=9)
    ax2.grid(alpha=0.3)

    # (3) 잔차 히스토그램
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(resid[inl], bins=60, color="green", alpha=0.6, label="inlier")
    ax3.hist(resid[~inl], bins=60, color="red", alpha=0.5, label="outlier")
    ax3.axvline(+thr, color="b", ls="--", lw=1.2)
    ax3.axvline(-thr, color="b", ls="--", lw=1.2, label=f"inlier 임계 ±{thr:.2f} m")
    ax3.set_xlabel("점-평면 부호 거리 (m)")
    ax3.set_ylabel("점 개수")
    ax3.set_title("(3) 잔차 분포\ninlier 는 0 근처, outlier 는 꼬리")
    ax3.legend(fontsize=9)

    fig.suptitle(f"RANSAC inlier/outlier — {args.scene}  "
                 f"(inlier {int(inl.sum())}/{len(pts)}, rms {res['rms_residual']:.3f} m)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = C.ensure_dir(os.path.join(run, "mirror_geometry"))
    path = os.path.join(out, "ransac_inliers.png")
    fig.savefig(path, dpi=120)
    print("[save]", path)


if __name__ == "__main__":
    main()
