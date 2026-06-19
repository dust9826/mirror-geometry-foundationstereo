"""make_mirror_gt_from_depth.py — GT depth + 거울마스크로 GT 거울평면 생성 (Blender 불필요).

ds_v1(Archive) 뷰는 input/depth_gt.npy 가 있고, Blender Z패스는 거울에서 '반사'가 아니라
거울 '표면(유리)' 깊이를 기록한다. 따라서 거울 마스크 픽셀을 K로 역투영하면 거울 표면
3D 점이 되고, 평면 피팅 = 좌측 카메라 OpenCV 좌표의 GT 거울평면이 된다.
(검증: blue_bathroom_v00 평면 rms ≈ 0.007 m = 평탄한 거울 표면)

출력: <scene>/mirror_gt/gt.json (eval_mirror_plane.py 와 호환 스키마)

사용:
  python scripts/make_mirror_gt_from_depth.py --scene archive_views/blue_bathroom_v00
  python scripts/make_mirror_gt_from_depth.py --root experiments/archive_views
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import mirror_geometry as MG


def build_from_depth(run_dir, min_pts=50):
    inp = os.path.join(run_dir, "input")
    dp = os.path.join(inp, "depth_gt.npy")
    mp = os.path.join(inp, "mirror_mask.png")
    if not (os.path.isfile(dp) and os.path.isfile(mp)):
        return None
    meta = C.load_meta(run_dir)
    K = np.asarray(meta["K"], dtype=np.float64)

    depth = np.load(dp).astype(np.float64)
    mask = C.load_mask(mp)
    pts = C.depth_to_points(depth, K)              # (H,W,3) 카메라좌표
    sel = mask & (depth > 0) & np.isfinite(depth)
    P = pts[sel].reshape(-1, 3)
    if len(P) < min_pts:
        return None

    res = MG.fit_plane_ransac(P)                    # 강건 평면(법선은 카메라쪽으로 정규화됨)
    n = np.asarray(res["normal"], dtype=np.float64)
    d = float(res["d"])
    resid = np.abs(P @ n + d)
    thr = max(float(res.get("thresh") or 0.0), 0.02)
    inl = P[resid < thr]
    if len(inl) < 10:
        inl = P
    bbox = MG.plane_bbox(inl, n)
    depth_info = MG.mirror_depth(n, d, inl)
    rms_all = float(np.sqrt((resid ** 2).mean()))

    gt = dict(
        scene=os.path.basename(run_dir),
        frame="left_camera_opencv",
        true_K=K.tolist(),
        gt_source="gt_depth_backproject",
        mirrors=[dict(
            name="mirror_gt_from_depth",
            normal=[float(x) for x in n], d=d,
            perp_distance=depth_info["perp_distance"],
            centroid=depth_info.get("centroid"),
            corners_3d=bbox["corners_3d"],
            plane_bbox=bbox,
            planarity_rms=rms_all,
            n_verts=int(len(P)),
        )],
    )
    out_dir = C.ensure_dir(os.path.join(run_dir, "mirror_gt"))
    json.dump(gt, open(os.path.join(out_dir, "gt.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    return gt["mirrors"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="experiments/<scene> (하위경로 가능)")
    ap.add_argument("--root", help="이 폴더 아래 모든 scene")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    args = ap.parse_args()

    if args.scene:
        dirs = [os.path.join(args.out_root, args.scene)]
    elif args.root:
        dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    else:
        raise SystemExit("--scene 또는 --root 필요")

    ok = skip = 0
    for d in dirs:
        m = build_from_depth(d)
        if m is None:
            skip += 1
            continue
        ok += 1
        print(f"[gt] {os.path.basename(d):28s} perp {m['perp_distance']:.2f}m "
              f"planRMS {m['planarity_rms']:.4f}m  normal "
              f"{[round(x,3) for x in m['normal']]}")
    print(f"\n생성 {ok}, 건너뜀 {skip} (depth_gt/mask 없음 등)")


if __name__ == "__main__":
    main()
