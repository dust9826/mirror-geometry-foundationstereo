"""
estimate_mirror_plane.py — FoundationStereo 출력만으로 거울 평면 기하 추정 (CLI).

mirror_geometry.py 의 세 평면 추정(PCA / RANSAC / IRLS robust)을 한 scene 에 대해
실행하고, 결과(법선/거리/bbox/반사행렬)와 시각화를 저장한다.
추가 segmentation 없이 fs/points.npy + input/mirror_mask.png 만 사용한다.

입력 (experiments/<scene>/):
  input/K.txt, input/mirror_mask.png, input/left.png
  fs/points.npy, fs/valid.npy, fs/depth_meter.npy
  fs/conf.npy (있으면 inlier 평균 conf 보고, 없으면 건너뜀)

출력 (experiments/<scene>/mirror_geometry/):
  result.json : 세 방법 결과 + bbox + reflection matrix + recommended(ransac)
  viz.png     : 마스크/bbox 오버레이 + 방법별 요약 텍스트 + residual 히스토그램
  plane.txt   : normal, d, 4x4 reflection matrix (MirrorGaussian 핸드오프용)

사용:
  python scripts/estimate_mirror_plane.py --scene cozy_living_room_baseline080
"""

import argparse
import json
import os
import sys

# scripts/ 자기 자신을 path 에 넣어 프로젝트 루트에서 실행해도 import common 가능.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import mirror_geometry as MG


def _jsonify(obj):
    """numpy 타입을 json 직렬화 가능한 형태로 변환."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _summary_for_json(res):
    """결과 dict 에서 inlier_mask(대용량 bool) 는 빼고 직렬화한다."""
    if res is None:
        return None
    out = {k: v for k, v in res.items() if k != "inlier_mask"}
    return _jsonify(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="experiments/<scene> (먼저 prepare_data + run_foundation_stereo)")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    ap.add_argument("--thresh", type=float, default=None, help="RANSAC inlier 거리(m); None 이면 자동")
    ap.add_argument("--iters", type=int, default=2000, help="RANSAC 반복 횟수")
    ap.add_argument("--max_depth", type=float, default=50.0,
                    help="이 깊이(m) 초과 점은 제외. disparity≈0 으로 깊이가 폭발하는 "
                         "spurious 점 제거(반사씬은 보통 ~20m 이내).")
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.scene)
    in_dir = os.path.join(run_dir, "input")
    fs_dir = os.path.join(run_dir, "fs")
    out_dir = C.ensure_dir(os.path.join(run_dir, "mirror_geometry"))

    # --- 입력 로드 ---
    pts_path = os.path.join(fs_dir, "points.npy")
    if not os.path.exists(pts_path):
        raise SystemExit(f"points.npy 없음: {pts_path}\n먼저: python scripts/run_foundation_stereo.py --scene {args.scene}")
    points = np.load(pts_path)                          # (H,W,3)
    valid = np.load(os.path.join(fs_dir, "valid.npy"))  # (H,W) bool
    depth = np.load(os.path.join(fs_dir, "depth_meter.npy"))  # (H,W)
    mask = C.load_mask(os.path.join(in_dir, "mirror_mask.png"))  # (H,W) bool
    K, baseline = C.read_K_txt(os.path.join(in_dir, "K.txt"))

    # conf 는 선택적
    conf = None
    conf_path = os.path.join(fs_dir, "conf.npy")
    if os.path.exists(conf_path):
        conf = np.load(conf_path)

    H, W = depth.shape[:2]

    # --- 거울 점 선택: mask & valid & 0<depth<=max_depth ---
    # 거울 영역은 disparity 가 0 근처로 잘못 나와 depth 가 폭발(수천 m)하는 픽셀이
    # 섞일 수 있어, max_depth 로 잘라 RANSAC/PCA 가 폭주하지 않게 한다.
    sel = mask & valid & (depth > 0) & (depth <= args.max_depth)
    n_sel = int(sel.sum())
    print(f"[scene] {args.scene}  size={W}x{H}")
    print(f"[select] mirror 점 개수 = {n_sel}  (mask={int(mask.sum())}, valid&depth = {int((valid & (depth>0)).sum())})")
    if n_sel < 3:
        # 거울 마스크가 비었거나(일부 scene 의 GT 마스크가 비정상) 유효 3D 점이 없음.
        # --all 파이프라인이 멈추지 않도록 skip 기록만 남기고 정상 종료한다.
        mask_px = int(mask.sum())
        if mask_px < 3:
            why = "이 scene 의 mirror_mask 가 비어 있음(렌더 시 거울 흰색표시 실패)."
        else:
            why = (f"mask 는 {mask_px}px 이지만 0<depth<={args.max_depth}m 인 유효 3D 점이 없음 "
                   "(FS 가 거울 반사를 매칭 못해 disparity≈0 -> depth 폭발). --max_depth 조정 가능.")
        msg = f"유효한 거울 3D 점이 3개 미만 - 평면 추정 skip. {why}"
        print("[skip]", msg)
        out_dir = C.ensure_dir(os.path.join(args.out_root, args.scene, "mirror_geometry"))
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump({"scene": args.scene, "status": "skipped",
                       "reason": msg, "mask_px": int(mask.sum())}, f, indent=2)
        return

    mirror_pts = points[sel].reshape(-1, 3).astype(np.float64)
    if conf is not None:
        print(f"[conf] mirror 영역 평균 confidence = {float(conf[sel].mean()):.4f}")

    # --- 세 가지 평면 추정 ---
    res_pca = MG.fit_plane_pca(mirror_pts)
    res_ransac = MG.fit_plane_ransac(mirror_pts, thresh=args.thresh, iters=args.iters)
    res_irls = MG.fit_plane_svd_robust(mirror_pts, thresh=args.thresh, iters=args.iters)

    methods = [("PCA (closed form)", res_pca),
               ("RANSAC", res_ransac),
               ("SVD robust (IRLS)", res_irls)]

    print("\n=== 평면 추정 결과 ===")
    for label, r in methods:
        if r is None:
            print(f"  [{label}] 실패(점 부족)")
            continue
        md = MG.mirror_depth(r["normal"], r["d"], mirror_pts)
        print(f"  [{label}]")
        print(f"      normal = [{r['normal'][0]:+.4f}, {r['normal'][1]:+.4f}, {r['normal'][2]:+.4f}]")
        print(f"      d = {r['d']:+.4f}   mirror_depth(perp) = {md['perp_distance']:.4f} m   view_z = {md['view_axis_depth']:.4f} m")
        print(f"      n_inliers = {r['n_inliers']}/{mirror_pts.shape[0]}   rms_residual = {r['rms_residual']:.5f} m")

    # --- best = RANSAC; bbox / image bbox / reflection matrix ---
    best = res_ransac if res_ransac is not None else res_pca
    best_label = "ransac" if res_ransac is not None else "pca"
    inlier_pts = mirror_pts[best["inlier_mask"]] if best["inlier_mask"].any() else mirror_pts
    bbox3d = MG.plane_bbox(inlier_pts, best["normal"])
    img_bbox = MG.image_bbox_from_mask(mask)
    refl = MG.reflection_matrix(best["normal"], best["d"])
    best_depth = MG.mirror_depth(best["normal"], best["d"], inlier_pts)

    # --- result.json ---
    result = {
        "scene": args.scene,
        "image_size": [int(W), int(H)],
        "baseline_m": float(baseline),
        "n_mirror_points": n_sel,
        "mean_conf": (float(conf[sel].mean()) if conf is not None else None),
        "methods": {
            "pca": _summary_for_json(res_pca),
            "ransac": _summary_for_json(res_ransac),
            "svd_robust": _summary_for_json(res_irls),
        },
        "recommended": "ransac" if res_ransac is not None else "pca",
        "recommended_result": _summary_for_json(best),
        "mirror_depth": _jsonify(best_depth),
        "plane_bbox": _jsonify(bbox3d),
        "image_bbox_xyxy": list(img_bbox) if img_bbox else None,
        "reflection_matrix_4x4": refl.tolist(),
        "note": "K 는 가정값일 수 있어 depth/거리 스케일에 영향. normal/reflection 은 K 스케일과 무관하게 방향은 보존.",
    }
    res_json_path = os.path.join(out_dir, "result.json")
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {res_json_path}")

    # --- plane.txt (MirrorGaussian 핸드오프) ---
    plane_txt_path = os.path.join(out_dir, "plane.txt")
    with open(plane_txt_path, "w", encoding="utf-8") as f:
        f.write(f"# mirror plane for scene: {args.scene}\n")
        f.write(f"# recommended method: {best_label}\n")
        f.write("# plane equation: n . x + d = 0   (camera coords, OpenCV, ||n||=1)\n")
        n = best["normal"]
        f.write(f"normal {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
        f.write(f"d {best['d']:.8f}\n")
        f.write(f"perp_distance_m {best_depth['perp_distance']:.8f}\n")
        f.write("# 4x4 Householder reflection matrix (row-major)\n")
        f.write("reflection_matrix\n")
        for row in refl:
            f.write("  " + " ".join(f"{v:.8f}" for v in row) + "\n")
    print(f"[save] {plane_txt_path}")

    # --- viz.png ---
    try:
        _save_viz(os.path.join(out_dir, "viz.png"), in_dir, mask, img_bbox,
                  mirror_pts, methods, best)
        print(f"[save] {os.path.join(out_dir, 'viz.png')}")
    except Exception as e:
        print("[viz] 시각화 건너뜀:", e)

    print("\nDONE:", out_dir)


def _save_viz(path, in_dir, mask, img_bbox, mirror_pts, methods, best):
    """좌: left + mask 오버레이 + image bbox / 우상: 방법별 요약 텍스트 / 우하: residual 히스토그램."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(13, 6))

    # 좌: 이미지 + 마스크 오버레이
    ax_img = fig.add_subplot(1, 2, 1)
    left_path = os.path.join(in_dir, "left.png")
    if os.path.exists(left_path):
        import imageio.v2 as imageio
        img = imageio.imread(left_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        ax_img.imshow(img[..., :3])
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = [1.0, 0.0, 0.0, 0.35]  # 빨강 반투명
    ax_img.imshow(overlay)
    if img_bbox:
        x0, y0, x1, y1 = img_bbox
        ax_img.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor="yellow", linewidth=2))
        ax_img.set_title(f"mirror mask + image bbox\n{img_bbox}")
    ax_img.axis("off")

    # 우상: 방법별 요약 텍스트
    ax_txt = fig.add_subplot(2, 2, 2)
    ax_txt.axis("off")
    lines = ["estimated mirror plane (camera coords)\n"]
    for label, r in methods:
        if r is None:
            lines.append(f"{label}: FAILED\n")
            continue
        md = MG.mirror_depth(r["normal"], r["d"], mirror_pts)
        n = r["normal"]
        lines.append(
            f"[{label}]\n"
            f"  n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})  d={r['d']:+.3f}\n"
            f"  depth(perp)={md['perp_distance']:.3f} m\n"
            f"  inliers={r['n_inliers']}/{mirror_pts.shape[0]}  rms={r['rms_residual']:.4f} m\n"
        )
    ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9, transform=ax_txt.transAxes)

    # 우하: best 평면의 residual 히스토그램
    ax_hist = fig.add_subplot(2, 2, 4)
    r = mirror_pts @ best["normal"] + best["d"]
    ax_hist.hist(r, bins=60, color="steelblue", alpha=0.85)
    ax_hist.axvline(0.0, color="k", lw=1)
    ax_hist.set_title(f"signed residual (best={best['method']})")
    ax_hist.set_xlabel("distance to plane [m]")
    ax_hist.set_ylabel("count")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
