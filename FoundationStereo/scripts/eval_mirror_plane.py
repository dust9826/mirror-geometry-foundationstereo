"""eval_mirror_plane.py — 추정 거울 평면을 GT 거울 평면과 비교.

지표 (mirror 를 plane 으로 표현 후):
  1. normal_angle_deg : 법선 사이 각 = arccos(|n_est . n_gt|) (180° 뒤집힘 무관)
  2. corner_dist_m    : 두 사각형 4모서리 대칭 최근접(chamfer) 평균 거리 (순서 무관)
  (보너스) perp_err_m  : 원점-평면 수직거리 오차
           center_dist_m: 평면 중심 거리

추정 평면 = mirror_geometry/result.json (RANSAC 권장: normal + plane_bbox.corners_3d).
  ※ 현재 RANSAC 은 '반사면'이라 GT 거울면과 차이가 큼(정공법 구현 전 baseline).
   정공법 결과를 같은 형식(normal+corners)으로 주면 그대로 평가 가능.
GT 평면   = mirror_gt/gt.json (mirrors 리스트). 여러 개면 추정과 normal 각이 최소인 것을 매칭.

사용:
  python scripts/eval_mirror_plane.py --scene legacy/computer_room_baseline080
  python scripts/eval_mirror_plane.py --root experiments/legacy        # GT 있는 scene 전부
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)


def normal_angle_deg(n1, n2):
    c = abs(float(np.dot(_unit(n1), _unit(n2))))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def corner_chamfer(c_est, c_gt):
    """두 (4,3) 모서리집합의 대칭 최근접 평균 거리 (모서리 순서 무관)."""
    a = np.asarray(c_est, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(c_gt, dtype=np.float64).reshape(-1, 3)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)  # (na,nb)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def load_estimate(run_dir):
    p = os.path.join(run_dir, "mirror_geometry", "result.json")
    if not os.path.isfile(p):
        return None
    r = json.load(open(p, encoding="utf-8"))
    rr = r.get("recommended_result", {})
    bb = r.get("plane_bbox") or {}
    if "normal" not in rr or "corners_3d" not in bb:
        return None
    return dict(normal=rr["normal"], corners=bb["corners_3d"],
                center=bb.get("center_3d"),
                perp=r.get("mirror_depth", {}).get("perp_distance"),
                method=r.get("recommended"))


def load_gt(run_dir):
    p = os.path.join(run_dir, "mirror_gt", "gt.json")
    if not os.path.isfile(p):
        return None
    g = json.load(open(p, encoding="utf-8"))
    return g.get("mirrors", [])


def eval_scene(run_dir):
    est = load_estimate(run_dir)
    gts = load_gt(run_dir)
    if est is None or not gts:
        return None
    # 여러 GT 거울 중 추정과 법선각 최소인 것 매칭
    cand = []
    for m in gts:
        if "normal" not in m:
            continue
        cand.append((normal_angle_deg(est["normal"], m["normal"]), m))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    ang, gt = cand[0]

    res = dict(
        matched_gt=gt.get("name"),
        normal_angle_deg=round(ang, 3),
        corner_dist_m=round(corner_chamfer(est["corners"], gt["corners_3d"]), 4),
    )
    if est.get("perp") is not None and gt.get("perp_distance") is not None:
        res["perp_err_m"] = round(abs(est["perp"] - gt["perp_distance"]), 4)
        res["perp_est_m"] = round(est["perp"], 3)
        res["perp_gt_m"] = round(gt["perp_distance"], 3)
    if est.get("center") is not None and gt.get("centroid") is not None:
        res["center_dist_m"] = round(float(np.linalg.norm(
            np.asarray(est["center"]) - np.asarray(gt["centroid"]))), 4)
    res["n_gt_mirrors"] = len(gts)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="experiments/<scene> (하위경로 가능)")
    ap.add_argument("--root", help="이 폴더 아래 scene 전부 (gt.json 있는 것)")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    args = ap.parse_args()

    if args.scene:
        scenes = [os.path.join(args.out_root, args.scene)]
    elif args.root:
        scenes = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                        if os.path.isdir(d))
    else:
        raise SystemExit("--scene 또는 --root 필요")

    rows = []
    for sd in scenes:
        r = eval_scene(sd)
        if r is None:
            continue
        r["scene"] = os.path.basename(sd)
        rows.append(r)

    if not rows:
        raise SystemExit("평가 가능 scene 없음 (result.json + gt.json 둘 다 필요)")

    print(f"\n{'scene':32s} {'normAng°':>8s} {'cornerD(m)':>10s} {'perpErr':>8s} {'centerD':>8s}  matchedGT")
    for r in sorted(rows, key=lambda x: x["normal_angle_deg"]):
        print(f"{r['scene'][:32]:32s} {r['normal_angle_deg']:8.2f} {r['corner_dist_m']:10.3f} "
              f"{r.get('perp_err_m', float('nan')):8.3f} {r.get('center_dist_m', float('nan')):8.3f}  "
              f"{r.get('matched_gt')}")
    arr = lambda k: np.array([r[k] for r in rows if k in r], dtype=float)
    print(f"\n중앙값: normalAngle {np.median(arr('normal_angle_deg')):.2f}°  "
          f"cornerDist {np.median(arr('corner_dist_m')):.3f} m  "
          f"perpErr {np.median(arr('perp_err_m')):.3f} m  (n={len(rows)})")
    print("주의: 추정=RANSAC(반사면)이라 GT 거울면과 차이 큼 = '반사면≠거울면' 정량화. "
          "정공법 결과를 result.json 형식으로 주면 동일 평가.")

    if args.root:
        out = os.path.join(args.root, "_eval_mirror_plane.json")
        json.dump(rows, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("저장:", out)


if __name__ == "__main__":
    main()
