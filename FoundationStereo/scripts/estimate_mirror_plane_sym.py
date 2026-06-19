"""estimate_mirror_plane_sym.py — 반사대칭으로 '진짜' 거울평면 추정 (정공법).

원리: FS 는 거울 안에서 반사된 가상점 P' 를 본다. 진짜 거울평면 M 에 대해
  P = reflect_M(P') 가 방의 실제점과 겹친다. 대응쌍 (P', Q) 하나가 평면을 결정
  (수직이등분면) → RANSAC:
    1) 거울 안 점 p' × 방 점 q 랜덤 쌍 → 수직이등분면 가설
    2) 거울 안 점들을 가설로 반사 → 방 점군 KD-tree NN 거리의 trimmed mean 스코어
       (거울은 카메라 뒤도 비추므로 대응 없는 점이 많다 → 절사 필수)
    3) 상위 가설을 ICP 식으로 정제 (반사→NN 대응→닫힌형 재추정)

추정은 FS 출력만 사용. GT(mirror_gt/gt.json)는 평가에만 사용.

사용:
  python scripts/estimate_mirror_plane_sym.py --scene ds_v1_seed0/computer_room_v00
  python scripts/estimate_mirror_plane_sym.py --scene ... --mask submodel  # 예측 mask
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import mirror_geometry as MG


# ----------------------------------------------------------------- 순수 기하
def reflect_points(P, n, d):
    """P:(N,3) 를 평면 n·x+d=0 으로 반사."""
    n = np.asarray(n, np.float64)
    return P - 2.0 * (P @ n + d)[:, None] * n[None, :]


def bisector_plane(p, q):
    """두 점의 수직이등분면 (n, d). n 은 p→q 방향. 퇴화(같은점)면 None."""
    v = np.asarray(q, np.float64) - np.asarray(p, np.float64)
    L = np.linalg.norm(v)
    if L < 1e-9:
        return None
    n = v / L
    d = -float(n @ ((np.asarray(p) + np.asarray(q)) / 2.0))
    return n, d


def refine_plane_from_pairs(Pp, Qc):
    """대응쌍 (p'_i, q_i) 들로 대칭평면 닫힌형 재추정.

    n = (q-p') 방향들의 주성분(부호는 평균방향), d = -median(n·중점).
    """
    V = np.asarray(Qc, np.float64) - np.asarray(Pp, np.float64)
    nv = np.linalg.norm(V, axis=1)
    ok = nv > 1e-9
    if ok.sum() < 3:
        return None
    U = V[ok] / nv[ok][:, None]
    # 주성분 (방향 분산 최대 축)
    M = U.T @ U
    w, vec = np.linalg.eigh(M)
    n = vec[:, -1]
    if n @ U.mean(axis=0) < 0:
        n = -n
    mid = (np.asarray(Pp, np.float64)[ok] + np.asarray(Qc, np.float64)[ok]) / 2.0
    d = -float(np.median(mid @ n))
    return n, d


def score_plane(P_s, tree, n, d, keep_frac=0.5):
    """P_s 반사 → tree(방 점군) NN 거리의 하위 keep_frac 평균(작을수록 좋음)."""
    R = reflect_points(P_s, n, d)
    dist, _ = tree.query(R, k=1)
    k = max(1, int(len(dist) * keep_frac))
    part = np.partition(dist, k - 1)[:k]
    return float(part.mean())


def estimate_sym_plane(P_mirror, Q_room, rng=None, n_hyp=4000,
                       n_score=600, n_refine=3000, icp_iters=8,
                       keep_frac=0.5, perp_range=None, verbose=False):
    """반사대칭 평면 추정 코어 (I/O 없음 — 단위테스트 가능).

    P_mirror: (N,3) 거울 안 FS 점(가상점), Q_room: (M,3) 방 점.
    반환 dict(normal, d, score, n_pairs, perp) 또는 None.
    """
    from scipy.spatial import cKDTree
    rng = rng or np.random.default_rng(0)

    tree = cKDTree(Q_room)
    P_s0 = P_mirror[rng.choice(len(P_mirror), size=min(150, len(P_mirror)),
                               replace=False)]            # 1차(저비용) 스코어용
    P_s = P_mirror[rng.choice(len(P_mirror), size=min(n_score, len(P_mirror)),
                              replace=False)]
    P_r = P_mirror[rng.choice(len(P_mirror), size=min(n_refine, len(P_mirror)),
                              replace=False)]

    # ---- 1) 가설 생성 + 2단계 스코어 (150pt 전수 → 상위 40개만 600pt 재평가) ----
    pi = rng.integers(0, len(P_mirror), size=n_hyp)
    qi = rng.integers(0, len(Q_room), size=n_hyp)
    best = []          # (score, n, d)
    for a, b in zip(pi, qi):
        h = bisector_plane(P_mirror[a], Q_room[b])
        if h is None:
            continue
        n, d = h
        perp = abs(d)   # 카메라(원점)에서 평면까지 수직거리
        if perp_range and not (perp_range[0] <= perp <= perp_range[1]):
            continue
        s = score_plane(P_s0, tree, n, d, keep_frac)
        best.append((s, n, d))
    if not best:
        return None
    best.sort(key=lambda t: t[0])
    top = [(score_plane(P_s, tree, n, d, keep_frac), n, d)
           for _, n, d in best[:40]]
    top.sort(key=lambda t: t[0])
    cands = top[:5]

    # ---- 2) ICP 정제 ----
    final = None
    for s0, n, d in cands:
        for _ in range(icp_iters):
            R = reflect_points(P_r, n, d)
            dist, idx = tree.query(R, k=1)
            thr = np.percentile(dist, 60)
            ok = dist <= thr
            if ok.sum() < 10:
                break
            h = refine_plane_from_pairs(P_r[ok], Q_room[idx[ok]])
            if h is None:
                break
            n, d = h
        s = score_plane(P_s, tree, n, d, keep_frac)
        if final is None or s < final[0]:
            final = (s, n, d)
        if verbose:
            print(f"    cand init {s0:.4f} -> refined {s:.4f} perp {abs(d):.2f}")
    s, n, d = final
    return dict(normal=n, d=float(d), score=float(s), perp=float(abs(d)))


# ----------------------------------------------------------------- I/O 래퍼
def load_scene(run_dir, mask_src="gt", erode_px=6, max_mirror_depth=60.0,
               max_room_depth=30.0, n_mirror=4000, n_room=80000, rng=None):
    import cv2
    import imageio.v2 as iio
    rng = rng or np.random.default_rng(0)

    pts = np.load(os.path.join(run_dir, "fs", "points.npy")).astype(np.float64)
    depth = np.load(os.path.join(run_dir, "fs", "depth_meter.npy")).astype(np.float64)
    if mask_src == "gt":
        m = iio.imread(os.path.join(run_dir, "input", "mirror_mask.png"))
    else:   # submodel 예측
        m = iio.imread(os.path.join(run_dir, "submodel_applied", "pred_mask.png"))
    m = (m if m.ndim == 2 else m[..., :3].mean(-1)) > 127
    if m.shape != depth.shape:   # 예측 mask 는 feature 해상도 복원이라 약간 다를 수 있음
        m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)

    k = np.ones((erode_px, erode_px), np.uint8)
    m_er = cv2.erode(m.astype(np.uint8), k, iterations=1).astype(bool)
    m_di = cv2.dilate(m.astype(np.uint8), k, iterations=2).astype(bool)

    valid = depth > 0
    sel_p = m_er & valid & (depth < max_mirror_depth)
    sel_q = (~m_di) & valid & (depth < max_room_depth)
    P = pts[sel_p].reshape(-1, 3)
    Q = pts[sel_q].reshape(-1, 3)
    if len(P) < 200 or len(Q) < 2000:
        return None
    if len(P) > n_mirror:
        P = P[rng.choice(len(P), n_mirror, replace=False)]
    if len(Q) > n_room:
        Q = Q[rng.choice(len(Q), n_room, replace=False)]
    return P, Q


def eval_vs_gt(run_dir, n, d):
    """GT 평면과 비교: 법선각(°), 수직거리 오차(m). 멀티거울이면 최솟값 기준."""
    p = os.path.join(run_dir, "mirror_gt", "gt.json")
    if not os.path.isfile(p):
        return None
    gt = json.load(open(p, encoding="utf-8"))
    out = []
    for m in gt["mirrors"]:
        ng = np.asarray(m["normal"], np.float64)
        cosv = abs(float(ng @ n)) / (np.linalg.norm(ng) * np.linalg.norm(n))
        ang = float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
        dperp = abs(abs(d) - float(m["perp_distance"]))
        out.append(dict(gt_name=m.get("name"), angle_deg=ang,
                        perp_err_m=dperp, gt_perp=float(m["perp_distance"])))
    out.sort(key=lambda r: r["angle_deg"] + r["perp_err_m"])
    return out[0]


def run_scene(scene, out_root=C.EXPERIMENTS_DIR, mask_src="gt", seed=0,
              n_hyp=4000, verbose=True):
    rng = np.random.default_rng(seed)
    run_dir = os.path.join(out_root, scene)
    data = load_scene(run_dir, mask_src=mask_src, rng=rng)
    if data is None:
        print(f"[skip] {scene}: 점 부족")
        return None
    P, Q = data
    # 거울평면 수직거리 범위: 0.3m ~ 가상점 깊이 90% 지점 (거울은 가상점보다 앞)
    perp_hi = float(np.percentile(np.linalg.norm(P, axis=1), 90))
    res = estimate_sym_plane(P, Q, rng=rng, n_hyp=n_hyp,
                             perp_range=(0.3, perp_hi), verbose=verbose)
    if res is None:
        print(f"[fail] {scene}: 가설 없음")
        return None
    n, d = MG.canonicalize_normal(res["normal"], P, res["d"])
    res["normal"], res["d"], res["perp"] = [float(x) for x in n], float(d), abs(float(d))

    ev = eval_vs_gt(run_dir, np.asarray(res["normal"]), res["d"])
    res["scene"], res["mask_src"] = scene, mask_src
    res["n_mirror_pts"], res["n_room_pts"] = int(len(P)), int(len(Q))
    res["eval_vs_gt"] = ev
    sub = "mirror_geometry_sym" if mask_src == "gt" else f"mirror_geometry_sym_{mask_src}"
    out_dir = C.ensure_dir(os.path.join(run_dir, sub))
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    msg = (f"[sym] {scene}: perp {res['perp']:.2f}m score {res['score']:.4f} "
           f"normal {[round(x, 3) for x in res['normal']]}")
    if ev:
        msg += (f"  | GT대비 각도 {ev['angle_deg']:.1f}° perp오차 {ev['perp_err_m']:.2f}m "
                f"(GT {ev['gt_perp']:.2f}m)")
    print(msg, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="experiments/<scene> (하위경로 가능)")
    ap.add_argument("--mask", default="gt", choices=["gt", "submodel"],
                    help="거울 mask 출처 (gt=mirror_mask.png, submodel=예측)")
    ap.add_argument("--n_hyp", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_scene(args.scene, mask_src=args.mask, seed=args.seed,
              n_hyp=args.n_hyp, verbose=not args.quiet)


if __name__ == "__main__":
    main()
