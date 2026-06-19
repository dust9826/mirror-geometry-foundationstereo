"""
mirror_geometry.py — FoundationStereo prior 만으로 거울 평면 기하 추정 (라이브러리).

제안서 "Find mirror geometry inside FoundationStereo's prior" 의 핵심 모듈.
추가 segmentation 네트워크 없이, FoundationStereo 가 뱉은 출력
(depth / points / disparity / cost-volume confidence) 과 거울 마스크만으로
거울의 (1) 평면 거리, (2) 평면 법선, (3) 평면 내/이미지 2D bbox 를 추정한다.

핵심 아이디어:
  거울 마스크가 표시하는 픽셀의 3D 점들은, 스테레오가 예측한 대로
  거울의 평면 표면 위(또는 근처)에 놓인다. 그 점들에 평면을 피팅한다.
  (반사가 약하면 FoundationStereo 는 거울 '표면' depth 를, 강하면 '반사된 장면'
   depth 를 예측하는 경향 -> residual 통계로 피팅 품질을 같이 보고한다.)

좌표 규약:
  카메라좌표 OpenCV 규약 (x 우, y 하, z 전방). 평면식은  n·x + d = 0.
  법선은 canonicalize_normal 로 카메라(원점) 쪽을 향하도록 부호를 맞춘다.

세 가지 평면 추정 함수 (각각 dict 반환):
  fit_plane_pca        : 닫힌형 PCA/최소제곱 평면 (제안서 "직접 평면 방정식").
  fit_plane_ransac     : 반사 outlier 에 강건한 RANSAC 평면 (제안서 "RANSAC 평면 피팅").
  fit_plane_svd_robust : RANSAC seed + Huber IRLS 정제 (보너스).

반환 dict 공통 키:
  normal(3,), d(float), inlier_mask(N,), n_inliers(int),
  mean_abs_residual(float), rms_residual(float), method(str).

의존성: numpy (필수), scipy.linalg (있으면 사용, 없어도 동작).
점 개수 < 3 이면 None 을 반환한다.
"""

import numpy as np


# --------------------------------------------------------------------- helpers
def _residuals(points, normal, d):
    """각 점의 평면까지 부호거리 = n·x + d  (||normal||==1 가정)."""
    return points @ normal + d


def _plane_stats(points, normal, d, inlier_mask=None):
    """residual 통계 dict 조각을 만든다."""
    r = _residuals(points, normal, d)
    if inlier_mask is not None and inlier_mask.any():
        rin = r[inlier_mask]
    else:
        rin = r
    return {
        "mean_abs_residual": float(np.mean(np.abs(rin))),
        "rms_residual": float(np.sqrt(np.mean(rin ** 2))),
    }


def _pca_plane(points):
    """centroid 통과, 공분산 최소고유벡터를 법선으로 하는 평면 (normal, d).

    가능하면 SVD 로 안정적으로 계산한다 (scipy 없이 numpy 만으로).
    """
    c = points.mean(axis=0)
    q = points - c
    # SVD 의 마지막 우특이벡터 = 최소분산 방향 = 평면 법선
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(normal @ c)
    return normal, d


# --------------------------------------------------------------------- (a) PCA
def fit_plane_pca(points_Nx3):
    """닫힌형 PCA / 최소제곱 평면 피팅 (제안서 '직접 평면 방정식, closed form').

    모든 점을 사용해 centroid 통과 + 최소분산 방향 법선 평면을 구한다.
    outlier 에 약하므로(LS) RANSAC 과 비교용 baseline 으로 쓴다.
    """
    pts = np.asarray(points_Nx3, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return None
    normal, d = _pca_plane(pts)
    normal, d = canonicalize_normal(normal, pts, d)
    stats = _plane_stats(pts, normal, d)
    return {
        "method": "pca",
        "normal": normal,
        "d": d,
        "inlier_mask": np.ones(pts.shape[0], dtype=bool),
        "n_inliers": int(pts.shape[0]),
        **stats,
    }


# --------------------------------------------------------------------- (b) RANSAC
def fit_plane_ransac(points_Nx3, thresh=None, iters=1000, min_inlier_frac=0.0):
    """RANSAC 평면 피팅 (제안서 'RANSAC 평면 피팅'), 반사 outlier 에 강건.

    thresh : inlier 판정 거리(m). None 이면 점 좌표 스케일로 자동(중앙값 거리의 1%).
    iters  : RANSAC 반복 횟수. 결정성을 위해 np.random.default_rng(0) 사용.
    최종 inlier 집합에 대해 PCA 로 평면을 refit 한다.
    """
    pts = np.asarray(points_Nx3, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n < 3:
        return None

    if thresh is None:
        # 점군 규모에 비례한 기본 임계값(데이터 단위가 m 라고 가정).
        span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        thresh = max(span * 0.01, 1e-3)

    rng = np.random.default_rng(0)
    best_inliers = None
    best_count = -1

    for _ in range(int(iters)):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        nrm = np.cross(v1, v2)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:  # 거의 일직선 -> 평면 정의 불가
            continue
        nrm = nrm / nn
        dd = -float(nrm @ p0)
        dist = np.abs(pts @ nrm + dd)
        inliers = dist < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        # 모든 표본이 퇴화 -> PCA 로 폴백
        res = fit_plane_pca(pts)
        if res is not None:
            res["method"] = "ransac(fallback_pca)"
        return res

    # 최종 inlier 로 PCA refit (consensus 정제)
    normal, d = _pca_plane(pts[best_inliers])
    normal, d = canonicalize_normal(normal, pts[best_inliers], d)
    # refit 된 평면으로 inlier 재판정
    dist = np.abs(pts @ normal + d)
    inlier_mask = dist < thresh

    stats = _plane_stats(pts, normal, d, inlier_mask)
    return {
        "method": "ransac",
        "normal": normal,
        "d": d,
        "thresh": float(thresh),
        "iters": int(iters),
        "inlier_mask": inlier_mask,
        "n_inliers": int(inlier_mask.sum()),
        **stats,
    }


# --------------------------------------------------------------------- (c) IRLS
def fit_plane_svd_robust(points_Nx3, thresh=None, iters=1000, irls_iters=10,
                         huber_delta=None):
    """RANSAC seed + Huber IRLS 가중 정제 (보너스, 제안서 robust 변형).

    RANSAC inlier 로 시작해, residual 에 Huber 가중치를 주며 가중 PCA 를
    반복(IRLS)해 평면을 정제한다. 큰 residual(반사된 장면 점 등)의 영향을 줄인다.
    """
    pts = np.asarray(points_Nx3, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return None

    seed = fit_plane_ransac(pts, thresh=thresh, iters=iters)
    if seed is None:
        return None
    normal, d = np.asarray(seed["normal"], float), float(seed["d"])

    if huber_delta is None:
        huber_delta = seed.get("thresh", 1e-2)

    w = np.ones(pts.shape[0])
    for _ in range(int(irls_iters)):
        # 가중 centroid + 가중 공분산의 최소고유벡터
        wsum = w.sum() + 1e-12
        c = (w[:, None] * pts).sum(axis=0) / wsum
        q = pts - c
        cov = (q * w[:, None]).T @ q
        evals, evecs = np.linalg.eigh(cov)
        normal = evecs[:, 0]  # 최소고유값 방향
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        d = -float(normal @ c)
        # Huber 가중치 갱신
        r = np.abs(pts @ normal + d)
        w = np.where(r <= huber_delta, 1.0, huber_delta / np.maximum(r, 1e-12))

    normal, d = canonicalize_normal(normal, pts, d)
    inlier_mask = np.abs(pts @ normal + d) < (seed.get("thresh") or huber_delta)
    stats = _plane_stats(pts, normal, d, inlier_mask)
    return {
        "method": "svd_robust_irls",
        "normal": normal,
        "d": d,
        "huber_delta": float(huber_delta),
        "inlier_mask": inlier_mask,
        "n_inliers": int(inlier_mask.sum()),
        **stats,
    }


# --------------------------------------------------------------------- normal sign
def canonicalize_normal(normal, points, d=None):
    """법선이 카메라(원점) 쪽을 향하도록 부호를 통일한다.

    규약: 카메라는 +Z 를 본다. 평면 위 점들의 평균 mean_point 는 카메라 앞(z>0)에
    있으므로, '카메라 쪽을 향하는' 법선이라면 dot(normal, mean_point) < 0 이어야 한다
    (법선이 원점 방향으로 거슬러 향함). 그렇지 않으면 normal, d 의 부호를 뒤집는다.
    d 가 주어지면 같은 부호로 함께 뒤집어 평면식 n·x + d = 0 을 보존한다.
    """
    normal = np.asarray(normal, dtype=np.float64)
    mean_point = np.asarray(points, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    if float(normal @ mean_point) > 0:
        normal = -normal
        if d is not None:
            d = -d
    if d is None:
        return normal
    return normal, d


# --------------------------------------------------------------------- depth
def mirror_depth(normal, d, points=None):
    """카메라 원점에서 평면까지 정보를 반환.

    반환 dict:
      perp_distance     : 원점-평면 수직거리 = |d| / ||normal|| (m).
      signed_d          : 정규화 평면식의 signed d (||normal||=1 기준).
      view_axis_depth   : (points 주어지면) inlier centroid 방향에서 평면과 +Z 시선이
                          만나는 z (시선축 따른 거울 depth). centroid 의 z 로 근사.
    """
    normal = np.asarray(normal, dtype=np.float64)
    nn = np.linalg.norm(normal) + 1e-12
    out = {
        "perp_distance": float(abs(d) / nn),
        "signed_d": float(d / nn),
    }
    if points is not None:
        c = np.asarray(points, dtype=np.float64).reshape(-1, 3).mean(axis=0)
        out["centroid"] = c.tolist()
        out["view_axis_depth"] = float(c[2])
    return out


# --------------------------------------------------------------------- bbox in-plane
def plane_bbox(points_inliers, normal):
    """inlier 3D 점을 평면 위로 투영해, 평면 내 2D 범위(width,height)와 4 모서리 3D 반환.

    평면 내 정규직교 축 (u, v) 를 cross product 로 구성하고, 점들을 (u,v) 좌표로
    투영해 min/max 범위를 잰다. 4 corner 는 그 사각형을 다시 3D 로 환산한 것.
    """
    pts = np.asarray(points_inliers, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return None
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # normal 과 가장 안 나란한 기저축을 골라 안정적으로 u 를 만든다.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(normal @ helper) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    c = pts.mean(axis=0)
    q = pts - c
    su = q @ u
    sv = q @ v
    umin, umax = float(su.min()), float(su.max())
    vmin, vmax = float(sv.min()), float(sv.max())
    width = umax - umin
    height = vmax - vmin

    # 평면 사각형 4 corner (3D, 카메라좌표)
    corners_uv = [(umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)]
    corners_3d = [(c + cu * u + cv * v).tolist() for cu, cv in corners_uv]

    return {
        "width_m": float(width),
        "height_m": float(height),
        "center_3d": c.tolist(),
        "axis_u": u.tolist(),
        "axis_v": v.tolist(),
        "corners_3d": corners_3d,
        "uv_min": [umin, vmin],
        "uv_max": [umax, vmax],
    }


# --------------------------------------------------------------------- reflection
def reflection_matrix(normal, d):
    """평면 (n·x + d = 0) 에 대한 4x4 Householder 반사행렬을 반환.

    MirrorGaussian 등에서 3D 가우시안/점을 거울면 기준으로 미러링할 때 쓴다.
    동차좌표 x_h=[x,y,z,1] 에 대해  x' = R @ x_h  가 거울 대칭점이 된다.

    유도: 단위법선 n, 평면식 n·x + d = 0 (||n||=1).
      반사 R3 = I - 2 n nᵀ (방향),  평행이동항 t = -2 d n (offset).
      [[I-2 n nᵀ, -2 d n],
       [0 0 0   ,   1   ]]
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nn = np.linalg.norm(n) + 1e-12
    n = n / nn
    d = float(d) / nn  # 평면식을 ||n||=1 로 정규화

    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = np.eye(3) - 2.0 * np.outer(n, n)
    R[:3, 3] = -2.0 * d * n
    return R


# --------------------------------------------------------------------- image bbox
def image_bbox_from_mask(mask):
    """거울 마스크의 타이트한 2D 이미지 bbox (x0,y0,x1,y1) 반환. 빈 마스크면 None."""
    m = np.asarray(mask)
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
