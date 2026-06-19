"""test_estimate_mirror_plane_sym.py — 반사대칭 평면 추정 순수함수/코어 테스트.

실행: .venv/Scripts/python.exe -m pytest scripts/test_estimate_mirror_plane_sym.py -q
"""
import numpy as np
import pytest

import estimate_mirror_plane_sym as S


def make_plane(n, d):
    n = np.asarray(n, np.float64)
    return n / np.linalg.norm(n), float(d)


def test_reflect_involution():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(100, 3)) * 3
    n, d = make_plane([0.2, -0.5, 1.0], -2.0)
    R = S.reflect_points(S.reflect_points(P, n, d), n, d)
    assert np.allclose(R, P, atol=1e-9)


def test_reflect_plane_points_fixed():
    """평면 위 점은 반사해도 자기 자신."""
    n, d = make_plane([0, 0, 1.0], -3.0)     # z=3 평면
    P = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 3.0]])
    assert np.allclose(S.reflect_points(P, n, d), P, atol=1e-12)


def test_bisector_plane_recovers_mirror():
    """q 와 p'=reflect(q) 의 수직이등분면 = 원래 평면."""
    n, d = make_plane([0.3, 0.1, -1.0], 2.5)
    q = np.array([0.7, -1.2, 4.0])
    p = S.reflect_points(q[None], n, d)[0]
    n2, d2 = S.bisector_plane(p, q)
    if n2 @ n < 0:
        n2, d2 = -n2, -d2
    assert np.allclose(n2, n, atol=1e-9)
    assert abs(d2 - d) < 1e-9


def test_refine_plane_from_pairs_exact():
    rng = np.random.default_rng(2)
    n, d = make_plane([0.5, -0.2, -1.0], 3.0)
    Q = rng.normal(size=(50, 3)) * 2 + np.array([0, 0, 2.0])
    P = S.reflect_points(Q, n, d)
    h = S.refine_plane_from_pairs(P, Q)
    assert h is not None
    n2, d2 = h
    if n2 @ n < 0:
        n2, d2 = -n2, -d2
    assert np.degrees(np.arccos(np.clip(abs(n2 @ n), -1, 1))) < 0.1
    assert abs(d2 - d) < 1e-6


def _synthetic_room(seed=3, noise=0.0, unmatched_frac=0.0):
    """합성 방: 박스 안 점군 Q + 평면 M 으로 만든 가상점 P'(=reflect(Q_vis))."""
    rng = np.random.default_rng(seed)
    # 방 점군 (카메라 원점 앞 z 1~6m)
    Q = np.column_stack([rng.uniform(-3, 3, 4000),
                         rng.uniform(-2, 2, 4000),
                         rng.uniform(1.0, 6.0, 4000)])
    n, d = make_plane([0.25, -0.1, -1.0], 3.2)    # 거울: 카메라 앞 ~3.2m
    vis = Q[rng.choice(len(Q), 1500, replace=False)]
    P = S.reflect_points(vis, n, d)
    if noise > 0:
        P = P + rng.normal(scale=noise, size=P.shape)
    if unmatched_frac > 0:
        # 카메라 뒤를 비춘 점 흉내: Q 에 대응 없는 가상점 추가
        k = int(len(P) * unmatched_frac)
        ghost = np.column_stack([rng.uniform(-3, 3, k),
                                 rng.uniform(-2, 2, k),
                                 rng.uniform(-6.0, -1.0, k)])
        P = np.vstack([P, S.reflect_points(ghost, n, d)])
    return P, Q, n, d


def _angle(n1, n2):
    return np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), -1, 1)))


def test_estimate_clean():
    P, Q, n, d = _synthetic_room()
    res = S.estimate_sym_plane(P, Q, rng=np.random.default_rng(0),
                               n_hyp=1500, perp_range=(0.3, 10))
    assert res is not None
    assert _angle(res["normal"], n) < 1.0
    assert abs(abs(res["d"]) - abs(d)) < 0.05


def test_estimate_noisy_with_unmatched():
    """노이즈 2cm + 대응없는 점 40% 에서도 복원."""
    P, Q, n, d = _synthetic_room(noise=0.02, unmatched_frac=0.4)
    res = S.estimate_sym_plane(P, Q, rng=np.random.default_rng(0),
                               n_hyp=2500, perp_range=(0.3, 10))
    assert res is not None
    assert _angle(res["normal"], n) < 3.0
    assert abs(abs(res["d"]) - abs(d)) < 0.15


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
