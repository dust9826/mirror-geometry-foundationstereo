"""make_mirror_gt.py 순수 함수 단위 테스트 (Blender 불필요).
실행: python scripts/test_make_mirror_gt.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from make_mirror_gt import world_plane_to_cam, project_points, blend_base


def test_identity_camera_z_forward():
    # Blender 카메라 = world identity (–Z 전방). 거울이 5m 앞(-Z)에서 카메라를 향함.
    cam = np.eye(4)
    normal_w = np.array([0.0, 0.0, 1.0])        # +Z(world) = 카메라 쪽
    centroid_w = np.array([0.0, 0.0, -5.0])     # –Z 5m 전방
    corners_w = np.array([[-1, -1, -5], [1, -1, -5], [1, 1, -5], [-1, 1, -5]], float)
    out = world_plane_to_cam(normal_w, centroid_w, corners_w, cam)
    # OpenCV: +Z 전방 → centroid 가 +5
    assert np.allclose(out["centroid"], [0, 0, 5], atol=1e-6), out["centroid"]
    # 법선: F=diag(1,-1,-1) 적용 → (0,0,-1)
    assert np.allclose(out["normal"], [0, 0, -1], atol=1e-6), out["normal"]
    # 평면식 d = -n·c = -((0,0,-1)·(0,0,5)) = 5
    assert abs(out["d"] - 5.0) < 1e-6, out["d"]
    expected_corners = np.array([[-1, 1, 5], [1, 1, 5], [1, -1, 5], [-1, -1, 5]], float)
    assert np.allclose(out["corners"], expected_corners, atol=1e-6), out["corners"]
    print("OK test_identity_camera_z_forward")


def test_y_flip_world_up_is_image_up():
    # world +Y(Blender 위)는 OpenCV 에서 -Y(위=작은 row) 가 돼야 한다 — y 플립 검증.
    cam = np.eye(4)
    out = world_plane_to_cam([0, 0, 1.0], [0.0, 1.0, -5.0],
                             [[0, 1, -5]] * 4, cam)
    assert np.allclose(out["centroid"], [0, -1, 5], atol=1e-6), out["centroid"]
    print("OK test_y_flip_world_up_is_image_up")


def test_x_preserved():
    # world +X 는 OpenCV +X 로 보존(플립 없음).
    cam = np.eye(4)
    out = world_plane_to_cam([0, 0, 1.0], [1.0, 0.0, -5.0],
                             [[1, 0, -5]] * 4, cam)
    assert np.allclose(out["centroid"], [1, 0, 5], atol=1e-6), out["centroid"]
    print("OK test_x_preserved")


def test_projection_principal_point():
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 288], [0, 0, 1.0]])
    uv = project_points(K, [[0.0, 0.0, 5.0]])     # 광축 위의 점 → 주점
    assert np.allclose(uv[0], [512, 288], atol=1e-6), uv
    print("OK test_projection_principal_point")


def test_projection_offset():
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 288], [0, 0, 1.0]])
    uv = project_points(K, [[1.0, 0.0, 5.0]])     # x=1,z=5 → u = 1000*1/5 + 512
    assert abs(uv[0, 0] - (200 + 512)) < 1e-6, uv
    assert abs(uv[0, 1] - 288) < 1e-6, uv
    print("OK test_projection_offset")


def test_blend_base_strips_suffix():
    assert blend_base("computer_room_baseline080") == "computer_room"
    assert blend_base("archiviz_baseline080") == "archiviz"
    assert blend_base("minigym_view04") == "minigym_view04"   # 접미사 없으면 그대로
    print("OK test_blend_base_strips_suffix")


if __name__ == "__main__":
    test_identity_camera_z_forward()
    test_y_flip_world_up_is_image_up()
    test_x_preserved()
    test_projection_principal_point()
    test_projection_offset()
    test_blend_base_strips_suffix()
    print("ALL PASS")
