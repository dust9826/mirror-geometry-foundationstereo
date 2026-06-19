"""export_mirror_gt.py — (Blender 전용) .blend 거울 본체 메쉬의 GT 평면(world)을 추출.

거울 메쉬 정점을 matrix_world 로 world 좌표 변환 후 최소제곱 평면을 피팅한다.
좌표 변환(OpenCV)·시각화는 하지 않는다(그건 make_mirror_gt.py 담당).

실행:
  blender --background scene.blend --python export_mirror_gt.py -- \
      --out gt_mirror_world.json --camera CM
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

# find_mirrors 재사용 (같은 폴더의 렌더 스크립트). Blender 안이라 bpy import 안전.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_reflect3r_stereo import find_mirrors


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="CM")
    p.add_argument("--mirror_object", default="")
    return p.parse_args(argv)


def world_vertices(obj):
    """거울 메쉬의 world 좌표 정점 (N,3). 모디파이어 반영 evaluated mesh, 실패 시 raw."""
    mw = obj.matrix_world
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        oe = obj.evaluated_get(dg)
        me = oe.to_mesh()
        verts = np.array([(mw @ v.co)[:] for v in me.vertices], dtype=np.float64)
        oe.to_mesh_clear()
    except Exception:
        verts = np.array([(mw @ v.co)[:] for v in obj.data.vertices], dtype=np.float64)
    return verts


def fit_world_plane(obj):
    """거울 메쉬 → world 평면 dict(name, normal, centroid, d, corners, planarity_rms, n_verts)."""
    verts = world_vertices(obj)
    if len(verts) < 3:
        raise ValueError(f"거울 '{obj.name}' 정점 {len(verts)}개 < 3 — 평면 피팅 불가")
    c = verts.mean(axis=0)
    # SVD: 최소 특이벡터 = 법선, 첫 두 벡터 = 평면 내 기저(u,v)
    _, _, vt = np.linalg.svd(verts - c)
    n = vt[-1] / (np.linalg.norm(vt[-1]) + 1e-12)
    u, v = vt[0], vt[1]
    resid = (verts - c) @ n
    rms = float(np.sqrt(np.mean(resid ** 2)))
    su, sv = (verts - c) @ u, (verts - c) @ v
    umin, umax, vmin, vmax = float(su.min()), float(su.max()), float(sv.min()), float(sv.max())
    corners = [(c + cu * u + cv * v).tolist()
               for cu, cv in [(umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)]]
    return {
        "name": obj.name,
        "normal": n.tolist(),
        "centroid": c.tolist(),
        "d": float(-(n @ c)),
        "corners": corners,
        "planarity_rms": rms,
        "n_verts": int(len(verts)),
    }


def mat4(m):
    return [list(row) for row in m]


def camera_intrinsics(cam):
    """PERSP 카메라의 실제 intrinsics (lens/sensor/fit/shift). 비PERSP/없음이면 None.

    중요: 데이터셋 K 는 lens=50mm 가정값이지만 .blend 실제 렌즈는 다를 수 있다
    (예: computer_room=30mm). overlay 재투영은 이 실제값을 써야 렌더 영상과 맞는다.
    """
    if cam is None or getattr(cam, "data", None) is None or cam.data.type != "PERSP":
        return None
    d = cam.data
    return {
        "lens_mm": float(d.lens),
        "sensor_width_mm": float(d.sensor_width),
        "sensor_height_mm": float(d.sensor_height),
        "sensor_fit": d.sensor_fit,
        "shift_x": float(d.shift_x),
        "shift_y": float(d.shift_y),
    }


def main():
    a = parse_args()
    mirrors = find_mirrors(a.mirror_object)
    cam = bpy.data.objects.get(a.camera)
    out = {
        "camera": a.camera,
        "left_camera_to_world": mat4(cam.matrix_world) if cam else None,
        "camera_intrinsics": camera_intrinsics(cam),
        "mirror_names": [o.name for o in mirrors],
        "mirrors": [fit_world_plane(o) for o in mirrors],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("WROTE", a.out, "mirrors:", out["mirror_names"])


if __name__ == "__main__":
    main()
