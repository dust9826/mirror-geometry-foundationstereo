"""probe_mirrors.py — (Blender 전용) 카메라에 보이는 거울 후보 진단.

각 MESH 의 (이름에 mirror 포함?, glossy 저거칠기 재질?, CM 카메라 투영 면적%)을
출력해, '화면에 크게 보이는 거울'이 무엇인지 / find_mirrors 가 옳게 골랐는지 점검한다.

실행:
  blender --background scene.blend --python probe_mirrors.py -- --camera CM
"""
import argparse
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_reflect3r_stereo import find_mirrors, _is_mirror_name


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="CM")
    return p.parse_args(argv)


def has_glossy(o):
    for sl in o.material_slots:
        m = sl.material
        if not (m and m.use_nodes):
            continue
        for n in m.node_tree.nodes:
            if n.type == "BSDF_GLOSSY":
                try:
                    if n.inputs["Roughness"].default_value <= 0.15:
                        return True
                except Exception:
                    return True
    return False


def proj_area_frac(scene, cam, o):
    """obj bbox 8 corner 를 카메라 뷰로 투영해 대략적 화면 면적 비율(0~1) 반환."""
    mw = o.matrix_world
    xs, ys, anyfront = [], [], False
    for c in o.bound_box:
        co = world_to_camera_view(scene, cam, mw @ __import__("mathutils").Vector(c))
        if co.z > 0:                       # 카메라 앞
            anyfront = True
            xs.append(min(max(co.x, 0.0), 1.0))
            ys.append(min(max(co.y, 0.0), 1.0))
    if not anyfront or not xs:
        return 0.0
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def main():
    a = parse_args()
    scene = bpy.context.scene
    cam = bpy.data.objects.get(a.camera)
    if cam is None:
        raise SystemExit("no camera " + a.camera)

    fm = set(o.name for o in find_mirrors(""))
    rows = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        rows.append((proj_area_frac(scene, cam, o), o.name,
                     "MIR" if _is_mirror_name(o.name) else "",
                     "GLOSS" if has_glossy(o) else "",
                     "<<find_mirrors" if o.name in fm else "",
                     len(o.data.vertices)))
    rows.sort(reverse=True)
    print("PROBE find_mirrors picked:", sorted(fm))
    print("PROBE %-26s %7s %4s %5s %5s %s" % ("name", "areaf%", "nameM", "gloss", "nvert", "picked"))
    for area, name, mir, gl, pick, nv in rows[:20]:
        print("PROBE %-26s %6.2f%% %4s %5s %5d %s" % (name[:26], area * 100, mir, gl, nv, pick))


if __name__ == "__main__":
    main()
