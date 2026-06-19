"""rerender_mask.py — (Blender 전용) 거울 마스크만 다시 렌더한다.

빈/이상 마스크 scene 을 위해, 현재 find_mirrors 로 거울 본체를 찾아
render_mask(workbench, 거울=흰/나머지=검) 로 마스크 PNG 만 빠르게 재생성한다.
전체 물리렌더 불필요(가림은 처리됨).

실행:
  blender --background scene.blend --python rerender_mask.py -- \
      --out mirror_mask_left.png --camera CM --width 1024 --height 576
"""
import argparse
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_reflect3r_stereo import find_mirrors, render_mask


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="CM")
    p.add_argument("--mirror_object", default="")
    p.add_argument("--glossy", action="store_true",
                   help="find_mirrors 대신 저거칠기 glossy/metallic 메쉬 전체를 거울로 사용")
    p.add_argument("--rough_max", type=float, default=0.15)
    p.add_argument("--hide_others", action="store_true")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    return p.parse_args(argv)


def glossy_mirrors(rough_max):
    """저거칠기 반사 재질 메쉬 전체. BSDF_GLOSSY 또는 Principled(metallic>=0.5,rough<=th)."""
    out = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        hit = False
        for sl in o.material_slots:
            m = sl.material
            if not (m and m.use_nodes):
                continue
            for n in m.node_tree.nodes:
                try:
                    if n.type == "BSDF_GLOSSY" and n.inputs["Roughness"].default_value <= rough_max:
                        hit = True
                    elif n.type == "BSDF_PRINCIPLED" and \
                            n.inputs["Metallic"].default_value >= 0.5 and \
                            n.inputs["Roughness"].default_value <= rough_max:
                        hit = True
                except Exception:
                    pass
            if hit:
                break
        if hit:
            out.append(o)
    return out


def main():
    a = parse_args()
    s = bpy.context.scene
    s.render.resolution_x = a.width
    s.render.resolution_y = a.height
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = "PNG"   # scene 기본이 EXR 등일 수 있어 명시
    s.render.image_settings.color_mode = "RGB"

    cam = bpy.data.objects.get(a.camera)
    if cam is None:
        raise SystemExit(f"카메라 '{a.camera}' 없음. 카메라들: "
                         + str([o.name for o in bpy.data.objects if o.type == 'CAMERA']))
    s.camera = cam

    if a.glossy:
        mirrors = glossy_mirrors(a.rough_max)
    else:
        mirrors = find_mirrors(a.mirror_object)
    if not mirrors:
        raise SystemExit("거울 메쉬를 못 찾음")
    print("mirrors:", [m.name for m in mirrors])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.hide_others:
        # 가림 제거 진단용: 거울만 보이게(나머지 hide_render) 흰색 렌더.
        from render_reflect3r_stereo import disable_compositor
        disable_compositor()
        mset = set(mirrors)
        s.render.engine = "BLENDER_WORKBENCH"
        s.display.shading.light = "FLAT"; s.display.shading.color_type = "MATERIAL"
        white = bpy.data.materials.new("HW"); white.diffuse_color = (1, 1, 1, 1)
        for o in bpy.data.objects:
            if o.type == "MESH":
                o.hide_render = (o not in mset)
                if o in mset:
                    o.data.materials.clear(); o.data.materials.append(white)
        s.render.filepath = a.out
        bpy.ops.render.render(write_still=True)
    else:
        render_mask(cam, mirrors, a.out)
    print("WROTE_MASK", a.out, [m.name for m in mirrors])


if __name__ == "__main__":
    main()
