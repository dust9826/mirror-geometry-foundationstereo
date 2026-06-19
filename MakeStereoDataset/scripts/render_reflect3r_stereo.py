"""
render_reflect3r_stereo.py

Reflect3r Blender scene에서 physical stereo pair(left/right RGB) + mirror mask +
metadata를 생성한다. (MakeStereoDataset 노트북 cell 30의 texture-mirror 방식을
독립 스크립트로 분리한 버전.)

처리 순서:
  1. Mirrored_CM 카메라로 거울 내부에 보일 이미지를 렌더 -> _mirrored_cm_texture.png
  2. 그 이미지를 Mirror 오브젝트 material 에 Image Texture 로 직접 부착
  3. CM(left) 카메라로 left.png 렌더
  4. CM 에서 local +X 로 baseline 만큼 옮긴 Cam_Right_Physical 생성 -> right.png 렌더
  5. Mirror 기준 binary mask -> mirror_mask_left.png
  6. stereo_meta.json 저장 (baseline, 카메라 extrinsics 등)

실행 예 (Blender 4.5):
  blender -b scene.blend --python render_reflect3r_stereo.py -- \
      --out_dir OUT --camera CM --mirror_camera Mirrored_CM --mirror_object Mirror \
      --baseline 0.08 --width 960 --height 540 --samples 32 --device OPTIX
"""

import bpy
import os
import json
import argparse
from mathutils import Vector


def parse_args():
    import sys
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--camera", default="CM")
    p.add_argument("--mirror_camera", default="Mirrored_CM")
    # 빈 값이면 거울 메쉬를 자동 탐색한다(scene 마다 이름이 다르므로 기본 자동).
    # 콤마구분 이름을 주면 그것을 우선 사용.
    p.add_argument("--mirror_object", default="")
    p.add_argument("--baseline", type=float, default=0.08)
    # 1024x576: 정확히 16:9(scene 1920x1080 비율 유지) + 가로세로 모두 32배수라
    # FoundationStereo(32배수 패딩, max_disp=416) 입력으로 패딩 없이 들어간다.
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--samples", type=int, default=256)
    # RTX 카드에서는 OPTIX 가 가장 빠름. GPU가 없거나 문제가 있으면 CPU 로 폴백.
    p.add_argument("--device", default="OPTIX",
                   choices=["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "CPU"])
    # --- 누락 텍스처 / world 조명 자동 폴백 ---
    # scene 의 텍스처 경로가 원작자 로컬 경로로 박혀 있어 missing 되면 마젠타로 깨진다.
    # 누락 이미지를 회색으로, 누락 world env texture 를 단색 조명으로 자동 대체한다.
    p.add_argument("--no_fix_textures", dest="fix_textures", action="store_false",
                   help="누락 텍스처 회색 폴백 비활성화")
    p.add_argument("--no_fix_world", dest="fix_world", action="store_false",
                   help="누락 world env texture 단색 폴백 비활성화")
    p.add_argument("--tex_gray", type=float, default=0.5,
                   help="누락 텍스처 대체 회색 밝기 (0~1)")
    p.add_argument("--world_strength", type=float, default=1.0,
                   help="world 폴백 시 background 세기")
    p.add_argument("--world_gray", type=float, default=0.6,
                   help="world 폴백 시 background 회색 밝기 (0~1)")
    # 결과가 어두우면 노출을 올린다(stop 단위). None 이면 scene 원본 exposure 유지.
    p.add_argument("--exposure", type=float, default=None,
                   help="view exposure 강제 지정(stop). 미지정 시 scene 원본 유지")
    p.set_defaults(fix_textures=True, fix_world=True)
    return p.parse_args(argv)


def disable_compositor():
    s = bpy.context.scene
    s.use_nodes = False
    s.render.use_compositing = False
    s.render.use_sequencer = False


def setup(width, height, samples, device, exposure=None):
    s = bpy.context.scene
    disable_compositor()
    s.render.resolution_x = width
    s.render.resolution_y = height
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = "PNG"
    s.render.engine = "CYCLES"
    s.cycles.samples = samples
    s.cycles.use_denoising = True

    if device.upper() == "CPU":
        s.cycles.device = "CPU"
        print("Render device: CPU")
    else:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = device.upper()
        except Exception:
            prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        enabled = []
        for d in prefs.devices:
            d.use = d.type != "CPU"
            if d.use:
                enabled.append((d.name, d.type))
        s.cycles.device = "GPU"
        print("Cycles compute_device_type:", prefs.compute_device_type)
        print("Enabled render devices:", enabled)
        if not enabled:
            print("WARNING: no GPU device enabled -> falling back to CPU")
            s.cycles.device = "CPU"

    # 거울 반사가 또렷하게 나오도록 light path 의 glossy bounce 를 충분히 확보.
    try:
        s.cycles.max_bounces = max(s.cycles.max_bounces, 12)
        s.cycles.glossy_bounces = max(s.cycles.glossy_bounces, 8)
        s.cycles.transmission_bounces = max(s.cycles.transmission_bounces, 8)
        s.cycles.transparent_max_bounces = max(s.cycles.transparent_max_bounces, 8)
    except Exception as e:
        print("WARN bounce setup:", e)

    # scene 원본 룩(Filmic/High Contrast/exposure)을 그대로 유지한다.
    # 이전엔 Standard 로 강제 덮어써서 대비가 빠지고 밋밋(아싱)하게 나왔다.
    # view transform 이 비정상일 때만 Filmic 으로 안전 폴백.
    try:
        vt = s.view_settings.view_transform
        if vt not in ("Filmic", "AgX", "Standard"):
            s.view_settings.view_transform = "Filmic"
        if exposure is not None:
            s.view_settings.exposure = exposure
        print("view_transform:", s.view_settings.view_transform,
              "look:", s.view_settings.look,
              "exposure:", round(s.view_settings.exposure, 3))
    except Exception as e:
        print("WARN view settings:", e)


def render(cam, path):
    s = bpy.context.scene
    disable_compositor()
    s.camera = cam
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)


def _is_mirror_name(name):
    """거울 본체로 볼 이름인지. 'Mirror Wall'(프레임/뒷판)과
    'Mirrored_CM'(카메라)은 본체에서 제외한다."""
    low = name.lower()
    if "irror" not in low:
        return False
    if "mirrored_cm" in low:        # 가상 카메라
        return False
    if "wall" in low:               # 거울 벽/프레임 (본체 아님)
        return False
    return True


def _render_single_mask_gray(cam, target, tmp_path, w, h):
    """target 메쉬만 흰색으로 workbench 렌더 후 bool 마스크(np) 반환. PIL 불필요."""
    import numpy as np
    s = bpy.context.scene
    disable_compositor()
    orig_engine = s.render.engine
    orig_hide = {o.name: o.hide_render for o in bpy.data.objects}
    orig_mats = {o.name: [sl.material for sl in o.material_slots]
                 for o in bpy.data.objects if o.type == "MESH"}
    ox, oy = s.render.resolution_x, s.render.resolution_y
    s.render.resolution_x, s.render.resolution_y = w, h

    white = bpy.data.materials.new("CAL_W"); white.diffuse_color = (1, 1, 1, 1)
    black = bpy.data.materials.new("CAL_B"); black.diffuse_color = (0, 0, 0, 1)
    s.render.engine = "BLENDER_WORKBENCH"
    s.display.shading.light = "FLAT"
    s.display.shading.color_type = "MATERIAL"
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        o.hide_render = False
        o.data.materials.clear()
        o.data.materials.append(white if o is target else black)

    s.camera = cam
    s.render.filepath = tmp_path
    bpy.ops.render.render(write_still=True)

    img = bpy.data.images.load(tmp_path, check_existing=False)
    arr = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], -1)[::-1]
    g = arr[:, :, 0] > 0.5
    bpy.data.images.remove(img)

    # 복원
    for o in bpy.data.objects:
        o.hide_render = orig_hide[o.name]
        if o.type == "MESH":
            o.data.materials.clear()
            for mt in orig_mats.get(o.name, []):
                o.data.materials.append(mt)
    s.render.engine = orig_engine
    s.render.resolution_x, s.render.resolution_y = ox, oy
    return g


def _load_mask_gray(path, w, h):
    import numpy as np
    img = bpy.data.images.load(path, check_existing=False)
    arr = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], -1)[::-1]
    g = arr[:, :, 0]
    bpy.data.images.remove(img)
    if g.shape[1] != w or g.shape[0] != h:
        ys = np.linspace(0, g.shape[0] - 1, h).astype(int)
        xs = np.linspace(0, g.shape[1] - 1, w).astype(int)
        g = g[ys][:, xs]
    return g > 0.5


def calibrate_mirrors_from_reference(cam, ref_mask_path, tmp_path,
                                     iou_thresh=0.1, w=480, h=270):
    """reflect3r 의 정답 inside_mask 를 기준으로 진짜 거울 오브젝트를 식별한다.

    원본 CM 뷰에서 각 메쉬 단독 mask 를 reference 와 IoU 비교, IoU>=thresh 인
    메쉬들을 거울로 확정. 이름/재질로 안 잡히는 거울(예: minigym 의 Cube.001~003)도
    정확히 찾는다. 반환: (mirror objects 리스트, [(name, iou, cov), ...] 디버그).
    """
    import numpy as np
    ref = _load_mask_gray(ref_mask_path, w, h)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    scored = []
    for o in meshes:
        m = _render_single_mask_gray(cam, o, tmp_path, w, h)
        if m.mean() < 0.0005:        # 화면에 거의 안 보이면 skip
            continue
        u = (m | ref).sum()
        iou = float((m & ref).sum()) / float(u) if u else 0.0
        if iou >= iou_thresh:
            scored.append((o, iou, float(m.mean() * 100)))
    scored.sort(key=lambda r: -r[1])
    mirrors = [o for (o, i, c) in scored]
    dbg = [(o.name, round(i, 3), round(c, 1)) for (o, i, c) in scored]
    return mirrors, dbg


def find_mirrors(requested=""):
    """거울 본체 메쉬들을 찾는다.

    1) --mirror_object 로 콤마구분 이름이 명시되면 그 중 존재하는 것
    2) 없으면 이름에 'mirror' 가 든 메쉬(Wall/카메라 제외) 자동 탐색
    3) 그래도 없으면 glossy(roughness 낮음) 재질을 가진 메쉬를 거울로 추정

    참고: 이름/재질로 안 잡히는 거울(minigym 의 Cube.xxx 등)은
    calibrate_mirrors_from_reference() 로 reference inside_mask 기준 식별이 정확.
    반환: 거울 메쉬 오브젝트 리스트(없으면 빈 리스트).
    """
    # 1) 명시적 이름
    names = [x.strip() for x in (requested or "").split(",") if x.strip()]
    found = [bpy.data.objects[n] for n in names
             if n in bpy.data.objects and bpy.data.objects[n].type == "MESH"]
    if found:
        return found

    # 2) 이름 기반 자동 탐색
    found = [o for o in bpy.data.objects
             if o.type == "MESH" and _is_mirror_name(o.name)]
    if found:
        return found

    # 3) glossy 재질 기반 추정 (이름에 mirror 가 전혀 없는 scene 대비)
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for sl in o.material_slots:
            m = sl.material
            if not (m and m.use_nodes):
                continue
            for n in m.node_tree.nodes:
                if n.type == "BSDF_GLOSSY":
                    try:
                        rough = n.inputs["Roughness"].default_value
                    except Exception:
                        rough = 0.0
                    if rough <= 0.1:
                        found.append(o)
                        break
            else:
                continue
            break
    return found


def enable_physical_mirror(mirrors):
    """거울(들)을 물리(레이트레이싱) 반사로 켠다.

    scene 의 거울 메쉬에는 보통 BSDF_GLOSSY(진짜 거울) 재질이 있는데
    hide_render=True 로 숨겨져 있다. 이 본체들 + 프레임('Mirror Wall')을
    렌더에 켜면 Cycles 가 방을 정확히 반사한다.
    glossy 재질이 없는 본체는 roughness 0 거울 재질로 보정한다.
    """
    mirror_set = set(mirrors)
    enabled = []
    # 거울 본체 + 프레임/뒷판('irror' 이름) 메쉬를 모두 렌더에 켠다.
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        if o in mirror_set or "irror" in o.name.lower():
            o.hide_render = False
            o.hide_viewport = False
            enabled.append(o.name)

    # 각 거울 본체가 glossy 인지 확인, 아니면 깨끗한 거울 재질로 보정
    for mirror_obj in mirrors:
        has_glossy = any(
            n.type == "BSDF_GLOSSY"
            for sl in mirror_obj.material_slots if sl.material and sl.material.use_nodes
            for n in sl.material.node_tree.nodes
        )
        # glossy 의 roughness 를 0 으로 낮춰 또렷한 반사 보장
        for sl in mirror_obj.material_slots:
            m = sl.material
            if m and m.use_nodes:
                for n in m.node_tree.nodes:
                    if n.type == "BSDF_GLOSSY":
                        try:
                            n.inputs["Roughness"].default_value = 0.0
                        except Exception:
                            pass

        if not has_glossy:
            mat = bpy.data.materials.new("Physical_Mirror")
            mat.use_nodes = True
            nt = mat.node_tree
            nt.nodes.clear()
            g = nt.nodes.new("ShaderNodeBsdfGlossy")
            out = nt.nodes.new("ShaderNodeOutputMaterial")
            try:
                g.inputs["Roughness"].default_value = 0.0
                g.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1.0)
            except Exception:
                pass
            nt.links.new(g.outputs["BSDF"], out.inputs["Surface"])
            mirror_obj.data.materials.clear()
            mirror_obj.data.materials.append(mat)
            print("physical mirror: glossy 재질 새로 생성 ->", mirror_obj.name)

    print("physical mirror enabled on:", enabled)


def make_right(left, baseline):
    old = bpy.data.objects.get("Cam_Right_Physical")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    right = left.copy()
    right.data = left.data.copy()
    right.name = "Cam_Right_Physical"
    bpy.context.collection.objects.link(right)

    offset = left.matrix_world.to_quaternion() @ Vector((baseline, 0, 0))
    right.location = left.location + offset
    right.rotation_euler = left.rotation_euler
    return right


def render_mask(cam, mirrors, path):
    s = bpy.context.scene
    disable_compositor()

    mirror_set = set(mirrors)
    orig_engine = s.render.engine
    orig_mats = {}
    orig_hide = {}
    for o in bpy.data.objects:
        orig_hide[o.name] = o.hide_render
        if o.type == "MESH":
            orig_mats[o.name] = [slot.material for slot in o.material_slots]

    white = bpy.data.materials.new("MASK_WHITE")
    white.diffuse_color = (1, 1, 1, 1)
    black = bpy.data.materials.new("MASK_BLACK")
    black.diffuse_color = (0, 0, 0, 1)

    s.render.engine = "BLENDER_WORKBENCH"
    s.display.shading.light = "FLAT"
    s.display.shading.color_type = "MATERIAL"

    # 거울 본체(들)만 흰색, 나머지(프레임 'Mirror Wall' 포함)는 검정.
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        o.hide_render = False
        o.data.materials.clear()
        o.data.materials.append(white if o in mirror_set else black)

    render(cam, path)

    for o in bpy.data.objects:
        o.hide_render = orig_hide[o.name]
        if o.type == "MESH":
            o.data.materials.clear()
            for m in orig_mats.get(o.name, []):
                o.data.materials.append(m)

    s.render.engine = orig_engine


def mat4(m):
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


def _image_missing(img):
    """이미지 데이터블록이 실제로 로드되지 않았는지(=missing) 판정."""
    if img is None:
        return False
    # packed image 는 파일이 없어도 데이터가 있으면 OK
    if getattr(img, "packed_file", None) is not None:
        return False
    # 픽셀 크기가 0 이면 로드 실패로 본다.
    try:
        if tuple(img.size) != (0, 0):
            return False
    except Exception:
        pass
    # 파일 경로가 실제로 존재하면 missing 아님
    try:
        path = bpy.path.abspath(img.filepath)
        if path and os.path.exists(path):
            return False
    except Exception:
        pass
    return True


def fix_missing_textures(gray=0.5):
    """누락된 image texture 를 회색 단색으로 대체한다.

    Blender 는 누락 텍스처를 마젠타로 표시하므로, 누락 이미지를 1x1 회색
    generated 이미지로 바꿔 마젠타를 없앤다. 반환: (총 이미지, 대체 개수)
    """
    total = 0
    fixed = 0
    for img in list(bpy.data.images):
        # 내부용/렌더결과 이미지는 건드리지 않음
        if img.type in {"RENDER_RESULT", "COMPOSITING"}:
            continue
        total += 1
        if not _image_missing(img):
            continue
        name = img.name
        path = img.filepath
        # 회색 generated 이미지로 교체 (기존 데이터블록을 in-place 로 치환)
        try:
            img.source = "GENERATED"
            img.generated_type = "BLANK"
            img.generated_width = 16
            img.generated_height = 16
            img.generated_color = (gray, gray, gray, 1.0)
            fixed += 1
            print("  fixed missing texture:", name, "<-", path)
        except Exception as e:
            print("  WARN could not fix:", name, e)
    print("fix_missing_textures: %d/%d replaced with gray %.2f" % (fixed, total, gray))
    return total, fixed


def _find_world_hdri():
    """설치된 Blender 의 world HDRI 경로를 찾는다.

    scene 이 원래 쓰던 studio.exr 을 1순위로(원작자 의도), 없으면 interior.exr,
    그 외 실내성 HDRI 순으로 고른다. bpy 바이너리 위치에서 datafiles 를 역추적.
    """
    import glob as _glob
    prefer = ["studio.exr", "interior.exr", "courtyard.exr", "city.exr"]
    # blender.exe -> .../Blender 5.1/ , 그 아래 <ver>/datafiles/studiolights/world
    base = os.path.dirname(bpy.app.binary_path)
    cands = _glob.glob(os.path.join(base, "**", "studiolights", "world", "*.exr"),
                       recursive=True)
    by_name = {os.path.basename(p).lower(): p for p in cands}
    for name in prefer:
        if name in by_name:
            return by_name[name]
    return cands[0] if cands else None


def fix_world(strength=1.0, gray=0.6):
    """world 의 environment texture 가 누락됐으면 단색 background 로 대체.

    cozy_living_room 등은 world 가 존재하지 않는 Blender 4.3 경로의 studio.exr 을
    참조해 환경광이 빠진다. env texture 노드가 누락이면 Background 만 남기고
    단색 조명을 준다. world 자체가 없으면 새로 만든다.
    """
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("FallbackWorld")
        scene.world = world
        print("fix_world: scene 에 world 가 없어 새로 생성")

    world.use_nodes = True
    nt = world.node_tree

    # 누락된 env texture 노드를 설치된 Blender 의 실내 HDRI 로 재연결한다.
    # scene 이 참조하던 studio.exr 은 없는 Blender 4.3 경로라서 깨졌을 뿐,
    # 같은 HDRI 가 현재 Blender 의 datafiles 에 들어 있다. 이를 다시 물려주면
    # 회색 평면보다 훨씬 부드럽고 따뜻한 환경광(참조 이미지 느낌)이 살아난다.
    hdri_path = _find_world_hdri()
    fixed = 0
    for node in nt.nodes:
        if node.type != "TEX_ENVIRONMENT":
            continue
        if node.image is not None and not _image_missing(node.image):
            continue
        if hdri_path:
            node.image = bpy.data.images.load(hdri_path, check_existing=True)
        else:
            # HDRI 를 못 찾으면 회색 평면으로 폴백
            img = bpy.data.images.new("WorldEnvFallback", 16, 16)
            img.generated_type = "BLANK"
            img.generated_color = (gray, gray, gray, 1.0)
            node.image = img
        fixed += 1

    if fixed == 0:
        print("fix_world: world env texture 정상 -> 그대로 둠")
        return False
    src = os.path.basename(hdri_path) if hdri_path else "gray %.2f" % gray
    print("fix_world: 누락 env texture %d개 -> %s 재연결 (원본 graph/strength 유지)"
          % (fixed, src))
    return True


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    setup(a.width, a.height, a.samples, a.device, a.exposure)

    # --- 누락 텍스처 / world 조명 자동 폴백 ---
    world_fixed = False
    tex_total = tex_fixed = 0
    if a.fix_world:
        world_fixed = fix_world(a.world_strength, a.world_gray)
    if a.fix_textures:
        tex_total, tex_fixed = fix_missing_textures(a.tex_gray)

    left = bpy.data.objects[a.camera]

    # 거울 본체 자동 탐색 (scene 마다 이름이 Mirror / Mirror1~4 / Mirror_Plane 등으로 다름)
    mirrors = find_mirrors(a.mirror_object)
    if not mirrors:
        raise RuntimeError("거울 메쉬를 찾지 못함 (이 scene 엔 거울이 없을 수 있음): "
                           + a.out_dir)
    print("mirrors found:", [m.name for m in mirrors])

    # 물리(레이트레이싱) 거울로 켠다 -> Cycles 가 방을 정확히 반사 (참조 이미지 방식)
    enable_physical_mirror(mirrors)

    right = make_right(left, a.baseline)

    render(left, os.path.join(a.out_dir, "left.png"))
    render(right, os.path.join(a.out_dir, "right.png"))
    render_mask(left, mirrors, os.path.join(a.out_dir, "mirror_mask_left.png"))

    with open(os.path.join(a.out_dir, "stereo_meta.json"), "w") as f:
        json.dump({
            "left_camera": left.name,
            "right_camera": right.name,
            "mirror_objects": [m.name for m in mirrors],
            "mirror_mode": "physical_raytraced",
            "baseline_m": a.baseline,
            "left_camera_to_world": mat4(left.matrix_world),
            "right_camera_to_world": mat4(right.matrix_world),
            "fallback": {
                "world_replaced": world_fixed,
                "textures_total": tex_total,
                "textures_replaced": tex_fixed,
            },
        }, f, indent=2)

    print("DONE", a.out_dir)


if __name__ == "__main__":
    main()
