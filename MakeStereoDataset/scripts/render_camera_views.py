"""
render_camera_views.py  (카메라 이동 데이터 증강)

한 scene 에서 카메라를 거울이 잘 보이는 여러 위치로 옮겨가며 stereo pair 를
여러 장 렌더한다. 같은 scene 으로 다수의 (left/right/mask/meta) 샘플을 확보한다.

동작:
  1. 거울 본체 자동 탐색(render_reflect3r_stereo.find_mirrors) → 중심 C, 법선 n 계산
  2. 거울 '앞쪽'(방 안쪽) 반구에서 카메라 포즈를 여러 개 샘플
     (거리 × 방위각 × 고도각), 전부 거울 중심 C 를 바라보게 함
  3. 빠른 mask 스카우트(BLENDER_WORKBENCH, path tracing 없음, 저해상도)로
     각 포즈의 거울 포함률(coverage %)을 측정
  4. coverage 가 [min_cov, max_cov] 인 포즈 중 다양하게 top-K 선정
  5. 선정된 포즈만 풀 Cycles stereo 렌더(left/right) + mask + meta 저장

scene 별 세팅(거울/조명 폴백, 물리거울, Filmic 룩)은 render_reflect3r_stereo 와 동일.

실행 예:
  blender -b minigym.blend --python render_camera_views.py -- \
      --out_dir OUT/minigym_views --camera CM \
      --baseline 0.08 --width 1024 --height 576 --samples 256 \
      --n_scout 45 --n_keep 6 --min_cov 8 --max_cov 55
"""

import bpy
import os
import sys
import json
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import render_reflect3r_stereo as R  # noqa: E402
from mathutils import Vector, Matrix  # noqa: E402

try:
    import numpy as np
except Exception:
    np = None


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--camera", default="CM")
    p.add_argument("--mirror_object", default="")
    # reflect3r 정답 inside_mask 경로. 주면 이걸로 거울 오브젝트를 자동 calibration.
    p.add_argument("--ref_mask", default="")
    p.add_argument("--baseline", type=float, default=0.08)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--device", default="OPTIX",
                   choices=["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "CPU"])
    p.add_argument("--exposure", type=float, default=None)
    # 포즈 샘플링
    p.add_argument("--n_scout", type=int, default=45, help="스카우트할 후보 포즈 수(대략)")
    p.add_argument("--n_keep", type=int, default=6, help="풀 렌더할 포즈 수")
    p.add_argument("--min_cov", type=float, default=8.0, help="거울 포함률 하한(%)")
    p.add_argument("--max_cov", type=float, default=55.0, help="거울 포함률 상한(%)")
    p.add_argument("--max_black", type=float, default=8.0,
                   help="검정 가림 비율 상한(%). 넘으면 못 쓰는 view 로 제외")
    p.add_argument("--dists", default="1.8,2.6,3.4", help="거울로부터 거리(m) 콤마구분")
    p.add_argument("--scout_w", type=int, default=320, help="스카우트 mask 해상도 가로")
    p.add_argument("--scout_h", type=int, default=180)
    return p.parse_args(argv)


def mirror_center_normal(mirrors):
    """거울 메쉬들의 월드 bbox 중심 C 와 평균 법선 n(단위) 반환."""
    pts = []
    nrm = Vector((0, 0, 0))
    for o in mirrors:
        for c in o.bound_box:
            pts.append(o.matrix_world @ Vector(c))
        # 폴리곤 법선 평균(월드)
        wn = o.matrix_world.to_3x3()
        for poly in o.data.polygons:
            nrm += (wn @ poly.normal)
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    cz = sum(p.z for p in pts) / len(pts)
    C = Vector((cx, cy, cz))
    if nrm.length < 1e-6:
        nrm = Vector((1, 0, 0))
    n = nrm.normalized()
    return C, n


def look_at_euler(loc, target):
    """loc 에서 target 을 바라보는 카메라 회전(euler) 반환."""
    direction = (target - loc)
    quat = direction.to_track_quat('-Z', 'Y')
    return quat.to_euler()


def set_camera_pose(cam, loc, target):
    cam.location = loc
    cam.rotation_euler = look_at_euler(loc, target)
    bpy.context.view_layer.update()


def sample_poses(C, n_front, dists, azimuths, elevations):
    """거울 앞(n_front 방향) 반구에서 카메라 위치 후보 생성.

    각 (dist, az, el) 에 대해 n_front 를 월드 Z 축으로 az 회전 후 수평축으로 el 틸트,
    그 방향으로 dist 만큼 떨어진 위치. 전부 C 를 바라본다.
    """
    up = Vector((0, 0, 1))
    poses = []
    for dist in dists:
        for az in azimuths:
            Rz = Matrix.Rotation(math.radians(az), 4, 'Z')
            dir_h = (Rz @ n_front).normalized()
            # 수평축 = dir_h × up (고도 틸트용)
            right = dir_h.cross(up)
            if right.length < 1e-6:
                right = Vector((1, 0, 0))
            right.normalize()
            for el in elevations:
                Rel = Matrix.Rotation(math.radians(el), 4, right)
                d = (Rel @ dir_h).normalized()
                loc = C + d * dist
                poses.append((loc.copy(), dist, az, el))
    return poses


def perturb_poses(base_loc, target, n_az, n_el, n_rad,
                  az_max=22.0, el_max=14.0, rad_frac=0.28):
    """원본 카메라 위치(base_loc)를 중심으로, 거울중심(target) 둘레로
    방위각/고도각/반경을 흔들어 다양한 카메라 위치를 만든다.

    원본 CM 이 이미 거울을 잘 보고 있으므로(예: minigym 38%),
    그 주변을 도는 게 거울 포함을 유지하면서 시점만 다양화하는 가장 안전한 방법.
    전부 target(거울 중심)을 바라본다.
    """
    v = base_loc - target
    radius = v.length
    if radius < 1e-6:
        radius = 1.0
    up = Vector((0, 0, 1))
    poses = []
    azs = ([0.0] if n_az <= 1 else
           [(-az_max + 2 * az_max * i / (n_az - 1)) for i in range(n_az)])
    els = ([0.0] if n_el <= 1 else
           [(-el_max + 2 * el_max * i / (n_el - 1)) for i in range(n_el)])
    rads = ([1.0] if n_rad <= 1 else
            [(1 - rad_frac + 2 * rad_frac * i / (n_rad - 1)) for i in range(n_rad)])
    for rmul in rads:
        for az in azs:
            Rz = Matrix.Rotation(math.radians(az), 4, 'Z')
            vz = Rz @ v
            right = vz.cross(up)
            if right.length < 1e-6:
                right = Vector((1, 0, 0))
            right.normalize()
            for el in els:
                Rel = Matrix.Rotation(math.radians(el), 4, right)
                d = (Rel @ vz).normalized()
                loc = target + d * (radius * rmul)
                poses.append((loc.copy(), round(radius * rmul, 2), az, el))
    return poses


def mask_coverage_fast(cam, mirrors, tmp_path, w, h):
    """저해상도 workbench mask 를 렌더하고 흰색 비율(%)을 반환."""
    s = bpy.context.scene
    ox, oy = s.render.resolution_x, s.render.resolution_y
    s.render.resolution_x, s.render.resolution_y = w, h
    R.render_mask(cam, mirrors, tmp_path)
    s.render.resolution_x, s.render.resolution_y = ox, oy

    img = bpy.data.images.load(tmp_path, check_existing=False)
    px = np.array(img.pixels[:]) if np is not None else None
    cov = 0.0
    if px is not None and px.size:
        px = px.reshape(-1, 4)
        white = (px[:, 0] > 0.5)
        cov = float(white.mean() * 100.0)
    bpy.data.images.remove(img)
    return cov


def black_ratio_fast(cam, tmp_path, w, h):
    """저해상도 RGB(EEVEE)를 빠르게 렌더해 '거의 순수 검정' 픽셀 비율(%)을 반환.

    카메라가 거울/벽 뒤로 가거나 극단 각도가 되면 화면 큰 영역이 검정으로 가려져
    데이터셋으로 못 쓰게 된다. 이를 거르기 위한 빠른 품질 지표.
    (Cycles 풀 렌더 대신 EEVEE 저해상으로 수초 내 측정)
    """
    s = bpy.context.scene
    ox, oy = s.render.resolution_x, s.render.resolution_y
    oeng = s.render.engine
    R.disable_compositor()
    s.render.resolution_x, s.render.resolution_y = w, h
    # EEVEE 로 빠르게 (버전별 엔진 id 호환 처리)
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            s.render.engine = eng
            break
        except Exception:
            continue
    s.camera = cam
    s.render.filepath = tmp_path
    try:
        bpy.ops.render.render(write_still=True)
    except Exception:
        s.render.engine = oeng
        s.render.resolution_x, s.render.resolution_y = ox, oy
        return 0.0

    br = 0.0
    img = bpy.data.images.load(tmp_path, check_existing=False)
    px = np.array(img.pixels[:]) if np is not None else None
    if px is not None and px.size:
        px = px.reshape(-1, 4)[:, :3]
        black = (px.max(axis=1) < 0.05)
        br = float(black.mean() * 100.0)
    bpy.data.images.remove(img)

    s.render.engine = oeng
    s.render.resolution_x, s.render.resolution_y = ox, oy
    return br


def pick_diverse(scout, n_keep, min_cov, max_cov, max_black):
    """coverage 가 범위 안이고 black 비율이 낮은(쓸 수 있는) 포즈를
    coverage 기준으로 고르게 n_keep 개 선정."""
    usable = [s for s in scout
              if min_cov <= s["coverage"] <= max_cov
              and s.get("black", 0.0) <= max_black]
    if not usable:
        # 쓸 수 있는 게 없으면 coverage 범위만이라도(없으면 그냥 black 낮은 순)
        usable = [s for s in scout if min_cov <= s["coverage"] <= max_cov]
        if not usable:
            return sorted(scout, key=lambda s: s.get("black", 100))[:n_keep]
    usable.sort(key=lambda s: s["coverage"])
    if len(usable) <= n_keep:
        return usable
    idx = [round(i * (len(usable) - 1) / (n_keep - 1)) for i in range(n_keep)]
    return [usable[i] for i in sorted(set(idx))]


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    R.setup(a.width, a.height, a.samples, a.device, a.exposure)
    world_fixed = R.fix_world() if True else False
    tex_total, tex_fixed = R.fix_missing_textures()

    cam = bpy.data.objects[a.camera]

    # 거울 식별: reference inside_mask 가 있으면 calibration(가장 정확),
    # 없으면 이름/재질 자동탐색으로 폴백.
    mirrors = []
    if a.ref_mask and os.path.exists(a.ref_mask):
        tmp = os.path.join(a.out_dir, "_calib_tmp.png")
        mirrors, dbg = R.calibrate_mirrors_from_reference(cam, a.ref_mask, tmp)
        if os.path.exists(tmp):
            os.remove(tmp)
        print("calibration (IoU vs ref):", dbg)
    if not mirrors:
        mirrors = R.find_mirrors(a.mirror_object)
    if not mirrors:
        raise RuntimeError("거울 메쉬를 찾지 못함: " + a.out_dir)
    print("mirrors found:", [m.name for m in mirrors])

    R.enable_physical_mirror(mirrors)

    C, n = mirror_center_normal(mirrors)
    # '앞쪽'(방 안쪽) = 원래 CM 이 있는 쪽
    if (cam.location - C).dot(n) < 0:
        n = -n
    print("mirror center:", tuple(round(v, 2) for v in C),
          "front normal:", tuple(round(v, 2) for v in n))

    # 원본 CM 포즈(거울을 이미 잘 봄)를 중심으로 perturb 하여 후보 생성
    base_loc = cam.location.copy()
    poses = perturb_poses(base_loc, C, n_az=5, n_el=3, n_rad=3)
    print("candidate poses:", len(poses), "base_loc:",
          tuple(round(v, 2) for v in base_loc))

    # --- 스카우트: 각 포즈의 거울 coverage 측정 (빠른 저해상 mask) ---
    tmp = os.path.join(a.out_dir, "_scout_mask.png")
    scout = []
    tmp_rgb = os.path.join(a.out_dir, "_scout_rgb.png")
    for i, (loc, dist, az, el) in enumerate(poses):
        set_camera_pose(cam, loc, C)
        cov = mask_coverage_fast(cam, mirrors, tmp, a.scout_w, a.scout_h)
        # coverage 범위 밖이면 black 측정 생략(렌더 절약)
        blk = 0.0
        if a.min_cov <= cov <= a.max_cov:
            blk = black_ratio_fast(cam, tmp_rgb, a.scout_w, a.scout_h)
        scout.append({"i": i, "loc": list(loc), "dist": dist, "az": az,
                      "el": el, "coverage": cov, "black": blk})
        print("  scout %02d dist=%.1f az=%d el=%d -> coverage=%.1f%% black=%.1f%%"
              % (i, dist, az, el, cov, blk))
    for f in (tmp, tmp_rgb):
        if os.path.exists(f):
            os.remove(f)

    kept = pick_diverse(scout, a.n_keep, a.min_cov, a.max_cov, a.max_black)
    print("kept poses (cov,black):",
          [(round(k["coverage"], 1), round(k.get("black", 0), 1)) for k in kept])

    # --- 선정 포즈 풀 렌더 ---
    index = []
    for vi, k in enumerate(kept):
        loc = Vector(k["loc"])
        set_camera_pose(cam, loc, C)
        vdir = os.path.join(a.out_dir, "view_%02d" % vi)
        os.makedirs(vdir, exist_ok=True)

        right = R.make_right(cam, a.baseline)
        left_path = os.path.join(vdir, "left.png")
        R.render(cam, left_path)
        R.render(right, os.path.join(vdir, "right.png"))
        R.render_mask(cam, mirrors, os.path.join(vdir, "mirror_mask_left.png"))

        # 최종 left.png 의 실제 검정 비율을 재측정(스카우트는 EEVEE 근사라 확인)
        black_final = 0.0
        try:
            img = bpy.data.images.load(left_path, check_existing=False)
            px = np.array(img.pixels[:]).reshape(-1, 4)[:, :3] if np is not None else None
            if px is not None and px.size:
                black_final = float((px.max(axis=1) < 0.05).mean() * 100.0)
            bpy.data.images.remove(img)
        except Exception:
            pass
        usable = black_final <= a.max_black

        meta = {
            "view": vi,
            "left_camera": cam.name,
            "right_camera": right.name,
            "mirror_objects": [m.name for m in mirrors],
            "mirror_mode": "physical_raytraced",
            "baseline_m": a.baseline,
            "scout_coverage_pct": k["coverage"],
            "scout_black_pct": k.get("black", 0.0),
            "black_pct_final": black_final,
            "usable": usable,
            "pose": {"dist": k["dist"], "az": k["az"], "el": k["el"],
                     "loc": list(loc)},
            "left_camera_to_world": R.mat4(cam.matrix_world),
            "right_camera_to_world": R.mat4(right.matrix_world),
            "fallback": {"world_replaced": world_fixed,
                         "textures_total": tex_total,
                         "textures_replaced": tex_fixed},
        }
        with open(os.path.join(vdir, "stereo_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        index.append({"view": vi, "dir": "view_%02d" % vi,
                      "coverage_pct": k["coverage"],
                      "black_pct_final": black_final, "usable": usable,
                      "pose": meta["pose"]})
        print("RENDERED view_%02d coverage~%.1f%% black=%.1f%% usable=%s"
              % (vi, k["coverage"], black_final, usable))

    with open(os.path.join(a.out_dir, "views_index.json"), "w") as f:
        json.dump({"scene_out": a.out_dir, "n_views": len(index),
                   "views": index}, f, indent=2)
    print("DONE views:", len(index), "->", a.out_dir)


if __name__ == "__main__":
    main()
