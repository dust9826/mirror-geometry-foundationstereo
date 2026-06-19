"""make_mirror_gt.py — GT 거울평면을 left-camera OpenCV 좌표로 추출/검증.

1) Blender(export_mirror_gt.py) 를 subprocess 로 호출해 world 평면 추출
2) world -> left-camera OpenCV 변환 (축 플립 F=diag(1,-1,-1))
3) experiments/<scene>/mirror_gt/gt.json + overlay.png 저장

사용:
  python scripts/make_mirror_gt.py --scene computer_room_baseline080
  python scripts/make_mirror_gt.py --all
  python scripts/make_mirror_gt.py --scene X --no-blender   # 기존 world json 재변환
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C


# --------------------------------------------------------------- 순수 변환 함수
def world_plane_to_cam(normal_w, centroid_w, corners_w, cam_to_world):
    """Blender world 평면을 left-camera OpenCV 좌표로 변환.

    normal_w     : (3,) world 단위 법선
    centroid_w   : (3,) world 평면 중심
    corners_w    : (4,3) world 평면 모서리
    cam_to_world : (4,4) Blender 카메라 matrix_world (cam->world, -Z 전방)
    반환 dict(normal(3,), d(float), centroid(3,), corners(4,3))  — OpenCV cam 좌표.
    """
    cam2world = np.asarray(cam_to_world, dtype=np.float64).reshape(4, 4)
    F = np.diag([1.0, -1.0, -1.0, 1.0])      # Blender(-Z전방/+Y상) → OpenCV(+Z전방/+Y하)
    M = F @ np.linalg.inv(cam2world)          # world → OpenCV 카메라
    R = M[:3, :3]

    def xf_pt(p):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        return (M @ np.array([p[0], p[1], p[2], 1.0]))[:3]

    raw_n = R @ np.asarray(normal_w, dtype=np.float64).reshape(3)
    norm_n = np.linalg.norm(raw_n)
    if norm_n < 1e-9:
        raise ValueError(f"Degenerate normal after transform: {raw_n}")
    n = raw_n / norm_n
    c = xf_pt(centroid_w)
    corners = np.array([xf_pt(p) for p in np.asarray(corners_w, float).reshape(-1, 3)])
    d = -float(n @ c)
    return {"normal": n, "d": d, "centroid": c, "corners": corners}


def project_points(K, pts):
    """(N,3) OpenCV 카메라좌표 점 → (N,2) 픽셀 (z>0 가정, z 클램프)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    z = np.clip(pts[:, 2], 1e-6, None)
    u = K[0, 0] * pts[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def blend_base(scene):
    """experiments scene 이름 → .blend base 이름 (baseline 접미사 제거)."""
    suffix = "_baseline080"
    return scene[:-len(suffix)] if scene.endswith(suffix) else scene


# --------------------------------------------------------------- 경로 상수
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
_MSD = os.path.normpath(os.path.join(C.PROJECT_ROOT, "..", "MakeStereoDataset"))
BLEND_ROOT = os.path.join(_MSD, "data", "source_scenes", "blender_source_files")
EXPORT_SCRIPT = os.path.join(_MSD, "scripts", "export_mirror_gt.py")

# scene 별 예외(find_mirrors/이름규칙 보정).
#   blend         : blender_source_files 기준 상대경로 (폴더/이름 불일치, view scene 등)
#   mirror_object : 거울 객체 콤마목록(공백 포함 이름 가능). 없으면 prepare_meta.mirror_objects 사용.
SCENE_OVERRIDES = {
    "bedroom_baseline080": {
        "mirror_object": "Wardrobe_04_001,Wardrobe_04_001.001,Wardrobe_04_001.002,"
                         "Wardrobe_04_001.003,Wardrobe_04_001.004"},
    "living_room_baseline080": {
        "blend": os.path.join("livingRoom_contemporary", "living_room.blend"),
        "mirror_object": "MirrorL,MirrorM,MirrorR"},
    "minigym_view04": {"blend": "minigym.blend"},   # 거울은 prepare_meta(Cube.001~003) 자동 사용
}


def mirror_object_for(scene, meta):
    """거울 객체 콤마목록: override > prepare_meta.mirror_objects > '' (find_mirrors 자동).

    데이터셋이 실제 렌더에 쓴 거울(prepare_meta.mirror_objects)을 우선 써서 GT 와 마스크의
    거울을 일치시킨다. 원본 렌더가 잘못 잡은 scene(예: bedroom)만 override 로 교정.
    """
    ov = SCENE_OVERRIDES.get(scene, {})
    if ov.get("mirror_object"):
        return ov["mirror_object"]
    mo = meta.get("mirror_objects")
    if isinstance(mo, (list, tuple)) and mo:
        return ",".join(str(x) for x in mo)
    if isinstance(mo, str):
        return mo
    return ""


def resolve_blend(scene):
    """scene → .blend 경로 (없으면 None). override > 이름규칙 > 폴더 탐색."""
    ov = SCENE_OVERRIDES.get(scene, {})
    if ov.get("blend"):
        cand = os.path.join(BLEND_ROOT, ov["blend"])
        return cand if os.path.isfile(cand) else None
    base = blend_base(scene)
    cand = os.path.join(BLEND_ROOT, base + ".blend")
    if os.path.isfile(cand):
        return cand
    d = os.path.join(BLEND_ROOT, base)
    if os.path.isdir(d):
        blends = sorted(glob.glob(os.path.join(d, "*.blend")))
        if len(blends) > 1:
            print(f"[warn] {scene}: 폴더에 .blend 여러개 {[os.path.basename(b) for b in blends]} → 첫번째 사용")
        if blends:
            return blends[0]
    return None


def run_blender_export(blend, out_json, mirror_object=""):
    """Blender headless 로 export_mirror_gt.py 실행 → out_json 생성."""
    if not os.path.isfile(BLENDER):
        raise SystemExit(f"Blender 실행파일 없음: {BLENDER}")
    C.ensure_dir(os.path.dirname(out_json))
    cmd = [BLENDER, "--background", blend, "--python", EXPORT_SCRIPT, "--",
           "--out", out_json, "--camera", "CM", "--mirror_object", mirror_object]
    print("RUN:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 or not os.path.isfile(out_json):
        raise SystemExit(f"Blender export 실패 (rc={rc}, out={out_json})")


def k_from_blender_intrinsics(intr, W, H):
    """Blender 카메라 intrinsics → 3x3 K (이미지 W,H 기준).

    sensor_fit 처리: VERTICAL 또는 (AUTO 이고 세로>가로)면 sensor_height/H 로,
    그 외(HORIZONTAL/AUTO 가로)면 sensor_width/W 로 초점거리(px)를 계산. 정사각 픽셀 가정.
    shift_x/y 는 Blender 규약상 큰 변(max(W,H)) 비율 → 픽셀로 환산해 주점 보정.
    """
    lens = intr["lens_mm"]
    sw, sh = intr["sensor_width_mm"], intr["sensor_height_mm"]
    fit = intr.get("sensor_fit", "AUTO")
    if fit == "VERTICAL" or (fit == "AUTO" and H > W):
        f = lens / sh * H
    else:
        f = lens / sw * W
    mx = max(W, H)
    cx = W / 2.0 + intr.get("shift_x", 0.0) * mx
    cy = H / 2.0 - intr.get("shift_y", 0.0) * mx
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_gt(scene, world_json, run_dir):
    """world json + prepare_meta → left-cam OpenCV gt dict (result.json 스키마 정렬)."""
    import mirror_geometry as MG

    meta = C.load_meta(run_dir)
    # 변환 카메라 = 데이터셋 실제 카메라(prepare_meta). baseline·views(다른 시점) 모두 올바름.
    cam_to_world = np.asarray(meta["left_camera_to_world"], dtype=np.float64)
    blend_cam = world_json.get("left_camera_to_world")
    # blend 의 기본 CM 과 데이터셋 카메라 차이(baseline≈0, view scene 은 큼=정상).
    cam_diff = (float(np.max(np.abs(np.asarray(blend_cam, dtype=np.float64) - cam_to_world)))
                if blend_cam is not None else None)

    # Blender 실제 K (있으면). gt 기하는 K 무관(절대 미터)하지만 overlay/다운스트림 투영에 필요.
    W, H = int(meta["width"]), int(meta["height"])
    intr = world_json.get("camera_intrinsics")
    true_K = k_from_blender_intrinsics(intr, W, H).tolist() if intr else None

    mirrors_out = []
    for m in world_json.get("mirrors", []):
        t = world_plane_to_cam(m["normal"], m["centroid"], m["corners"], cam_to_world)
        n, d = MG.canonicalize_normal(t["normal"], t["corners"], t["d"])  # 카메라쪽으로 부호 정렬
        bbox = MG.plane_bbox(t["corners"], n)
        depth = MG.mirror_depth(n, d, t["corners"])
        mirrors_out.append({
            "name": m["name"],
            "normal": [float(x) for x in n],
            "d": float(d),
            "perp_distance": depth["perp_distance"],
            "centroid": [float(x) for x in t["centroid"]],
            "corners_3d": t["corners"].tolist(),
            "plane_bbox": bbox,
            "planarity_rms": m.get("planarity_rms"),
            "n_verts": m.get("n_verts"),
        })
    return {
        "scene": scene,
        "frame": "left_camera_opencv",
        "image_size": [W, H],
        "true_K": true_K,
        "true_K_source": "blender_camera_lens" if true_K is not None else None,
        "camera_source": "prepare_meta_left_camera_to_world",
        "blend_vs_dataset_cam_diff": cam_diff,
        "mirrors": mirrors_out,
        "world": world_json,
    }


def save_overlay(path, run_dir, gt):
    """GT 평면 4모서리+중심을 K로 재투영해 left.png / mirror_mask.png 위에 오버레이.

    렌더 영상과 맞추려면 Blender 실제 K(gt['true_K'])를 써야 한다(데이터셋 K.txt 는
    lens 가정값이라 폴리곤이 부풀어 정합 판단이 불가). 축은 이미지 범위로 고정해
    화면 밖으로 확장되는 거울(프레임 밖)도 in-frame 부분을 또렷이 비교한다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    if gt.get("true_K") is not None:
        K = np.asarray(gt["true_K"], dtype=np.float64)
    else:
        K, _ = C.read_K_txt(os.path.join(run_dir, "input", "K.txt"))
    left = imageio.imread(os.path.join(run_dir, "input", "left.png"))[..., :3]
    mask = imageio.imread(os.path.join(run_dir, "input", "mirror_mask.png"))
    H, W = left.shape[:2]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    panels = [(ax[0], left, None, "left + GT"),
              (ax[1], mask, "gray", "mask + GT")]
    for a_, img, cmap, title in panels:
        a_.imshow(img, cmap=cmap)
        a_.set_title(title)
        a_.set_xlim(0, W)        # 축을 이미지 범위로 고정 (autoscale 방지)
        a_.set_ylim(H, 0)
        a_.axis("off")
        for m in gt["mirrors"]:
            uv = project_points(K, m["corners_3d"])
            poly = np.vstack([uv, uv[0]])
            a_.plot(poly[:, 0], poly[:, 1], "-", lw=2, color="lime")
            cuv = project_points(K, [m["centroid"]])[0]
            a_.plot(cuv[0], cuv[1], "x", color="red", ms=10, mew=3)
    fig.suptitle(f"GT mirror plane — {gt['scene']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def process_scene(scene, args):
    """단일 scene: (옵션)Blender 추출 → 변환 → gt.json + overlay.png."""
    run_dir = C.run_dir_for(scene)
    if not os.path.isdir(run_dir):
        print(f"[skip] {scene}: experiments 폴더 없음")
        return False
    meta = C.load_meta(run_dir)
    if meta.get("left_camera_to_world") is None:
        print(f"[skip] {scene}: prepare_meta 에 left_camera_to_world 없음")
        return False
    out_dir = C.ensure_dir(os.path.join(run_dir, "mirror_gt"))
    world_json_path = os.path.join(out_dir, "gt_mirror_world.json")

    if not args.no_blender:
        blend = resolve_blend(scene)
        if not blend:
            print(f"[skip] {scene}: .blend 못찾음 (base={blend_base(scene)})")
            return False
        mobj = mirror_object_for(scene, meta)
        print(f"[{scene}] mirror_object = {mobj or '(auto/find_mirrors)'}")
        run_blender_export(blend, world_json_path, mobj)

    if not os.path.isfile(world_json_path):
        print(f"[skip] {scene}: world json 없음 ({world_json_path})")
        return False
    with open(world_json_path, encoding="utf-8") as f:
        world_json = json.load(f)
    if not world_json.get("mirrors"):
        print(f"[warn] {scene}: 거울 0개 (gt.json 은 빈 mirrors 로 저장)")

    gt = build_gt(scene, world_json, run_dir)
    with open(os.path.join(out_dir, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    save_overlay(os.path.join(out_dir, "overlay.png"), run_dir, gt)

    for m in gt["mirrors"]:
        rms = m.get("planarity_rms") or 0.0
        if rms > 0.02:
            print(f"[warn] {scene}/{m['name']}: planarity_rms={rms:.3f}m (비평면 거울 의심)")
    cd = gt.get("blend_vs_dataset_cam_diff")
    cd_s = "n/a" if cd is None else f"{cd:.2e}"
    print(f"[ok] {scene}: mirrors={len(gt['mirrors'])} cam_diff(blend↔dataset)={cd_s} → {out_dir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-blender", action="store_true",
                    help="기존 gt_mirror_world.json 만 재변환(Blender 미실행)")
    args = ap.parse_args()

    if args.all:
        scenes = sorted(d for d in os.listdir(C.EXPERIMENTS_DIR)
                        if os.path.isdir(os.path.join(C.EXPERIMENTS_DIR, d))
                        and not d.startswith("_"))
    elif args.scene:
        scenes = [args.scene]
    else:
        raise SystemExit("--scene <name> 또는 --all 필요")

    results = [(s, process_scene(s, args)) for s in scenes]
    print("\n#### SUMMARY")
    for s, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {s}")


if __name__ == "__main__":
    main()
