"""
prepare_data.py — MakeStereoDataset 의 완성된 stereo_dataset scene 하나를
FoundationStereo + mirror-geometry 실험 입력 형식으로 변환한다.

하는 일:
  1. ../MakeStereoDataset/data/stereo_dataset/<scene>/ 에서 left/right/mirror_mask 복사
  2. stereo_meta.json 의 baseline + (가정) intrinsic 으로 K.txt 생성
  3. experiments/<scene>/input/ 에 정리 + prepare_meta.json 기록

사용:
  python scripts/prepare_data.py --list
  python scripts/prepare_data.py --scene cozy_living_room_baseline080
  python scripts/prepare_data.py --scene minigym_baseline080 --lens 50 --sensor 36
"""

import argparse
import glob
import json
import os
import shutil

import imageio.v2 as imageio

import common as C


def list_scenes():
    if not os.path.isdir(C.STEREO_DATASET_DIR):
        return []
    out = []
    for d in sorted(os.listdir(C.STEREO_DATASET_DIR)):
        full = os.path.join(C.STEREO_DATASET_DIR, d)
        if os.path.isfile(os.path.join(full, "left.png")):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="stereo_dataset 폴더명 (예: cozy_living_room_baseline080)")
    ap.add_argument("--src_dir", help="left/right/mirror_mask_left/stereo_meta 가 든 임의 폴더 "
                    "(stereo_dataset_views 등). 지정 시 --scene 은 출력 이름으로만 쓰임.")
    ap.add_argument("--name", help="--src_dir 사용 시 출력 experiments/<name> 지정(기본: 경로에서 유추)")
    ap.add_argument("--list", action="store_true", help="사용 가능한 scene 목록")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    # intrinsic 가정 (meta 에 K 가 없으므로)
    ap.add_argument("--lens", type=float, default=50.0, help="Blender 렌즈 초점거리(mm)")
    ap.add_argument("--sensor", type=float, default=36.0, help="센서 가로(mm)")
    ap.add_argument("--fx", type=float, default=None, help="픽셀 초점거리 직접 지정(우선)")
    ap.add_argument("--hfov", type=float, default=None, help="수평 화각(deg) 로 지정")
    args = ap.parse_args()

    if args.src_dir:
        # 임의 폴더(예: stereo_dataset_views/minigym/view_04) 직접 지정.
        src = os.path.abspath(args.src_dir)
        if not os.path.isdir(src):
            raise SystemExit(f"src_dir 없음: {src}")
        if args.name:
            scene_name = args.name
        elif args.scene:
            scene_name = args.scene
        else:
            # 경로 마지막 두 단계를 합쳐 이름 생성 (minigym_view_04 식)
            parts = [p for p in src.replace("\\", "/").split("/") if p]
            scene_name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        args.scene = scene_name
    else:
        scenes = list_scenes()
        if args.list or not args.scene:
            if not scenes:
                print("stereo_dataset 가 비었습니다:", C.STEREO_DATASET_DIR)
            else:
                print(f"{len(scenes)} scene(s) in {C.STEREO_DATASET_DIR}:")
                for s in scenes:
                    print("  ", s)
            if not args.scene:
                return
        src = os.path.join(C.STEREO_DATASET_DIR, args.scene)
        if not os.path.isdir(src):
            raise SystemExit(f"scene 없음: {src}  (--list 로 확인)")

    left_src = os.path.join(src, "left.png")
    right_src = os.path.join(src, "right.png")
    mask_src = os.path.join(src, "mirror_mask_left.png")
    meta_src = os.path.join(src, "stereo_meta.json")
    for p in (left_src, right_src, meta_src):
        if not os.path.exists(p):
            raise SystemExit(f"필수 파일 없음: {p}")

    with open(meta_src) as f:
        meta = json.load(f)
    baseline = float(meta.get("baseline_m", 0.08))

    h, w = imageio.imread(left_src).shape[:2]
    K = C.intrinsics_from_assumption(w, h, lens_mm=args.lens, sensor_mm=args.sensor,
                                     fx=args.fx, hfov_deg=args.hfov)

    run_dir = os.path.join(args.out_root, args.scene)
    in_dir = C.ensure_dir(os.path.join(run_dir, "input"))
    shutil.copyfile(left_src, os.path.join(in_dir, "left.png"))
    shutil.copyfile(right_src, os.path.join(in_dir, "right.png"))
    if os.path.exists(mask_src):
        shutil.copyfile(mask_src, os.path.join(in_dir, "mirror_mask.png"))
    C.write_K_txt(os.path.join(in_dir, "K.txt"), K, baseline)

    prep = {
        "scene": args.scene,
        "source_dir": src,
        "width": w, "height": h,
        "baseline_m": baseline,
        "K": K.tolist(),
        "K_source": ("fx" if args.fx else "hfov" if args.hfov else "lens/sensor"),
        "lens_mm": args.lens, "sensor_mm": args.sensor,
        "K_is_assumed": True,
        "mirror_objects": meta.get("mirror_objects", meta.get("mirror_object")),
        "left_camera_to_world": meta.get("left_camera_to_world"),
        "right_camera_to_world": meta.get("right_camera_to_world"),
        "note": "K(초점거리)는 가정값. disparity 무관, depth/평면거리 스케일만 영향.",
    }
    with open(os.path.join(in_dir, "prepare_meta.json"), "w") as f:
        json.dump(prep, f, indent=2)

    print("prepared:", run_dir)
    print("  K=\n", K)
    print("  baseline(m):", baseline, " size:", (w, h))
    print("  mirror mask:", os.path.exists(mask_src))


if __name__ == "__main__":
    main()
