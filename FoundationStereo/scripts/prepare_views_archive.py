"""prepare_views_archive.py — Archive(ds_v1, render_views.py 출력)를
FoundationStereo + mirror-geometry 입력 형식으로 변환한다.

ds_v1 뷰 폴더(예: Archive/blue_bathroom/v00/):
  L.png R.png            : stereo pair (left/right)
  L_mirror.npy (bool)    : 거울 마스크 GT
  L_depth.npy (float32)  : 좌영상 GT depth(m)
  disp_gt.npy (float32)  : GT disparity(px)
  meta.json              : fx,fy,cx,cy,baseline_m,width,height,mirror_objects,camera_pose_override ...

기존 stereo_dataset 과 차이:
  - meta.json 에 **실제 intrinsic** 이 있음 → K_is_assumed=False (가정 K 불필요!)
  - **GT depth/disparity** 동봉 → FS 정확도 직접 평가 가능 (input/ 에 depth_gt.npy/disp_gt.npy 로 보존)
  - 거울 마스크가 .npy(bool) → mirror_mask.png 로 변환

사용:
  python scripts/prepare_views_archive.py --list
  python scripts/prepare_views_archive.py --scene blue_bathroom --view v00
  python scripts/prepare_views_archive.py --scene blue_bathroom --all_views
  python scripts/prepare_views_archive.py --all
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import imageio.v2 as imageio

import common as C

ARCHIVE = os.path.normpath(os.path.join(C.PROJECT_ROOT, "..", "Archive"))


def list_scenes_views(archive=None):
    archive = archive or ARCHIVE
    out = {}
    if not os.path.isdir(archive):
        return out
    for sc in sorted(os.listdir(archive)):
        sd = os.path.join(archive, sc)
        if not os.path.isdir(sd):
            continue
        views = sorted(v for v in os.listdir(sd)
                       if os.path.isfile(os.path.join(sd, v, "meta.json")))
        if views:
            out[sc] = views
    return out


def prepare_view(scene, view, out_name=None, out_root=C.EXPERIMENTS_DIR, archive=None):
    vdir = os.path.join(archive or ARCHIVE, scene, view)
    meta_p = os.path.join(vdir, "meta.json")
    if not os.path.isfile(meta_p):
        raise SystemExit(f"meta.json 없음: {meta_p}")
    meta = json.load(open(meta_p, encoding="utf-8"))

    name = out_name or f"{scene}_{view}"
    in_dir = C.ensure_dir(os.path.join(out_root, name, "input"))

    shutil.copyfile(os.path.join(vdir, "L.png"), os.path.join(in_dir, "left.png"))
    shutil.copyfile(os.path.join(vdir, "R.png"), os.path.join(in_dir, "right.png"))

    # 거울 마스크 (bool npy) → png (흰=거울)
    m = np.load(os.path.join(vdir, "L_mirror.npy"))
    imageio.imwrite(os.path.join(in_dir, "mirror_mask.png"),
                    (m.astype(np.uint8) * 255))

    # 실제 intrinsic 으로 K.txt
    K = np.array([[meta["fx"], 0.0, meta["cx"]],
                  [0.0, meta["fy"], meta["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    C.write_K_txt(os.path.join(in_dir, "K.txt"), K, meta["baseline_m"])

    # GT depth/disparity 보존 (FS 평가용)
    for src, dst in [("L_depth.npy", "depth_gt.npy"), ("disp_gt.npy", "disp_gt.npy")]:
        sp = os.path.join(vdir, src)
        if os.path.isfile(sp):
            shutil.copyfile(sp, os.path.join(in_dir, dst))

    prep = dict(
        scene=name, source_dir=vdir,
        width=meta["width"], height=meta["height"],
        baseline_m=meta["baseline_m"],
        K=K.tolist(), K_source="meta_real", K_is_assumed=False,
        fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"],
        mirror_objects=meta.get("mirror_objects"),
        camera_pose_override=meta.get("camera_pose_override"),
        mirror_visible_fraction_proj=meta.get("mirror_visible_fraction_proj"),
        mirror_unoccluded_fraction=meta.get("mirror_unoccluded_fraction"),
        pair_type=meta.get("pair_type"),
        cam_matrix_world=meta.get("cam_matrix_world"),
        # ds_v1 seed판: 렌더러가 GT 거울평면을 직접 출력(world+cam 좌표) → 평가용 보존
        mirror_plane_world=meta.get("mirror_plane_world"),
        mirror_plane_cam=meta.get("mirror_plane_cam"),
        scene_blend=meta.get("scene_blend"),
        has_gt_depth=os.path.isfile(os.path.join(in_dir, "depth_gt.npy")),
        has_gt_disp=os.path.isfile(os.path.join(in_dir, "disp_gt.npy")),
        note="ds_v1(render_views) 출력. 실제 intrinsic 사용. GT depth/disp 동봉.",
    )
    json.dump(prep, open(os.path.join(in_dir, "prepare_meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--view")
    ap.add_argument("--name", help="출력 experiments/<name> (기본: <scene>_<view>)")
    ap.add_argument("--all_views", action="store_true", help="해당 scene 의 모든 뷰")
    ap.add_argument("--all", action="store_true", help="모든 scene 의 모든 뷰")
    ap.add_argument("--list", action="store_true", help="Archive scene/뷰 목록만 출력")
    ap.add_argument("--archive_root", default=ARCHIVE,
                    help="ds_v1 루트(기본 ../Archive). 예: ../Archive/ds_v1_seed0")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    args = ap.parse_args()

    sv = list_scenes_views(args.archive_root)
    if args.scene and not args.all and not args.all_views and args.view:
        print("prepared:", prepare_view(args.scene, args.view, args.name, args.out_root,
                                        archive=args.archive_root))
        return

    targets = []
    if args.all:
        for sc, views in sv.items():
            targets += [(sc, v) for v in views]
    elif args.all_views and args.scene:
        targets = [(args.scene, v) for v in sv.get(args.scene, [])]
    else:
        print(f"Archive: {args.archive_root}")
        for sc, views in sv.items():
            print(f"  {sc}: {len(views)} views ({views[0]}..{views[-1]})")
        print("\n--scene X --view v00  |  --scene X --all_views  |  --all")
        return

    for sc, v in targets:
        print("prepared:", prepare_view(sc, v, None, args.out_root,
                                        archive=args.archive_root))
    print(f"\n총 {len(targets)} 뷰 준비 완료 → {args.out_root}")


if __name__ == "__main__":
    main()
