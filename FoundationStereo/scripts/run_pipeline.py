"""
run_pipeline.py — 한 scene 에 대해 전체 파이프라인을 순서대로 실행하는 드라이버.

  prepare_data -> run_foundation_stereo -> estimate_mirror_plane -> analyze_features
  (옵션) submodel_head

MakeStereoDataset 의 make_stereo_dataset.py 에 대응하는 "한 방" 실행 스크립트.

사용:
  python scripts/run_pipeline.py --scene cozy_living_room_baseline080
  python scripts/run_pipeline.py --scene minigym_baseline080 --submodel
  python scripts/run_pipeline.py --all            # stereo_dataset 전체
"""

import argparse
import os
import subprocess
import sys

import common as C

PY = sys.executable
S = C.SCRIPT_DIR


def step(name, cmd):
    print("\n" + "=" * 70)
    print("STEP:", name)
    print("=" * 70)
    rc = subprocess.call([PY] + cmd)
    if rc != 0:
        print(f"!! step '{name}' 실패 (rc={rc})")
    return rc == 0


def run_scene(scene, args):
    ok = True
    ok &= step("prepare_data", [os.path.join(S, "prepare_data.py"), "--scene", scene,
                                "--lens", str(args.lens), "--sensor", str(args.sensor)])
    fs = [os.path.join(S, "run_foundation_stereo.py"), "--scene", scene,
          "--valid_iters", str(args.valid_iters)]
    if args.ckpt:
        fs += ["--ckpt", args.ckpt]
    ok &= step("run_foundation_stereo", fs)
    ok &= step("estimate_mirror_plane", [os.path.join(S, "estimate_mirror_plane.py"), "--scene", scene])
    ok &= step("analyze_features", [os.path.join(S, "analyze_features.py"), "--scene", scene])
    if args.submodel:
        step("submodel_head", [os.path.join(S, "submodel_head.py"), "--scene", scene])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--valid_iters", type=int, default=32)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--sensor", type=float, default=36.0)
    ap.add_argument("--submodel", action="store_true")
    args = ap.parse_args()

    if args.all:
        scenes = [d for d in sorted(os.listdir(C.STEREO_DATASET_DIR))
                  if os.path.isfile(os.path.join(C.STEREO_DATASET_DIR, d, "left.png"))]
    elif args.scene:
        scenes = [args.scene]
    else:
        raise SystemExit("--scene <name> 또는 --all 필요")

    results = []
    for s in scenes:
        results.append((s, run_scene(s, args)))

    print("\n" + "#" * 70 + "\nSUMMARY")
    for s, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {s}")


if __name__ == "__main__":
    main()
