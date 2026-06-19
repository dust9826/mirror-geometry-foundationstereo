"""run_archive_batch.py — Archive(ds_v1) 뷰 전체를 준비 + 전체 파이프라인 실행 (재개 가능).

각 뷰: prepare_views_archive.prepare_view → run_pipeline_full
       (FS → estimate_mirror_plane → viz_ransac_inliers → analyze_features
        → [submodel_head] → [apply_submodel → submodel_applied])
출력: experiments/archive_views/<scene>_<view>/  (minigym_view04 와 동일 구성)

재개: fs/depth_meter.npy 가 이미 있으면 건너뜀(--force 로 강제 재실행).

사용:
  python scripts/run_archive_batch.py                         # 전체 (4 scene × 50 = 200)
  python scripts/run_archive_batch.py --scenes blue_bathroom --max_views 3
  python scripts/run_archive_batch.py --no_submodel           # 단일 submodel 학습 생략(빠름, 권장)
  python scripts/run_archive_batch.py --force
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import prepare_views_archive as PV

PY = sys.executable
S = C.SCRIPT_DIR
GROUP = "archive_views"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", help="쉼표목록(미지정=전체)")
    ap.add_argument("--max_views", type=int, default=None, help="scene 당 뷰 수 제한(테스트용)")
    ap.add_argument("--no_submodel", action="store_true", help="단일 submodel 학습 생략")
    ap.add_argument("--no_apply", action="store_true", help="submodel_applied 생략")
    ap.add_argument("--force", action="store_true", help="이미 처리된 뷰도 재실행")
    args = ap.parse_args()

    sv = PV.list_scenes_views()
    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else list(sv)
    tasks = []
    for sc in scenes:
        views = sv.get(sc, [])
        if args.max_views:
            views = views[:args.max_views]
        tasks += [(sc, v) for v in views]

    print(f"처리 대상: {len(tasks)} 뷰  (scenes={scenes})")
    done = skipped = fail = 0
    for i, (sc, v) in enumerate(tasks, 1):
        name = f"{GROUP}/{sc}_{v}"
        marker = os.path.join(C.EXPERIMENTS_DIR, name, "fs", "depth_meter.npy")
        if os.path.isfile(marker) and not args.force:
            print(f"[skip {i}/{len(tasks)}] {name}")
            skipped += 1
            continue
        print(f"\n########## [{i}/{len(tasks)}] {name} ##########")
        try:
            PV.prepare_view(sc, v, out_name=name)
        except SystemExit as e:
            print(f"  !! prepare 실패: {e}")
            fail += 1
            continue
        cmd = [PY, os.path.join(S, "run_pipeline_full.py"), "--scene", name]
        if args.no_submodel:
            cmd.append("--no_submodel")
        if args.no_apply:
            cmd.append("--no_apply")
        rc = subprocess.call(cmd)
        if rc == 0:
            done += 1
        else:
            fail += 1
            print(f"  !! run_pipeline_full 실패 rc={rc}")

    print(f"\n#### BATCH SUMMARY: done {done}, skipped {skipped}, fail {fail}, total {len(tasks)}")


if __name__ == "__main__":
    main()
