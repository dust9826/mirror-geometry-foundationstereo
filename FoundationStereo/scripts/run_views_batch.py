"""run_views_batch.py — ds_v1(Archive seed판) 전체 씬×뷰에 prepare + 전체 파이프라인 일괄 실행.

각 뷰마다:
  prepare_views_archive(준비) → run_pipeline_full(FS→평면→inlier viz→GT→RQ→submodel→apply)

resume: 마지막 산출물(submodel_applied/metrics.json)이 이미 있으면 건너뜀(--force 로 재실행).
끝나면 eval_gt_depth.py --root 로 GT depth/disp 일괄 평가까지 수행.

사용:
  python scripts/run_views_batch.py --archive_root ../Archive/ds_v1_seed0
  python scripts/run_views_batch.py --archive_root ../Archive/ds_v1_seed0 --scenes terrazzo,minigym
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C
import prepare_views_archive as PV

PY = sys.executable
S = C.SCRIPT_DIR


def run(script, *argv):
    return subprocess.call([PY, os.path.join(S, script)] + list(argv)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive_root", required=True, help="예: ../Archive/ds_v1_seed0")
    ap.add_argument("--group", default=None,
                    help="experiments/<group>/<scene>_<view> (기본: archive_root 폴더명)")
    ap.add_argument("--scenes", default=None, help="쉼표목록 필터(기본 전체)")
    ap.add_argument("--force", action="store_true", help="완료된 뷰도 재실행")
    ap.add_argument("--no_eval", action="store_true", help="마지막 eval_gt_depth 생략")
    ap.add_argument("--gpu_lock", default=None,
                    help="run_pipeline_full 에 전달할 FS 직렬화 락파일 경로")
    ap.add_argument("--post_only", action="store_true",
                    help="run_fs_server 가 만든 fs/.done 뷰만 후처리(--no_fs). "
                         "준비 안 된 뷰는 폴링 대기")
    args = ap.parse_args()

    root = os.path.abspath(args.archive_root)
    group = args.group or os.path.basename(os.path.normpath(root))
    sv = PV.list_scenes_views(root)
    if args.scenes:
        keep = {s.strip() for s in args.scenes.split(",") if s.strip()}
        sv = {k: v for k, v in sv.items() if k in keep}
    targets = [(sc, v) for sc, views in sv.items() for v in views]
    if not targets:
        raise SystemExit(f"대상 뷰 없음: {root}")

    print(f"[batch] {root} → experiments/{group}/  ({len(sv)} scene, {len(targets)} views"
          f"{', post_only' if args.post_only else ''})", flush=True)
    t0 = time.time()
    done = fail = skip = 0

    def process(i, sc, v):
        nonlocal done, fail
        name = f"{group}/{sc}_{v}"
        el = time.time() - t0
        n_run = done + fail
        eta = (el / n_run * (len(targets) - i + 1)) if n_run else 0
        print(f"\n##### [{i}/{len(targets)}] {name}  "
              f"(done {done}, fail {fail}, skip {skip}, "
              f"elapsed {el/60:.0f}m, ETA {eta/60:.0f}m) #####", flush=True)
        extra = ["--gpu_lock", args.gpu_lock] if args.gpu_lock else []
        if args.post_only:
            extra.append("--no_fs")
        else:
            try:
                PV.prepare_view(sc, v, out_name=name, archive=root)
            except SystemExit as e:
                print(f"!! prepare 실패: {e}")
                fail += 1
                return
        if run("run_pipeline_full.py", "--scene", name, *extra):
            done += 1
        else:
            fail += 1

    if args.post_only:
        # fs/.done 된 뷰부터 후처리, 안 된 뷰는 폴링 대기 (fs server 와 병렬 가동)
        pending = list(enumerate(targets, 1))
        idle = 0
        while pending:
            progressed = False
            for item in list(pending):
                i, (sc, v) = item
                name = f"{group}/{sc}_{v}"
                run_dir = os.path.join(C.EXPERIMENTS_DIR, name)
                if not args.force and os.path.isfile(
                        os.path.join(run_dir, "submodel_applied", "metrics.json")):
                    skip += 1
                    pending.remove(item)
                    continue
                if os.path.isfile(os.path.join(run_dir, "fs", ".done")):
                    process(i, sc, v)
                    pending.remove(item)
                    progressed = True
            if pending and not progressed:
                idle += 1
                if idle >= 80:   # ~20분 무진행 → fs server 죽음/해당 뷰 추론실패로 간주
                    print(f"!! post 대기 타임아웃, 미처리 {len(pending)}개 포기: "
                          f"{[f'{s}_{v}' for _, (s, v) in pending[:10]]}")
                    fail += len(pending)
                    break
                time.sleep(15)   # fs server 를 기다림
            else:
                idle = 0
    else:
        for i, (sc, v) in enumerate(targets, 1):
            name = f"{group}/{sc}_{v}"
            run_dir = os.path.join(C.EXPERIMENTS_DIR, name)
            marker = os.path.join(run_dir, "submodel_applied", "metrics.json")
            if not args.force and os.path.isfile(marker):
                skip += 1
                continue
            process(i, sc, v)

    print(f"\n[batch] 끝: done {done}, fail {fail}, skip {skip}, "
          f"총 {(time.time()-t0)/3600:.1f}h")
    if not args.no_eval:
        run("eval_gt_depth.py", "--root", os.path.join(C.EXPERIMENTS_DIR, group))


if __name__ == "__main__":
    main()
