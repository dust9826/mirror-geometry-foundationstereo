"""run_fs_server.py — FS 모델을 한 번만 로드해 모든 뷰를 연속 추론(모델 상주).

ds_v1/v2(Archive seed판) 루트들을 받아:
  뷰마다 prepare(input/ 없으면) → infer_scene(fs/ 저장, 끝나면 fs/.done)

여러 루트·씬을 라운드로빈으로 섞어 진행해 후처리 워커들(--post_only)이
모든 샤드에서 고르게 일감을 받도록 한다.

사용:
  python scripts/run_fs_server.py --archive_roots ../Archive/ds_v1_seed0,../Archive/ds_v2_seed1 \
      --exclude_scenes loft
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C
import prepare_views_archive as PV
import run_foundation_stereo as FS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive_roots", required=True, help="쉼표목록")
    ap.add_argument("--exclude_scenes", default="", help="쉼표목록(예: loft)")
    ap.add_argument("--scenes", default=None, help="포함 필터(쉼표, 기본 전체)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--force", action="store_true", help="fs/.done 있어도 재추론")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개 뷰만(테스트용)")
    args = ap.parse_args()

    roots = [os.path.abspath(r.strip()) for r in args.archive_roots.split(",") if r.strip()]
    excl = {s.strip() for s in args.exclude_scenes.split(",") if s.strip()}
    incl = ({s.strip() for s in args.scenes.split(",") if s.strip()}
            if args.scenes else None)

    # (root, group, scene) 별 뷰 리스트 → 씬 단위 라운드로빈으로 평탄화
    lists = []
    total = 0
    for root in roots:
        group = os.path.basename(os.path.normpath(root))
        sv = PV.list_scenes_views(root)
        for sc, views in sv.items():
            if sc in excl or (incl and sc not in incl):
                continue
            lists.append([(root, group, sc, v) for v in views])
            total += len(views)
    order = []
    i = 0
    while any(lists):
        for L in lists:
            if i < len(L):
                order.append(L[i])
        i += 1
        if all(i >= len(L) for L in lists):
            break
    assert len(order) == total
    if args.limit:
        order = order[:args.limit]
        total = len(order)

    print(f"[fs_server] roots={len(roots)}, 대상 {total} views", flush=True)
    model, _cfg, _ = FS.load_model(args.ckpt)

    t0 = time.time()
    done = skip = fail = 0
    for n, (root, group, sc, v) in enumerate(order, 1):
        name = f"{group}/{sc}_{v}"
        run_dir = os.path.join(C.EXPERIMENTS_DIR, name)
        if not args.force and os.path.isfile(os.path.join(run_dir, "fs", ".done")):
            skip += 1
            continue
        try:
            if not os.path.isfile(os.path.join(run_dir, "input", "prepare_meta.json")):
                PV.prepare_view(sc, v, out_name=name, archive=root)
            FS.infer_scene(model, name)
            done += 1
        except Exception:
            print(f"!! infer 실패: {name}")
            traceback.print_exc()
            fail += 1
        if (done + fail) % 20 == 0 and (done + fail) > 0:
            el = time.time() - t0
            rate = el / max(done + fail, 1)
            eta = rate * (total - n)
            print(f"[fs_server] {n}/{total} (done {done}, skip {skip}, fail {fail}) "
                  f"{rate:.1f}s/view, ETA {eta/60:.0f}m", flush=True)

    print(f"[fs_server] 끝: done {done}, skip {skip}, fail {fail}, "
          f"총 {(time.time()-t0)/60:.0f}m", flush=True)


if __name__ == "__main__":
    main()
