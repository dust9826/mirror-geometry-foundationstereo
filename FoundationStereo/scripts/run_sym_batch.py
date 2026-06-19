"""run_sym_batch.py — 반사대칭 거울평면 추정을 전체 뷰에 일괄 실행.

  python scripts/run_sym_batch.py --shard 0/8        # 8분할 중 0번 (병렬용)
  python scripts/run_sym_batch.py --aggregate        # 끝나고 집계만

뷰당 ~5-15s (FS 재추론 없음, fs/points 재사용). resume: result.json 있으면 skip.
집계: experiments/_eval_sym.csv / _eval_sym.json (씬별 중앙값·성공률).
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import estimate_mirror_plane_sym as SY


def list_views(groups):
    out = []
    for g in groups:
        gd = os.path.join(C.EXPERIMENTS_DIR, g)
        if not os.path.isdir(gd):
            continue
        for name in sorted(os.listdir(gd)):
            if os.path.isfile(os.path.join(gd, name, "fs", "points.npy")):
                out.append(f"{g}/{name}")
    return out


def _sub(mask):
    return "mirror_geometry_sym" if mask == "gt" else f"mirror_geometry_sym_{mask}"


def aggregate(groups, mask="gt", scenes=None):
    rows = []
    for v in list_views(groups):
        if scenes and v.split("/")[-1].rsplit("_v", 1)[0] not in scenes:
            continue
        p = os.path.join(C.EXPERIMENTS_DIR, v, _sub(mask), "result.json")
        if not os.path.isfile(p):
            continue
        r = json.load(open(p, encoding="utf-8"))
        ev = r.get("eval_vs_gt") or {}
        sc = v.split("/")[-1].rsplit("_v", 1)[0]
        rows.append(dict(view=v, scene=sc, perp=r.get("perp"), score=r.get("score"),
                         angle_deg=ev.get("angle_deg"), perp_err_m=ev.get("perp_err_m"),
                         gt_perp=ev.get("gt_perp")))
    if not rows:
        print("결과 없음")
        return
    tag = "" if mask == "gt" else f"_{mask}"
    with open(os.path.join(C.EXPERIMENTS_DIR, f"_eval_sym{tag}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def stats(rs):
        a = np.array([r["angle_deg"] for r in rs if r["angle_deg"] is not None])
        d = np.array([r["perp_err_m"] for r in rs if r["perp_err_m"] is not None])
        if not len(a):
            return None
        return dict(n=len(a),
                    angle_med=float(np.median(a)), angle_mean=float(a.mean()),
                    perp_med=float(np.median(d)), perp_mean=float(d.mean()),
                    ok_5deg=float((a < 5).mean()), ok_10deg=float((a < 10).mean()),
                    ok_02m=float((d < 0.2).mean()), ok_05m=float((d < 0.5).mean()),
                    ok_both=float(((a < 10) & (d < 0.5)).mean()))

    per_scene = {}
    for sc in sorted({r["scene"] for r in rows}):
        s = stats([r for r in rows if r["scene"] == sc])
        if s:
            per_scene[sc] = s
    overall = stats(rows)
    json.dump(dict(overall=overall, per_scene=per_scene, mask=mask),
              open(os.path.join(C.EXPERIMENTS_DIR, f"_eval_sym{tag}.json"), "w",
                   encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n===== 반사대칭 평면 vs GT ({overall['n']} views, mask={mask}) =====")
    print(f"법선각  중앙값 {overall['angle_med']:.1f}°  (<5° {overall['ok_5deg']*100:.0f}%"
          f", <10° {overall['ok_10deg']*100:.0f}%)")
    print(f"perp오차 중앙값 {overall['perp_med']:.2f}m  (<0.2m {overall['ok_02m']*100:.0f}%"
          f", <0.5m {overall['ok_05m']*100:.0f}%)")
    print(f"동시(<10° & <0.5m): {overall['ok_both']*100:.0f}%")
    print(f"\n{'scene':22s} {'n':>4s} {'각도med':>8s} {'perp med':>9s} {'<10°&<0.5m':>11s}")
    for sc, s in per_scene.items():
        print(f"{sc:22s} {s['n']:4d} {s['angle_med']:7.1f}° {s['perp_med']:8.2f}m "
              f"{s['ok_both']*100:10.0f}%")
    print(f"\n저장: experiments/_eval_sym.csv / _eval_sym.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="ds_v1_seed0,ds_v2_seed1")
    ap.add_argument("--shard", default="0/1", help="i/n")
    ap.add_argument("--mask", default="gt", choices=["gt", "submodel"])
    ap.add_argument("--n_hyp", type=int, default=4000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="집계만 수행")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    if args.aggregate:
        aggregate(groups, mask=args.mask)
        return

    si, sn = map(int, args.shard.split("/"))
    views = [v for i, v in enumerate(list_views(groups)) if i % sn == si]
    print(f"[shard {si}/{sn}] {len(views)} views", flush=True)
    t0 = time.time()
    done = skip = fail = 0
    for i, v in enumerate(views, 1):
        out = os.path.join(C.EXPERIMENTS_DIR, v, "mirror_geometry_sym", "result.json")
        if not args.force and os.path.isfile(out):
            skip += 1
            continue
        try:
            r = SY.run_scene(v, mask_src=args.mask, n_hyp=args.n_hyp, verbose=False)
            done += 1 if r else 0
            fail += 0 if r else 1
        except Exception as e:
            print(f"!! {v}: {e}", flush=True)
            fail += 1
        if i % 25 == 0:
            el = time.time() - t0
            eta = el / i * (len(views) - i)
            print(f"[shard {si}] {i}/{len(views)} done {done} fail {fail} skip {skip} "
                  f"({el/i:.1f}s/view, ETA {eta/60:.0f}m)", flush=True)
    print(f"[shard {si}] 끝: done {done}, fail {fail}, skip {skip}, "
          f"총 {(time.time()-t0)/60:.0f}m", flush=True)


if __name__ == "__main__":
    main()
