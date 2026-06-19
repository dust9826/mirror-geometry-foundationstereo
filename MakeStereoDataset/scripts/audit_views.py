"""
audit_views.py

이미 렌더된 카메라-이동 데이터셋(data/stereo_dataset_views/<scene>/view_*)을
사후 점검한다. Blender 불필요 — left.png / mirror_mask_left.png 만 읽는다.

쓸 수 없는 view 판정(데이터셋 부적합):
  - black_pct  : '거의 순수 검정' 픽셀 비율. 거울/벽 뒤로 카메라가 가서
                 화면이 크게 가려진 경우(뒷면·검정 과다) -> 못 씀.
  - coverage   : 거울 포함률. 목표 범위[min_cov,max_cov] 밖이면 부적합.
  - dark_pct   : 전체적으로 너무 어두운 프레임 가드(옵션).

usable = (black_pct <= max_black) and (min_cov <= coverage <= max_cov)
         and (dark_pct <= max_dark)

결과: 각 view stereo_meta.json 에 audit 결과 기록 + 콘솔/JSON 요약.
부적합 view 는 기본은 표시만, --move-bad 면 _rejected/ 로 이동.

실행:
  python scripts/audit_views.py
  python scripts/audit_views.py --max_black 8 --min_cov 15 --max_cov 45 --move-bad
"""

import argparse
import json
import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWS = os.path.join(ROOT, "data", "stereo_dataset_views")


def metrics(left_path, mask_path):
    a = np.asarray(Image.open(left_path).convert("RGB"))
    g = a.mean(axis=2)
    black = float((a.max(axis=2) < 12).mean() * 100)
    dark = float((g < 25).mean() * 100)
    mean = float(g.mean())
    cov = None
    if os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert("L"))
        cov = float((m > 128).mean() * 100)
    return {"black_pct": round(black, 2), "dark_pct": round(dark, 2),
            "mean": round(mean, 1), "coverage_pct": round(cov, 2) if cov is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_black", type=float, default=8.0)
    ap.add_argument("--max_dark", type=float, default=55.0)
    ap.add_argument("--min_cov", type=float, default=15.0)
    ap.add_argument("--max_cov", type=float, default=45.0)
    ap.add_argument("--scenes", default="", help="콤마구분(미지정=전체)")
    ap.add_argument("--move-bad", dest="move_bad", action="store_true",
                    help="부적합 view 를 scene/_rejected/ 로 이동")
    args = ap.parse_args()

    if not os.path.isdir(VIEWS):
        print("no views dir:", VIEWS)
        return
    want = {s.strip() for s in args.scenes.split(",") if s.strip()}
    scenes = sorted(d for d in os.listdir(VIEWS)
                    if os.path.isdir(os.path.join(VIEWS, d)))
    if want:
        scenes = [s for s in scenes if s in want]

    summary = []
    tot_usable = tot_total = 0
    print("%-22s %-9s %6s %6s %7s  %s" %
          ("scene/view", "verdict", "black", "dark", "mirror", ""))
    print("-" * 72)
    for sc in scenes:
        sdir = os.path.join(VIEWS, sc)
        views = sorted(v for v in os.listdir(sdir)
                       if v.startswith("view_") and os.path.isdir(os.path.join(sdir, v)))
        s_usable = 0
        for v in views:
            vdir = os.path.join(sdir, v)
            lp = os.path.join(vdir, "left.png")
            mp = os.path.join(vdir, "mirror_mask_left.png")
            if not os.path.exists(lp):
                continue
            mt = metrics(lp, mp)
            cov = mt["coverage_pct"]
            usable = (mt["black_pct"] <= args.max_black and
                      mt["dark_pct"] <= args.max_dark and
                      cov is not None and args.min_cov <= cov <= args.max_cov)
            reason = []
            if mt["black_pct"] > args.max_black:
                reason.append("black>%.0f" % args.max_black)
            if mt["dark_pct"] > args.max_dark:
                reason.append("dark")
            if cov is None or not (args.min_cov <= cov <= args.max_cov):
                reason.append("cov_out")
            tot_total += 1
            s_usable += int(usable)
            tot_usable += int(usable)
            print("%-22s %-9s %5.1f%% %5.1f%% %6.1f%%  %s" %
                  ("%s/%s" % (sc, v), "OK" if usable else "REJECT",
                   mt["black_pct"], mt["dark_pct"],
                   cov if cov is not None else -1, ",".join(reason)))

            # meta 갱신
            mj = os.path.join(vdir, "stereo_meta.json")
            meta = {}
            if os.path.exists(mj):
                try:
                    meta = json.load(open(mj, encoding="utf-8"))
                except Exception:
                    meta = {}
            meta["audit"] = {**mt, "usable": usable, "reasons": reason}
            json.dump(meta, open(mj, "w", encoding="utf-8"), indent=2)

            if args.move_bad and not usable:
                rej = os.path.join(sdir, "_rejected")
                os.makedirs(rej, exist_ok=True)
                os.replace(vdir, os.path.join(rej, v))

        summary.append({"scene": sc, "views": len(views), "usable": s_usable})

    print("-" * 72)
    for s in summary:
        print("  %-22s usable %d / %d" % (s["scene"], s["usable"], s["views"]))
    print("TOTAL usable: %d / %d" % (tot_usable, tot_total))
    out = os.path.join(VIEWS, "audit_summary.json")
    json.dump({"summary": summary, "total_usable": tot_usable,
               "total": tot_total, "params": vars(args)},
              open(out, "w", encoding="utf-8"), indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
