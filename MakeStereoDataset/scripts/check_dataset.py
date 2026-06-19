# -*- coding: utf-8 -*-
"""
check_dataset.py

data/stereo_dataset 의 각 결과 폴더를 점검한다.
  - 파일 4종(left/right/mask/meta) 존재 여부
  - left/right 밝기/마젠타 비율 (렌더 깨짐 판정)
  - mirror_mask 의 흰색 비율 = 거울 포함률(coverage %)
"""
import os
import sys
import json
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "stereo_dataset")


def stats(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    mean = float(a.mean())
    magenta = float(((a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] > 200)).mean() * 100)
    black = float((a.max(axis=2) < 10).mean() * 100)
    return mean, magenta, black


def mask_coverage(path):
    m = np.asarray(Image.open(path).convert("L"))
    return float((m > 128).mean() * 100)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = sorted(d for d in os.listdir(DS) if os.path.isdir(os.path.join(DS, d)))
    if target:
        dirs = [d for d in dirs if target in d]

    print("%-32s %6s %6s %6s %7s %s" % ("scene", "Lmean", "mag:%", "blk:%", "mirror%", "status"))
    print("-" * 78)
    for d in dirs:
        p = os.path.join(DS, d)
        lf = os.path.join(p, "left.png")
        rf = os.path.join(p, "right.png")
        mf = os.path.join(p, "mirror_mask_left.png")
        jf = os.path.join(p, "stereo_meta.json")
        if not all(os.path.exists(x) for x in (lf, rf, mf, jf)):
            print("%-32s  MISSING FILES" % d)
            continue
        lmean, lmag, lblk = stats(lf)
        cov = mask_coverage(mf)
        # 깨짐 판정: 마젠타 많거나 거의 검정이면 broken
        status = "ok"
        if lmag > 3:
            status = "MAGENTA(%.1f%%)" % lmag
        elif lblk > 60:
            status = "TOO_DARK"
        elif cov < 0.3:
            status = "no_mirror"
        print("%-32s %6.1f %6.2f %6.2f %7.2f %s" % (d, lmean, lmag, lblk, cov, status))


if __name__ == "__main__":
    main()
