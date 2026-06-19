# -*- coding: utf-8 -*-
"""colab_all.zip / colab_all_vitl.zip 갱신 — 모델은 기존 zip에서 복사,
scripts/ 와 demo 입력은 현재 로컬 최신본으로 전부 교체. (colab_loso.zip 은 변경 없음)

demo 입력 = ds_v1_seed0/computer_room_v00 (정공법 성공률 100% 장면, 실제 intrinsic+GT 동봉).
"""
import glob
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
COLAB = os.path.join(ROOT, "colab")
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py")))
DEMO_SRC = os.path.join(ROOT, "experiments", "ds_v1_seed0", "computer_room_v00", "input")
DEMO_FILES = sorted(glob.glob(os.path.join(DEMO_SRC, "*"))) if os.path.isdir(DEMO_SRC) else []

for name in ("colab_all.zip", "colab_all_vitl.zip"):
    src = os.path.join(COLAB, name)
    if not os.path.isfile(src):
        print("skip(없음):", name)
        continue
    tmp = src + ".new"
    with zipfile.ZipFile(src) as zin, \
         zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        kept = 0
        for info in zin.infolist():
            if info.filename.startswith("scripts/"):
                continue                     # scripts 는 최신본으로 교체
            if DEMO_FILES and info.filename.startswith("experiments/demo/"):
                continue                     # demo 입력도 교체
            zout.writestr(info, zin.read(info.filename))
            kept += 1
        for p in SCRIPTS:
            zout.write(p, "scripts/" + os.path.basename(p))
        for p in DEMO_FILES:
            zout.write(p, "experiments/demo/input/" + os.path.basename(p))
    shutil.move(tmp, src)
    print(f"WROTE {name}: 보존 {kept} + scripts {len(SCRIPTS)} + demo {len(DEMO_FILES)} "
          f"({os.path.getsize(src)/1e6:.0f} MB)")
