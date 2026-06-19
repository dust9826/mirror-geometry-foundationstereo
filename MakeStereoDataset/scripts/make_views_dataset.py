"""
make_views_dataset.py  (전체 자동화 호스트)

모든(또는 선택) scene 에 대해 카메라 이동 데이터셋을 자동 생성한다.
각 scene 마다:
  1. reflect3r 정답 inside_mask(rendered_data/<scene>/masks/inside_mask.png) 자동 매핑
  2. Blender 로 render_camera_views.py 실행
     - reference mask 로 거울 오브젝트 calibration
     - 원본 CM 주변 카메라 포즈 perturb -> 거울 포함률 스카우트
     - 포함률 [min_cov,max_cov] view 만 풀 stereo 렌더
  3. 결과/실패를 요약

scene 폴더명 != rendered_data 폴더명인 경우(green_house->greenhouse,
living_room->livingroom_contemporary)는 정규화 매핑으로 해결.

실행 (venv python):
  python scripts/make_views_dataset.py            # 전체
  python scripts/make_views_dataset.py --scenes minigym,gym,bedroom
  python scripts/make_views_dataset.py --min_cov 15 --max_cov 45 --n_keep 6 --samples 256
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SCENE_ROOT = os.path.join(DATA_ROOT, "source_scenes")
BSF = os.path.join(SCENE_ROOT, "blender_source_files")
RENDERED = os.path.join(SCENE_ROOT, "rendered_data")
OUT_ROOT = os.path.join(DATA_ROOT, "stereo_dataset_views")
VIEW_SCRIPT = os.path.join(SCRIPT_DIR, "render_camera_views.py")


def find_blender():
    txt = os.path.join(PROJECT_ROOT, ".blender_path.txt")
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            p = f.read().strip().strip('"')
        if p and os.path.exists(p):
            return p
    env = os.environ.get("BLENDER")
    if env and os.path.exists(env):
        return env
    from shutil import which
    w = which("blender")
    if w:
        return w
    base = r"C:\Program Files\Blender Foundation"
    if os.path.isdir(base):
        for d in sorted(os.listdir(base), reverse=True):
            exe = os.path.join(base, d, "blender.exe")
            if os.path.exists(exe):
                return exe
    return None


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def list_scene_blends():
    return sorted(glob.glob(os.path.join(BSF, "**", "*.blend"), recursive=True))


def rendered_dirs():
    if not os.path.isdir(RENDERED):
        return {}
    out = {}
    for d in os.listdir(RENDERED):
        full = os.path.join(RENDERED, d)
        if os.path.isdir(full):
            out[_norm(d)] = full
    return out


def resolve_ref_mask(blend_path, rd_map):
    """blend 파일 -> rendered_data/<scene>/masks/inside_mask.png 매핑.

    파일명 정규화 우선, 단 부모 폴더가 blender_source_files 가 아니면(예:
    livingRoom_contemporary/living_room.blend) 부모 폴더명을 우선 시도.
    """
    fname = os.path.splitext(os.path.basename(blend_path))[0]
    parent = os.path.basename(os.path.dirname(blend_path))
    keys = []
    if _norm(parent) != _norm("blender_source_files"):
        keys.append(_norm(parent))   # 부모 폴더 우선 (contemporary 케이스)
    keys.append(_norm(fname))
    for k in keys:
        if k in rd_map:
            mask = os.path.join(rd_map[k], "masks", "inside_mask.png")
            if os.path.exists(mask):
                return mask, os.path.basename(rd_map[k])
    return None, None


def run_scene(blender, blend_path, ref_mask, args):
    scene_name = os.path.splitext(os.path.basename(blend_path))[0]
    out_dir = os.path.join(OUT_ROOT, scene_name)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        blender, "-b", blend_path, "--python", VIEW_SCRIPT, "--",
        "--out_dir", out_dir, "--camera", args.camera,
        "--baseline", str(args.baseline),
        "--width", str(args.width), "--height", str(args.height),
        "--samples", str(args.samples), "--device", args.device,
        "--n_keep", str(args.n_keep),
        "--min_cov", str(args.min_cov), "--max_cov", str(args.max_cov),
    ]
    if ref_mask:
        cmd += ["--ref_mask", ref_mask]
    if args.exposure is not None:
        cmd += ["--exposure", str(args.exposure)]

    # blender 내장 python 이 PIL 없이도 동작하도록(calibration 은 numpy 만 사용).
    env = dict(os.environ)
    print("\n" + "=" * 70)
    print("SCENE:", scene_name, "| ref_mask:", "yes" if ref_mask else "NO(자동탐색)")
    print("=" * 70)
    rc = subprocess.call(cmd, env=env)

    n_views = 0
    idx = os.path.join(out_dir, "views_index.json")
    if os.path.exists(idx):
        try:
            with open(idx, encoding="utf-8") as f:
                n_views = json.load(f).get("n_views", 0)
        except Exception:
            pass
    return {"scene": scene_name, "rc": rc, "n_views": n_views,
            "ref_mask": bool(ref_mask), "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="", help="콤마구분 scene 이름(미지정=전체)")
    ap.add_argument("--camera", default="CM")
    ap.add_argument("--baseline", type=float, default=0.08)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--device", default="OPTIX")
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--n_keep", type=int, default=6)
    ap.add_argument("--min_cov", type=float, default=15.0)
    ap.add_argument("--max_cov", type=float, default=45.0)
    ap.add_argument("--require_ref", action="store_true",
                    help="reference mask 없는 scene 은 건너뜀")
    args = ap.parse_args()

    blender = find_blender()
    if not blender:
        print("ERROR: Blender 못 찾음. setup_local.ps1 실행 또는 .blender_path.txt 설정")
        sys.exit(1)
    print("Blender:", blender)

    rd_map = rendered_dirs()
    blends = list_scene_blends()
    want = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if want:
        wn = {_norm(w) for w in want}
        blends = [b for b in blends
                  if _norm(os.path.splitext(os.path.basename(b))[0]) in wn]

    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for b in blends:
        ref, rdname = resolve_ref_mask(b, rd_map)
        if ref is None and args.require_ref:
            print("SKIP (no ref_mask):", os.path.basename(b))
            results.append({"scene": os.path.splitext(os.path.basename(b))[0],
                            "rc": None, "n_views": 0, "ref_mask": False,
                            "out_dir": None, "skipped": True})
            continue
        results.append(run_scene(blender, b, ref, args))

    print("\n" + "#" * 70)
    print("SUMMARY (views dataset)")
    print("#" * 70)
    total_views = 0
    for r in results:
        if r.get("skipped"):
            print("  [SKIP] %-22s (reference mask 없음)" % r["scene"])
            continue
        status = "OK" if r["rc"] == 0 and r["n_views"] > 0 else "FAIL"
        ref = "ref" if r["ref_mask"] else "auto"
        print("  [%-4s] %-22s views=%d (%s)" %
              (status, r["scene"], r["n_views"], ref))
        total_views += r["n_views"]
    print("TOTAL views:", total_views)

    with open(os.path.join(OUT_ROOT, "dataset_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"results": results, "total_views": total_views,
                   "params": vars(args)}, f, indent=2)
    print("summary ->", os.path.join(OUT_ROOT, "dataset_summary.json"))


if __name__ == "__main__":
    main()
