"""
make_stereo_dataset.py  (host orchestrator)

MakeStereoDataset.ipynb 를 로컬(Windows) 실행용 커맨드라인 스크립트로 옮긴 버전.
노트북의 Colab 전용 부분(drive.mount, apt-get, Blender 다운로드, 절대경로)을 제거하고
로컬 경로 + 설치된 Blender 로 동작하도록 했다.

하는 일:
  1. (선택) HuggingFace 에서 reflect3r scene(.blend + 텍스처) 다운로드 -> data/source_scenes
  2. scene 목록 출력 / 선택
  3. 설치된 Blender 를 subprocess 로 호출해 scripts/render_reflect3r_stereo.py 실행
     -> data/stereo_dataset/<scene>/{left,right,mirror_mask_left}.png + stereo_meta.json
  4. (선택) 결과 미리보기 PNG(montage) 저장

venv 의 python 으로 실행한다 (Blender 내장 python 아님):

  # 1) scene 다운로드 (최초 1회, 수 GB)
  python scripts/make_stereo_dataset.py download

  # 2) scene 목록 보기
  python scripts/make_stereo_dataset.py list

  # 3) 한 scene 렌더
  python scripts/make_stereo_dataset.py render --scene cozy_living_room \
      --width 960 --height 540 --samples 32 --baseline 0.08

  # 4) 모든 scene 렌더
  python scripts/make_stereo_dataset.py render --all
"""

import argparse
import glob
import os
import subprocess
import sys

# --------------------------------------------------------------------- paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SCENE_ROOT = os.path.join(DATA_ROOT, "source_scenes")
OUT_ROOT = os.path.join(DATA_ROOT, "stereo_dataset")
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, "render_reflect3r_stereo.py")

HF_REPO_ID = "jinggogogo/reflect3r_synthetic_data"


# --------------------------------------------------------------------- helpers
def find_blender():
    """설치된 Blender 실행파일 경로를 찾는다.

    우선순위: --blender 인자 > 환경변수 BLENDER > .blender_path.txt > PATH >
              C:\\Program Files\\Blender Foundation\\* (최신 버전)
    """
    # .blender_path.txt (setup_local.ps1 이 기록)
    txt = os.path.join(PROJECT_ROOT, ".blender_path.txt")
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            p = f.read().strip().strip('"')
        if p and os.path.exists(p):
            return p

    # 환경변수
    env = os.environ.get("BLENDER")
    if env and os.path.exists(env):
        return env

    # PATH
    from shutil import which
    w = which("blender")
    if w:
        return w

    # 표준 설치 경로 (최신 버전 우선)
    base = r"C:\Program Files\Blender Foundation"
    if os.path.isdir(base):
        dirs = sorted(os.listdir(base), reverse=True)
        for d in dirs:
            exe = os.path.join(base, d, "blender.exe")
            if os.path.exists(exe):
                return exe
    return None


def list_scenes():
    return sorted(glob.glob(os.path.join(SCENE_ROOT, "**", "*.blend"), recursive=True))


def scene_path_by_name(name):
    """scene 이름(확장자 없이)으로 .blend 경로를 찾는다."""
    for p in list_scenes():
        if os.path.splitext(os.path.basename(p))[0] == name:
            return p
    return None


# --------------------------------------------------------------------- commands
def cmd_download(args):
    from huggingface_hub import snapshot_download
    os.makedirs(SCENE_ROOT, exist_ok=True)
    print("Downloading reflect3r scenes -> ", SCENE_ROOT)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=SCENE_ROOT,
        allow_patterns=[
            "blender_source_files/**",
            "textures/**",
            "assets/**",
            "**/*.jpg", "**/*.jpeg", "**/*.png",
            "**/*.exr", "**/*.hdr", "**/*.blend",
        ],
    )
    scenes = list_scenes()
    print(f"\nDownloaded. {len(scenes)} .blend files found.")
    for i, p in enumerate(scenes):
        print(f"  {i:2d}  {os.path.relpath(p, SCENE_ROOT)}")


def cmd_list(args):
    scenes = list_scenes()
    if not scenes:
        print("No .blend files. 먼저 `download` 를 실행하세요.")
        return
    print(f"{len(scenes)} scene(s):")
    for i, p in enumerate(scenes):
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"  {i:2d}  {name:30s}  {os.path.relpath(p, PROJECT_ROOT)}")


def render_one(blender, scene_path, args):
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]
    out_dir = os.path.join(OUT_ROOT, f"{scene_name}_baseline{int(args.baseline * 1000):03d}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        blender, "-b", scene_path,
        "--python", RENDER_SCRIPT, "--",
        "--out_dir", out_dir,
        "--camera", args.camera,
        "--mirror_camera", args.mirror_camera,
        "--mirror_object", args.mirror_object,
        "--baseline", str(args.baseline),
        "--width", str(args.width),
        "--height", str(args.height),
        "--samples", str(args.samples),
        "--device", args.device,
    ]
    if args.exposure is not None:
        cmd += ["--exposure", str(args.exposure)]
    print("\n" + "=" * 70)
    print("RENDER:", scene_name, "->", out_dir)
    print("=" * 70)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"!! Blender exited with code {rc} for scene {scene_name}")
        return None, out_dir

    if args.preview:
        try:
            make_preview(out_dir)
        except Exception as e:
            print("preview skipped:", e)
    return rc, out_dir


def make_preview(out_dir):
    """left/right/mask 를 가로로 붙인 preview.png 저장 (matplotlib 없이 PIL 만)."""
    from PIL import Image
    names = ["left.png", "right.png", "mirror_mask_left.png"]
    imgs = []
    for n in names:
        p = os.path.join(out_dir, n)
        if os.path.exists(p):
            imgs.append(Image.open(p).convert("RGB"))
    if not imgs:
        return
    h = min(im.height for im in imgs)
    imgs = [im.resize((int(im.width * h / im.height), h)) for im in imgs]
    total_w = sum(im.width for im in imgs)
    canvas = Image.new("RGB", (total_w, h), (30, 30, 30))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    out = os.path.join(out_dir, "preview.png")
    canvas.save(out)
    print("preview:", out)


def cmd_render(args):
    blender = args.blender or find_blender()
    if not blender or not os.path.exists(blender):
        print("ERROR: Blender 실행파일을 찾지 못했습니다.")
        print("  setup_local.ps1 을 실행하거나 --blender 로 경로를 지정하세요.")
        sys.exit(1)
    print("Blender:", blender)

    scenes = list_scenes()
    if not scenes:
        print("No .blend files. 먼저 `download` 를 실행하세요.")
        sys.exit(1)

    if args.all:
        targets = scenes
    elif args.scene:
        p = scene_path_by_name(args.scene)
        if not p:
            print(f"scene '{args.scene}' 을(를) 찾지 못했습니다. `list` 로 확인하세요.")
            sys.exit(1)
        targets = [p]
    else:
        # 기본: 첫 번째 scene (노트북과 동일하게 cozy_living_room 우선)
        p = scene_path_by_name("cozy_living_room") or scenes[0]
        targets = [p]

    os.makedirs(OUT_ROOT, exist_ok=True)
    done = []
    for sp in targets:
        rc, out_dir = render_one(blender, sp, args)
        done.append((os.path.basename(sp), rc, out_dir))

    print("\n" + "#" * 70)
    print("SUMMARY")
    for name, rc, out_dir in done:
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  [{status}] {name}  ->  {out_dir}")


# --------------------------------------------------------------------- argparse
def build_parser():
    p = argparse.ArgumentParser(description="reflect3r stereo dataset (local)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="HuggingFace 에서 scene 다운로드")
    sub.add_parser("list", help="다운로드된 scene 목록")

    r = sub.add_parser("render", help="Blender 로 stereo dataset 렌더")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--scene", help="scene 이름 (확장자 없이, 예: cozy_living_room)")
    g.add_argument("--all", action="store_true", help="모든 scene 렌더")
    r.add_argument("--blender", help="blender.exe 경로 (미지정 시 자동 탐색)")
    r.add_argument("--camera", default="CM")
    r.add_argument("--mirror_camera", default="Mirrored_CM")
    # 빈 값이면 렌더 스크립트가 거울 메쉬를 자동 탐색 (scene 마다 이름이 다름)
    r.add_argument("--mirror_object", default="")
    r.add_argument("--baseline", type=float, default=0.08)
    # 1024x576: 16:9(scene 1920x1080 비율 유지) + 가로세로 모두 32배수 ->
    # FoundationStereo(32배수 패딩, max_disp=416) 입력으로 패딩 없이 들어간다.
    r.add_argument("--width", type=int, default=1024)
    r.add_argument("--height", type=int, default=576)
    r.add_argument("--samples", type=int, default=256)
    # RTX 5070 Ti -> OPTIX 권장 (코랩 T4 의 CUDA 와 달리 로컬은 OPTIX 가 빠르고 안정적)
    r.add_argument("--device", default="OPTIX",
                   choices=["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "CPU"])
    r.add_argument("--exposure", type=float, default=None,
                   help="결과가 어두우면 노출 올림(stop). 미지정 시 scene 원본 유지")
    r.add_argument("--no-preview", dest="preview", action="store_false",
                   help="결과 preview.png 생성 안 함")
    r.set_defaults(preview=True)
    return p


def main():
    args = build_parser().parse_args()
    {
        "download": cmd_download,
        "list": cmd_list,
        "render": cmd_render,
    }[args.command](args)


if __name__ == "__main__":
    main()
