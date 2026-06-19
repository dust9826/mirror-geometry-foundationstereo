"""
common.py — FoundationStereo 로컬 프로젝트 공용 유틸.

이 파일은 모든 실험 스크립트(run_foundation_stereo / mirror_geometry /
analyze_features / submodel_head)가 공유하는 "계약(contract)"을 정의한다.

핵심 데이터 규약 (experiments/<run>/ 아래):
  input/
    left.png, right.png          : stereo pair
    K.txt                        : FoundationStereo 형식 intrinsic (1행 K 9값, 2행 baseline[m])
    mirror_mask.png              : 거울 영역 GT 마스크 (흰=거울)
    prepare_meta.json            : 원본 scene/baseline/K 출처 등
  fs/                            : run_foundation_stereo.py 출력
    disp.npy                     : (H,W) float32 disparity (px)
    depth_meter.npy              : (H,W) float32 metric depth (m)
    points.npy                   : (H,W,3) float32 카메라좌표 XYZ
    valid.npy                    : (H,W) bool  유효 픽셀(z>0, 비가림)
    conf.npy                     : (H,W) float32 cost-volume softmax 최대확률(신뢰도)
    entropy.npy                  : (H,W) float32 cost-volume softmax 엔트로피
    features.npz                 : AHCF 전/후 등 prior 텐서 (analyze_features 용)
    vis_disp.png, cloud.ply
  mirror_geometry/, features/, submodel/ : 각 실험 출력

intrinsic 주의:
  stereo_meta.json 에는 K(초점거리)가 저장돼 있지 않다(렌더 스크립트가 extrinsic/
  baseline 만 기록). 따라서 K 는 Blender 카메라 가정(기본 lens=50mm, sensor=36mm,
  가로 맞춤)으로부터 계산한다. 정확한 K 가 필요하면 prepare_data 의 --lens/--fx/--hfov
  로 덮어쓰거나, tools/dump_intrinsics.py 로 .blend 에서 추출해 넣는다.
  disparity 자체는 K 와 무관하므로 영향 없고, depth 스케일/평면 거리만 K 에 의존한다.
"""

import json
import os
import sys

import numpy as np

# Windows 콘솔(cp949)에서 한글/유니코드(—, →, ≈ 등) 출력 시 UnicodeEncodeError 가
# 나지 않도록 stdout/stderr 를 UTF-8 로 재설정한다. 모든 스크립트가 common 을
# import 하므로 여기서 한 번만 처리하면 전체에 적용된다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------- paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.join(PROJECT_ROOT, "repo")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")

# MakeStereoDataset (자매 프로젝트)의 완성된 stereo dataset 위치
STEREO_DATASET_DIR = os.path.normpath(
    os.path.join(PROJECT_ROOT, "..", "MakeStereoDataset", "data", "stereo_dataset")
)


# --------------------------------------------------------------------- intrinsics
def intrinsics_from_assumption(width, height, lens_mm=50.0, sensor_mm=36.0,
                               fx=None, fy=None, cx=None, cy=None, hfov_deg=None):
    """Blender 카메라 가정으로 3x3 K 를 만든다.

    우선순위: 명시 fx/fy > hfov_deg > lens/sensor.
    sensor_fit=HORIZONTAL 기준 fx = lens/sensor_width * W, 정사각 픽셀(fy=fx).
    """
    if fx is None:
        if hfov_deg is not None:
            fx = (width * 0.5) / np.tan(np.radians(hfov_deg) * 0.5)
        else:
            fx = lens_mm / sensor_mm * width
    if fy is None:
        fy = fx
    if cx is None:
        cx = width * 0.5
    if cy is None:
        cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def write_K_txt(path, K, baseline):
    """FoundationStereo run_demo 형식 K.txt 저장 (1행: K 9값, 2행: baseline)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.8f}" for v in K.reshape(-1)) + " \n")
        f.write(f"{float(baseline)}\n")


def read_K_txt(path):
    """K.txt 를 읽어 (K(3,3), baseline) 반환."""
    with open(path) as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].split()))).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline


# --------------------------------------------------------------------- io
def load_mask(path, thresh=0.5):
    """거울 마스크 PNG 를 (H,W) bool 로 로드 (흰=True)."""
    import imageio.v2 as imageio
    m = imageio.imread(path)
    if m.ndim == 3:
        m = m[..., :3].mean(axis=-1)
    return (m.astype(np.float32) / 255.0) > thresh


def load_meta(run_dir):
    p = os.path.join(run_dir, "input", "prepare_meta.json")
    with open(p, encoding="utf-8") as f:   # UTF-8 명시 (Windows cp949 기본 회피)
        return json.load(f)


# --------------------------------------------------------------------- geometry
def depth_to_points(depth, K, zmin=1e-6):
    """(H,W) depth -> (H,W,3) 카메라좌표 XYZ (z 전방, OpenCV 규약)."""
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    return xyz


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d


def run_dir_for(scene, root=None):
    root = root or EXPERIMENTS_DIR
    return os.path.join(root, scene)
