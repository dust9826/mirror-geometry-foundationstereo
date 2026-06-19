# FoundationStereo — 로컬(Windows) 실행 + Mirror Geometry 실험

`FoundationStereo.ipynb`(Colab 전용)를 로컬에서 돌릴 수 있게 옮기고,
제안서 **"Find Mirror Geometry Inside FoundationStereo's prior"** 의 실험들을 붙인 버전입니다.

자매 프로젝트 `MakeStereoDataset` 가 만든 stereo dataset(`left/right/mirror_mask`)을
입력으로 받아, **추가 분할 네트워크 없이 FoundationStereo 의 prior 만으로**
거울 기하학(거울 깊이·평면 법선·바운딩 박스)을 추정합니다.

## 목표 (제안서 요약)
스테레오 입력 → FoundationStereo prior → **Mirror Plane Estimation** → MirrorGaussian.
별도 segmentation 학습 없이 다음 3가지를 추정:
1. **Mirror Depth** — 카메라~거울 평면 거리(스칼라)
2. **Plane Normal** — 평면 단위 법선
3. **Bounding Box** — 평면 내 2D 범위 + 이미지 2D bbox

평면 추정은 제안서대로 **세 방향을 병렬**로 구현/비교:
1. **RANSAC 평면 피팅** (반사 outlier 에 강건)
2. **직접 평면 방정식** (inlier 로 closed-form, PCA/SVD)
3. **서브 모델 학습** (FS 특징 위 작은 head)

추가로 제안서의 RQ —
"**AHCF**(Attentive Hybrid Cost Filtering) 전/후 특징 비교로 거울 마스크 신호가
생기는가", "cost volume confidence/entropy 가 거울 영역 단서가 되는가" — 를
정량 분석(`analyze_features.py`)합니다.

## 환경
- Windows 11, Python 3.12, **RTX 5070 Ti (Blackwell, sm_120)**
- **torch 2.7.1 + cu128** 사용. ⚠️ repo `environment.yml` 의 `torch==2.4.1`(cu121)은
  Blackwell 미지원이라 그대로 쓰면 GPU 커널 에러가 납니다.
- `flash-attn` 불필요(코드에서 import 안 함), `xformers` 선택(없으면 표준 attention 폴백).

## 파일 구성
```
setup_local.ps1            # venv + torch(cu128) + requirements + repo clone + 모델 다운로드
requirements.txt           # 호스트(venv) 패키지 (torch 는 cu128 휠로 별도 설치)
repo/                      # (clone) NVlabs/FoundationStereo
pretrained_models/         # (다운) 23-51-11(ViT-L, best) / 11-33-40(ViT-S, fast)
scripts/
  common.py                # 공용 유틸 + 데이터 규약(contract)
  prepare_data.py          # stereo_dataset scene -> 실험 입력(left/right/K.txt/mask)
  run_foundation_stereo.py # FS 추론 + prior 텐서 추출(헤드리스)
  estimate_mirror_plane.py # 거울 평면 3방법(RANSAC/PCA/IRLS) + normal/depth/bbox + reflection
  mirror_geometry.py        # (라이브러리) 평면 피팅/기하 수식
  analyze_features.py       # AHCF 전/후 + cost conf/entropy 가 거울 신호인지 정량분석(RQ)
  submodel_head.py          # 작은 head 로 FS 특징->거울 마스크 학습(접근 #3)
  run_pipeline.py           # 한 scene 전체 파이프라인 한 방 실행
experiments/<scene>/       # (생성) 입력 + 결과
  input/  fs/  mirror_geometry/  features/  submodel/
```

## 사용법

### 0) 1회 세팅
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_local.ps1
```
venv(`.venv`) 생성 → torch cu128 → requirements → `repo/` clone → `pretrained_models/` 다운로드.
모델 다운로드(gdown)가 구글드라이브 quota 로 실패하면, 위 폴더 링크에서 `23-51-11`
폴더(`model_best_bp2.pth` + `cfg.yaml`)를 받아 `pretrained_models/` 에 직접 넣으세요.

이후 명령은 venv python 으로:
```powershell
.\.venv\Scripts\python.exe scripts\<script>.py ...
```

### 1) 입력 scene 목록 / 준비
```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --list
.\.venv\Scripts\python.exe scripts\prepare_data.py --scene cozy_living_room_baseline080
```
`MakeStereoDataset\data\stereo_dataset\` 의 완성된 scene 을 가져와 `experiments\<scene>\input\`
에 정리하고 `K.txt`(가정 intrinsic)를 만듭니다.

> **intrinsic 주의**: `stereo_meta.json` 에는 초점거리 K 가 없어, Blender 카메라 가정
> (기본 lens=50mm, sensor=36mm)으로 K 를 계산합니다. disparity 는 K 무관이고,
> depth/평면거리 **스케일**만 영향받습니다. 정확한 값이 필요하면
> `--lens/--sensor/--fx/--hfov` 로 덮어쓰세요.

### 2) 한 방 실행 (권장)
```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --scene cozy_living_room_baseline080
.\.venv\Scripts\python.exe scripts\run_pipeline.py --all          # 전체 scene
.\.venv\Scripts\python.exe scripts\run_pipeline.py --scene minigym_baseline080 --submodel
```
`prepare_data → run_foundation_stereo → estimate_mirror_plane → analyze_features` 순서로 돕니다.

### 3) 개별 실행
```powershell
# FoundationStereo 추론 (disp/depth/points/cloud + prior 텐서 features.npz)
.\.venv\Scripts\python.exe scripts\run_foundation_stereo.py --scene <scene> --ckpt pretrained_models\23-51-11\model_best_bp2.pth

# 거울 평면 추정 (3방법 비교, result.json/viz.png/plane.txt)
.\.venv\Scripts\python.exe scripts\estimate_mirror_plane.py --scene <scene>

# AHCF 전/후 + cost conf/entropy 가 거울 신호인지 (report.json/panels.png/mirror_signal.npy)
.\.venv\Scripts\python.exe scripts\analyze_features.py --scene <scene>

# 서브 모델 head (단일 scene 데모 또는 멀티 scene 일반화)
.\.venv\Scripts\python.exe scripts\submodel_head.py --scene <scene>
.\.venv\Scripts\python.exe scripts\submodel_head.py --scenes a,b,c --val_scene c
```

## 출력물 (experiments/<scene>/)
- `fs/` : `disp.npy`, `depth_meter.npy`, `points.npy`, `valid.npy`, `conf.npy`,
  `entropy.npy`, `features.npz`(AHCF 전/후 분포 통계 + side-tuning feature), `vis_disp.png`, `cloud.ply`
- `mirror_geometry/` : `result.json`(3방법 normal/d/depth/rms + bbox + 4x4 reflection),
  `viz.png`, `plane.txt`(MirrorGaussian 핸드오프용 평면+반사행렬)
- `features/` : `report.json`(신호별 거울 분리 AUC 랭킹), `panels.png`, `mirror_signal.npy`
- `submodel/` : `metrics.json`(AUC/IoU/F1), `pred_mask.png`, `panel.png`, `head.pt`

## MirrorGaussian 핸드오프
`estimate_mirror_plane.py` 가 평면 법선 `n`, 거리 `d`, 그리고 평면에 대한
**4×4 Householder 반사행렬**(`plane.txt`/`result.json`)을 출력합니다.
이 행렬로 3D Gaussian 을 거울면 기준으로 미러링하면 MirrorGaussian 의 반사 항으로
바로 넣을 수 있습니다.

## 코랩 → 로컬 변경점
- `drive.mount`, `pip install`(노트북 내), `gdown` 인라인 → `setup_local.ps1` 로 이동
- 절대경로(`/content/...`, Google Drive) → 프로젝트 상대경로
- torch: cu121 2.4.1 → **cu128 2.7.1** (Blackwell)
- Open3D GUI 시각화(헤드리스에서 죽음) → 제거, `cloud.ply` 저장만
- 데이터셋: `MakeStereoDataset` 의 완성 stereo dataset 을 입력으로 사용

## 알려진 한계 / 메모
- intrinsic K 는 가정값(위 주의 참고). 절대 깊이가 중요하면 실제 K 를 넣어야 함.
- FoundationStereo 는 거울에서 (반사 강도에 따라) 거울 표면 깊이 또는 반사된 씬의 깊이를
  예측할 수 있음 — 제안서가 분석하려는 바로 그 지점. `mirror_geometry` 의 residual 통계,
  `analyze_features` 의 conf/entropy 로 어느 쪽인지 확인 가능.
- `xformers` 미설치 시 DINOv2 가 표준 attention 으로 동작(느릴 수 있으나 결과 동일).
