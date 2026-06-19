# 재현 가이드 (REPRODUCE)

이 저장소는 **결과물 보관본**이라 무거운 산출물이 빠져 있다. 아래 순서로 외부 의존물을
재취득하면 전체 파이프라인을 다시 돌릴 수 있다. 수치 결론 자체는 이미
`FoundationStereo/experiments/`(result JSON/CSV)·`report/`·`paper/`에 보존돼 있어,
재생성 없이도 논문 수치는 모두 확인 가능하다.

## 0. 무엇이 제외됐나 (용량 정리 시 삭제된 것)

| 제외 항목 | 크기(당시) | 복구 방법 |
|---|---|---|
| `FoundationStereo/experiments/**` 의 `.npy/.npz/.ply/.png` (FS 추론 텐서·포인트클라우드·시각화·뷰별 패널) | ~123GB | 1~4단계 후 파이프라인 재실행 |
| `Archive/` 렌더 데이터셋 (ds_v1_seed0 + ds_v2_seed1) | ~16GB | 2단계(드라이브) 또는 3단계(재렌더) |
| `MakeStereoDataset/data/` (.blend 소스 + 렌더 출력) | ~7GB | 3단계(.blend는 HuggingFace) |
| `FoundationStereo/pretrained_models/` (ViT-L/ViT-S/onnx) | ~4.6GB | 1단계(NVlabs) |
| `FoundationStereo/repo/` (NVlabs 클론) | ~38MB | 1단계 |
| `FoundationStereo/colab/*.zip` (재현 번들) | ~1.7GB | `build_colab_*.py`로 재생성 |

## 1. FoundationStereo 코드 + 가중치

```bash
# 코드: NVlabs 클론 → FoundationStereo/repo
git clone https://github.com/NVlabs/FoundationStereo FoundationStereo/repo
```
- pretrained 가중치는 NVlabs FoundationStereo repo의 다운로드 안내를 따른다.
  본 프로젝트는 `23-51-11`(ViT-L, best)과 `11-33-40`(ViT-S)를 사용했고
  `FoundationStereo/pretrained_models/<id>/` 에 둔다.
- 서브모델 입력 feature는 **반드시 같은 ckpt로 추출**해야 한다(ViT-S=136차원 vs ViT-L=232차원).
  본 프로젝트 기본 파이프라인은 ViT-S(11-33-40, 136d)를 사용.

## 2. 테스트 데이터셋 (빠른 길)

이미 렌더된 아카이브(ds_v1_seed0 / ds_v2_seed1)를 받는다 — **휘발될 수 있음**:
- https://drive.google.com/drive/folders/1nLNR5I0sW88uuq7VjEfwBzC2EQtZPVP5

`Archive/ds_v1_seed0/`, `Archive/ds_v2_seed1/` 로 풀면 된다. 각 뷰는
`L.png/R.png + L_mirror.npy(거울 마스크) + L_depth.npy/disp_gt.npy(GT) + meta.json`
이며, `meta.json`에 실제 intrinsic과 GT 거울평면(world+cam)이 포함돼 K 가정이 불필요하다.

## 3. .blend 소스에서 직접 재렌더 (느리지만 출처가 영구)

거울 합성 씬은 **reflect3r** 데이터셋에서 가져온다:
- https://huggingface.co/datasets/jinggogogo/reflect3r_synthetic_data/tree/main/blender_source_files

`.blend`를 받아 `MakeStereoDataset/data/source_scenes/blender_source_files/` 에 두고
`MakeStereoDataset/scripts/`의 렌더 도구로 뷰 데이터셋을 생성한다
(`make_views_dataset.py` / `render_camera_views.py` / `render_reflect3r_stereo.py`,
거울 GT는 `export_mirror_gt.py`, 마스크 점검·재렌더는 `probe_mirrors.py`/`rerender_mask.py`).
Blender 5.1 헤드리스(`blender --background <blend> --python ...`)로 실행.
seed별 카메라 교란만 다르며 v00은 두 seed 공통 기준뷰다.

## 4. 파이프라인 재실행

```bash
cd FoundationStereo
# (1) 아카이브 뷰 → experiments 입력 변환
python scripts/prepare_views_archive.py --all          # 또는 --scene/--view
# (2) 전체 배치: FS 추론 → 평면추정 → feature분석 → submodel → split 적용
python scripts/run_views_batch.py                      # FS 추론은 --gpu_lock 으로 직렬화
#     (단일 씬: python scripts/run_pipeline_full.py --scene <group>/<scene>_<view>)
# (3) 반사대칭 거울평면 정공법 평가
python scripts/run_sym_batch.py --aggregate            # → experiments/_eval_sym.csv/json
# (4) 논문 수치 재검증
python scripts/verify_paper_numbers.py                 # → experiments/_paper_numbers.json
```

주요 단일 스크립트:
- `run_foundation_stereo.py` / `run_fs_server.py` — FS 추론 + prior 텐서 추출
- `analyze_features.py` — AHCF 전후 feature 분리도(RQ2)
- `submodel_head.py` / `apply_submodel.py` — 픽셀당 136d MLP 검출 head
- `train_eval_splits.py` / `train_eval_extra.py` — view/scene/single/rep-view split 학습·평가
- `estimate_mirror_plane.py` (RANSAC/PCA/IRLS) / `estimate_mirror_plane_sym.py` (반사대칭 정공법)
- `make_mirror_gt.py` / `eval_mirror_plane.py` / `eval_gt_depth.py` — GT 생성·평가
- 단위테스트: `test_estimate_mirror_plane_sym.py`, `test_make_mirror_gt.py`

## 5. Colab 재현

`FoundationStereo/colab/*.ipynb` + `build_colab_drag.py`/`build_colab_zips.py`로 번들 재생성.
번들 안에 open3d 설치 필요(repo `Utils.py` 가 상단에서 import).
