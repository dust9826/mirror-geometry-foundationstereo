# 실험 노트 — Find Mirror Geometry Inside FoundationStereo's prior

제안서의 3가지 평면 추정 방안 + 2가지 연구질문(RQ)을 구현하고, 자매 프로젝트
`MakeStereoDataset` 의 stereo dataset 에 대해 돌린 결과 메모.

## 데이터
입력: `MakeStereoDataset/data/stereo_dataset/<scene>/` (left/right/mirror_mask_left).
- 거울은 **물리(레이트레이싱) 반사**로 렌더됨 → FoundationStereo 가 거울을 통해
  **반사된 씬의 깊이**를 보는 어려운(그리고 제안서가 노린) 케이스.
- GT mirror mask 가 정상인 scene(>10%): computer_room, gym, loft, livingroom,
  blue_bathroom, sunlight, archiviz, flat-archiviz, scandinavian, green_house,
  cozy_living_room, minimal_interior 등.
- ⚠️ mask 가 비어 있는 scene(렌더 시 거울 흰색표시 실패): minigym, living_room,
  terrazzo, modern_living_room → 평면 추정은 skip 처리(파이프라인은 계속 진행).
- intrinsic K 는 가정값(lens=50mm). depth 절대 스케일에만 영향.

## 접근 1·2: 기하학적 평면 추정 (estimate_mirror_plane.py)
거울 마스크 영역의 FoundationStereo 3D 점에 평면을 적합.
- **PCA / 직접 평면 방정식(접근 2, closed form)**: 반사 outlier 에 취약.
- **RANSAC(접근 1)**: 반사 outlier 에 강건. consensus inlier 로 재적합.
- **IRLS(보너스)**: RANSAC seed + Huber 가중 정밀화.

### cozy_living_room_baseline080 결과
| 방법 | normal | rms residual | inliers |
|---|---|---|---|
| PCA(직접) | (0.078, −0.970, −0.231) | **0.800 m** | 77798/77798 |
| RANSAC | (0.875, 0.268, −0.403) | **0.045 m** | 30856/77798 |
| IRLS | (0.026, −0.967, −0.253) | 0.117 m | 16602/77798 |

→ 제안서 가설대로 **PCA 는 반사 outlier 로 오염(rms 0.8m), RANSAC 은 강건(rms 0.045m)**.
   거울 영역의 view-axis 깊이 ≈ 21 m → FoundationStereo 가 거울 표면이 아니라
   **반사된 씬(깊은 곳)** 을 예측함을 정량 확인. (MirrorGaussian 핸드오프용 4×4
   반사행렬은 `mirror_geometry/plane.txt`.)

## RQ1: AHCF 전/후 특징이 거울 마스크 신호를 주는가 (analyze_features.py)
AHCF(corr_feature_att + cost_agg) **이전(corr_stem)** 과 **이후(cost_agg)** cost volume 에
동일 classifier 를 적용, disparity 분포의 confidence/entropy 를 GT 마스크로 분리도(AUC) 평가.

### cozy_living_room_baseline080 결과 (feature 144×256, AUC)
| 신호 | AUC | 극성 |
|---|---|---|
| **post_entropy** (AHCF 후) | **0.657** | 거울에서↑ |
| **post_conf** (AHCF 후) | **0.653** | 거울에서↓ |
| delta_conf | 0.593 | |
| pre_entropy (AHCF 전) | 0.513 | ~chance |
| pre_conf (AHCF 전) | 0.511 | ~chance |
| feat_left (side-tuning) L2/PC1 | 0.52 | 약함 |

→ **핵심 발견**: AHCF **이전** cost volume 은 거울을 거의 구분 못 함(AUC≈0.51, 우연수준).
   AHCF **이후** confidence/entropy 는 거울 분리도가 **0.65 로 상승**.
   즉 *Attentive Hybrid Cost Filtering 이 거울 영역 신호를 실제로 만들어낸다* —
   제안서 RQ1 에 대한 긍정적 증거. (거울은 photometric consistency 를 깨므로
   AHCF 후 matching 신뢰도가 낮고 entropy 가 높음.)
   시각 확인: `features/panels.png` 에서 post_conf/post_entropy 만 거울 구조가 보임.

## 접근 3: 서브 모델 head (submodel_head.py)
FoundationStereo 특징(side-tuning feat_left 224ch + cost conf/entropy 통계) 위에
작은 MLP head 만 학습(FS 동결)하여 per-pixel 거울 확률 예측.

### cozy_living_room_baseline080 (단일 scene, pixel 70/30 split)
- AUC **0.999**, IoU@0.5 **0.919**, F1 **0.958**, acc 0.989.
→ FS prior 특징만으로 작은 head 가 거울을 거의 완벽 분리 → **무거운 segmentation
   네트워크 불필요**라는 제안서 주장 뒷받침.
- ⚠️ 단일 scene pixel split 은 공간적 누설 위험 → 진짜 일반화는 멀티 scene 평가
  (`--scenes a,b,c --val_scene c`)로 측정해야 함. (TODO: 정상 mask scene 들로 LOSO.)

## 전체 dataset 결과 (`run_pipeline.py --all`, 17 scene)
RANSAC rms 는 거의 모든 scene 에서 cm 단위(0.01~0.06 m)로, PCA(0.12~0.92 m)보다 한 자릿수
작음 → **반사 outlier 강건성 일관 확인**. 최적 거울 신호는 대부분 AHCF 유도 특징
(post_*/delta_*)이며 AUC 0.55~0.80. 서브모델 head 는 모든 scene 에서 AUC 0.96~0.999.

(아래 표는 `experiments/_summary_table.md` 에도 저장됨. submodel 은 단일 scene pixel split.)

| scene | PCA rms(m) | RANSAC rms(m) | best AHCF signal (AUC) | submodel AUC/IoU |
|---|---|---|---|---|
| archiviz | 0.427 | **0.047** | delta_conf (0.55) | 0.992/0.85 |
| bedroom | 0.153 | **0.022** | feat_left_l2 (0.59) | 0.960/0.37 |
| blue_bathroom | 0.291 | **0.034** | delta_argdisp (0.67) | 0.995/0.87 |
| computer_room | 0.430 | **0.027** | delta_entropy (0.68) | 0.988/0.85 |
| cozy_living_room | 0.804 | **0.040** | post_conf (0.59) | 0.999/0.92 |
| flat-archiviz | 0.427 | **0.047** | delta_conf (0.55) | 0.992/0.85 |
| green_house | 0.464 | **0.026** | post_entropy (0.74) | 0.985/0.73 |
| gym | 0.919 | **0.062** | delta_argdisp (0.60) | 0.989/0.85 |
| livingroom | 0.270 | **0.028** | post_entropy (0.65) | 0.977/0.73 |
| minimal_interior | 0.206 | **0.012** | delta_argdisp (0.65) | 0.998/0.86 |
| modern_living_room | 0.124 | **0.040** | feat_left_pc1 (0.80) | (점 95개, 표본부족) |
| scandinavian | 0.644 | **0.051** | post_entropy (0.75) | 0.996/0.88 |
| sunlight | 0.707 | **0.047** | feat_left_l2 (0.57) | 0.989/0.83 |
| living_room / loft / minigym / terrazzo | — | — | — | skip(빈 mask 또는 반사 매칭 실패) |

> loft 는 mask 가 있지만 거울 반사가 너무 멀어(FS disparity≈0) 유효 깊이 점이 없어 skip.
> `--max_depth`(기본 50m)로 disparity≈0 깊이폭발 점을 제거한다.

## 접근 3 — 진짜 일반화: LOSO (Leave-One-Scene-Out) cross-scene
단일 scene pixel-split(AUC~0.99)은 같은 이미지 내 공간 누설이 있어 부풀려진 값.
**한 scene 을 통째로 빼고(unseen) 나머지 11개로만 학습 → 빠진 scene 으로 평가**.
(`submodel_head.py --scenes <12개> --val_scene <X>`, 출력 `experiments/_loso/<X>/`)

| held-out (unseen) | AUC | IoU | F1 |
|---|---|---|---|
| archiviz | 0.968 | 0.684 | 0.812 |
| bedroom | 0.731 | 0.053 | 0.101 |
| blue_bathroom | 0.965 | 0.420 | 0.591 |
| computer_room | 0.956 | 0.689 | 0.816 |
| cozy_living_room | 0.931 | 0.245 | 0.393 |
| flat-archiviz | 0.968 | 0.684 | 0.812 |
| green_house | 0.956 | 0.426 | 0.597 |
| gym | 0.947 | 0.598 | 0.749 |
| livingroom | 0.864 | 0.424 | 0.596 |
| minimal_interior | 0.893 | 0.344 | 0.511 |
| scandinavian | 0.873 | 0.437 | 0.608 |
| sunlight | 0.884 | 0.450 | 0.621 |
| **평균 / 중앙값** | **0.911 / 0.939** | **0.454 / 0.431** | |

→ **정직한 일반화 결론**: 한 번도 못 본 scene 에서도 거울 영역 분리 AUC 평균 **0.91**(중앙값 0.94).
   즉 FS prior 특징은 scene 을 외운 게 아니라 *거울다움* 자체를 인코딩 → 제안서 핵심 주장 확증.
   다만 픽셀 정밀도(IoU 0.45)는 within-scene(0.7~0.9)보다 떨어짐: 임계 0.5 고정 + 작은 거울
   (bedroom 3.6% → AUC 0.73/IoU 0.05)에서 약함. scene별 임계 튜닝/후처리로 개선 여지.
   (archiviz·flat-archiviz 는 거의 동일 scene 쌍이라 서로 학습에 들어가 약간 낙관적.)

## ⚠️ 중요 한계: RANSAC 평면 = 거울면이 아니라 "반사면"
거울 안쪽 점에 평면을 맞추면 거울 표면이 아니라 **거울에 비친 반사 장면(가상 이미지)** 에
평면이 맞는다. 데이터 확인: 거울 안 깊이가 테두리(거울이 박힌 벽)보다 1.6~1.7배 깊음
(minigym_view04 15.4m vs 8.9m / computer_room 8.9 vs 5.5 / cozy 21.4 vs 13.4).
- 이유: 평면거울은 빛을 반사 → 스테레오는 거울 뒤 가상 이미지를 봄. 가상 점들도 평평한
  면(반사된 벽)을 이뤄 rms 가 작게(~3cm) 나오지만 그건 **반사면**이지 거울이 아님.
- 거울 평면 M = "진짜 점과 반사점의 수직이등분면". 반사점만으론 M 복원 불가.
- 즉석 우회(거울 테두리 링 RANSAC)도 주변 바닥/타 벽이 섞여 실패(minigym/computer_room 은
  법선 (0,1,0)=바닥을 잡음). 단순 방법으론 안 됨.
- **제대로 된 방향**: (1) 반사 대칭 — reflect_M(거울안 점) ≈ 방 점구름 이 되도록 M 최적화(ICP류,
  대응 필요), (2) 거울 silhouette 경계 깊이만 정밀 추출, (3) 서브모델 head 가 평면 파라미터까지
  GT 로 회귀 학습.
- **결론(제안서 난제 실증)**: 거울 *검출*(mask/bbox)은 FS prior(AHCF/submodel)로 잘 됨.
  거울 *평면 기하*는 추가 기하 제약 없이는 안 됨 → "순수 스테레오론 거울 기하 못 잡는다"는
  제안서 문제정의가 그대로 확인됨. estimate_mirror_plane 의 평면은 "반사면" 으로 해석할 것.

## 범용 모델 1개 학습 → 새 데이터 적용 (apply_submodel.py)
LOSO 12-fold 와 별개로, "head 하나를 학습해 저장하고 새 데이터에 적용"하는 배포형 흐름.
  - 학습: `apply_submodel.py train --train_scenes <12개> --model_out experiments/_model/general_head.pt`
    (held-out 없이 12개 stereo_dataset scene 전부로 학습. 442,368 픽셀, 입력 136차원)
  - 적용: `apply_submodel.py apply --load <모델> --scene <새 scene>`

### 새 데이터 minigym_view04 적용 결과 (학습에 전혀 안 쓰인 외부 scene)
| | AUC | IoU | F1 | accuracy |
|---|---|---|---|---|
| minigym_view04 | **0.947** | **0.737** | 0.849 | 0.863 |
→ 다른 scene 류(체육관, 거울 3개, 다른 카메라 각도)인데도 AUC 0.95. prior 가 scene 이 아니라
  "거울다움"을 인코딩한다는 가장 강한 증거(LOSO 보다 더 외부 데이터). 산출물:
  `experiments/minigym_view04/submodel_applied/{pred_mask,panel,metrics}`.

### 서브모델 입력 = FoundationStereo prior (픽셀당 136차원)
- feat_left 128 (side-tuning 특징, ViT-S 기준) + cost volume 통계 8개
  (pre/post_conf, pre/post_entropy, pre/post_argdisp, delta_conf, delta_entropy).
- RGB/좌표/깊이는 안 넣음. **오직 FS 내부 prior 만 입력**, GT mask 는 라벨(정답).
  AUC 높음 = prior 에 거울 정보가 실제로 들어있다는 직접 증거(제안서 가설 검증).

### ⚠️ 체크포인트 일관성 (중요)
서브모델 입력 prior 는 **같은 FoundationStereo 체크포인트로 뽑아야** 한다.
- ViT-S(11-33-40): feat_left 128 → 입력 136차원
- ViT-L(23-51-11): feat_left 224 → 입력 232차원
→ 섞으면 차원 불일치로 적용 불가. run_pipeline.py 는 `--ckpt` 미지정 시 pretrained_models 에서
  자동 선택(현재 11-33-40 우선)하므로, --all 로 만든 12개 scene 은 ViT-S(136) 기준.
  새 데이터도 **반드시 같은 ckpt(11-33-40)** 로 run_foundation_stereo 해야 general_head.pt 가 먹는다.
  (minigym_view04 도 처음 ViT-L 로 뽑아 232차원이 나와 불일치 → ViT-S 로 재생성 후 적용 성공.)

## RANSAC inlier/outlier 시각화
`viz_ransac_inliers.py --scene <X>` → `mirror_geometry/ransac_inliers.png`:
좌영상 위 inlier(초록)/outlier(빨강), top-down X-Z 산점도 + 적합 평면, 잔차 히스토그램.
예: computer_room 188358점 중 inlier 57751(30.7%)만 평면에 남고 반사 outlier 는 꼬리로 분리.

## 거울 평면 기하의 정공법 — 반사 대칭 (다음 핵심 작업, 미구현)
거울면 M 은 평면 피팅이 아니라 **반사 대칭(reflection symmetry)** 으로 풀어야 한다.
- 물리: 실제 점 P 와 거울 속 상 P′ 은 M 에 대한 거울 반사 관계. → **M = 선분 P–P′ 의 수직이등분면**.
  대응쌍의 연결선은 모두 M 에 수직(평행)이고, 중점들은 모두 M 위에 있음.
- 데이터 연결: 거울 밖 FS 점 = 실제 방 {P}, 거울 안 FS 점 = 가상 점 {P′}.
- 방법: "reflect_M({P′}) 가 {P} 와 가장 잘 겹치는 M" 을 찾기 — M 후보 RANSAC + ICP 류 정합,
  또는 대응쌍 3개로 수직이등분면 closed-form. **학습 불필요, 순수 기하로 가능.**
- 어려운 점(조건부 가능): (1) 대응 — 반사는 handedness 뒤집어 일반 매칭 어려움,
  (2) 거울이 직접 안 보이는 영역을 비추면 대응할 P 없음, (3) 반사가 평평한 벽뿐이면 퇴화(M 무수).
- 권장 파이프라인: **submodel/AHCF 로 mask 검출 → 그 영역의 {P′} 만 골라 반사 대칭으로 M 추정.**
  (검출=prior/학습, 평면=기하 — 제안서의 "prior 로 찾고 기하로 평면" 조합.)

## 다음 할 일
1. ✅ 정상 mask scene 전체 평면 추정 + 표 집계(`run_pipeline.py --all`).
2. ✅ 서브모델 LOSO 12-fold 일반화(평균 AUC 0.91) + 범용모델→외부데이터(minigym_view04 AUC 0.95).
3. ⬜ **반사 대칭 기반 거울 평면 추정기 프로토타입**(위 섹션) — 현재 RANSAC "반사면" 한계의 정답.
4. ⬜ RQ2: 거울 경계의 에피폴라/disparity 불연속(직접뷰↔반사뷰 불일치)에서 방향 단서 추출.
5. ⬜ MirrorGaussian 에 plane.txt 반사행렬 연동 데모.
6. ⬜ (데이터) 비정상 mask scene 재렌더 — MakeStereoDataset 쪽. minigym 은 stereo_dataset_views 로 대체 가능.
7. ⬜ 서브모델 IoU 개선: scene별 임계 튜닝 + 공간 후처리(연결요소/CRF), 작은 거울 대응.
