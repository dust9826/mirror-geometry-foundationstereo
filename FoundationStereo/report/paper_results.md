# Find Mirror Geometry Inside FoundationStereo's Prior — 논문용 결과 정리

작성일: 2026-06-06 · **모든 수치는 원본 결과 파일에서 재집계·검증됨**
(`scripts/verify_paper_numbers.py` → `experiments/_paper_numbers.json`)

> 포함 기준: 일반화 가능한 설정만 (단일 씬 특화 모델, 데이터 결함 씬(loft), 구버전
> 일화적 수치 제외). 모든 평가는 GT 대비 정량 지표.

---

## 1. 실험 설정

| 항목 | 내용 |
|---|---|
| 데이터셋 | 실내 합성 12씬 × 카메라 교란 시드 2세트 = **1,537 스테레오 뷰** (seed0: 770, seed1: 767) |
| 렌더 | 1280×720, fx=fy=1066.67, baseline 0.1 m (실제 intrinsic, 가정 없음) |
| 뷰 구성 | 씬당 55~66뷰: 기준뷰 v00 + 랜덤 교란(±0.3 m 평행이동, ±5° 회전) |
| GT | 뷰별 거울 mask, depth, disparity, **거울평면**(법선/거리, 해석값·depth역투영 교차검증) |
| 스테레오 모델 | FoundationStereo ViT-S (사전학습, **미세조정 없음**) |
| Prior feature | 픽셀당 136-d: side-tuning feat 128 + cost-volume 통계 8 (AHCF 전/후 conf·entropy·argdisp), 1/4 해상도 184×320 |
| 제외 | loft 씬(스테레오 렌더 결함: R=L), 단일 씬 특화 모델(일반화 불가 설정) |

---

## 2. R1 — FS는 거울에서 체계적으로 실패한다 (GT 정량화)

GT disparity/depth 대비 오차를 거울/비거울 영역으로 분리, **뷰별 중앙값**(robust):

| 뷰별 중앙값 | 비거울 (seed0 / seed1) | 거울 (seed0 / seed1) | 비율 |
|---|---|---|---|
| disp MAE (px) | 0.209 / 0.222 | **8.723 / 8.840** | **×41.7 / ×39.8** |
| bad@1px (%) | 2.59 / 2.78 | **98.43 / 98.36** | — |
| bad@3px (%) | 0.48 / 0.52 | **90.18 / 90.44** | — |
| depth MAE (m) | 0.151 / 0.148 | **2.574 / 2.605** | **×17.0 / ×17.6** |
| AbsRel | 0.043 / 0.043 | **0.523 / 0.525** | ×12.2 |

- 두 시드가 사실상 동일 → 카메라 샘플링에 강건한 결론.
- (부기) 픽셀가중 평균: 거울 disp MAE 9.06/9.14 px, bad@1px 96.6%.
  비거울 픽셀가중 평균(6.77/6.52 px)은 FS가 발산한 소수 이상뷰(36/1,537)에 의해
  부풀려짐 — 본문은 중앙값 사용.

![GT eval seed0](figures_seeds/eval_gt_depth_ds_v1_seed0.png)

---

## 3. R2 — Prior 단일 신호는 약하지만 방향은 일관, 학습 조합은 강하다

cost-volume 단일 채널의 거울/비거울 분리력(AUC), **1,537뷰 전체 집계**:

| 신호 | AUC 중앙값 | AUC 평균 |
|---|---|---|
| entropy — AHCF 이전 | 0.545 | 0.551 |
| **entropy — AHCF 이후** | **0.587** | **0.607** |
| confidence — AHCF 이전 | 0.542 | 0.549 |
| confidence — AHCF 이후 | 0.562 | 0.581 |
| 뷰별 최고 post 신호 | 0.591 | — |

- **AHCF 통과 후 신호가 일관되게 상승** (entropy +0.042, conf +0.020 중앙값) —
  거울 단서가 cost aggregation 단계에서 형성된다는 방향성.
- 그러나 단일 채널 AUC ≈ 0.59 는 그 자체로는 약함 → 거울 검출에는 **136-d prior 의
  학습된 조합**이 필요하며(§4), 이 조합은 AUC 0.93~0.96 에 도달한다.
- ※ 구버전 기록의 "단일신호 AUC 0.65~0.70"은 2개 씬 일화였음. 12씬×1,537뷰
  집계가 본 수치를 대체한다.

---

## 4. R3 — 거울 검출: 경량 head 하나로 일반화

픽셀별 136-d prior feature → MLP(136→64→1), 뷰당 4,000px 균형 샘플, 300 epoch.
FS 백본은 고정(추론만). 세 가지 일반화 설정, **동일 프로토콜**:

| 설정 | 학습 | 평가 | AUC | IoU | F1 |
|---|---|---|---|---|---|
| **unseen 씬** (scene_split) | 8씬 전체 뷰 (1,026) | **본 적 없는 4씬** 511뷰 | **0.928** | **0.492** | 0.642 |
| unseen 뷰 (view_split) | 12씬 70% (1,073) | 같은 씬 30% 뷰 464 | 0.958 | 0.546 | 0.692 |
| 학습 12뷰 (rep_view) | **씬당 기준뷰 1장** (12) | unseen 뷰 459 | 0.955 | 0.540 | 0.686 |

보조 관찰:
- **과적합 없음**: view_split train 0.957/0.550 ≈ test 0.958/0.546.
- **unseen 씬 갭이 작음**: scene_split train 0.954/0.557 → test 0.928/0.492
  (AUC −0.026, IoU −0.065) — "거울다움" feature 가 씬 독립적.
- **뷰 다양성은 거의 불필요**: 12장 학습(rep_view)이 1,073장 학습과 동급
  (ΔAUC 0.003) — feature 가 시점 불변임을 시사.
- scene_split test 씬별: blue_bathroom 0.975/0.602, gym 0.943/0.517,
  terrazzo 0.903/0.574, minimal_interior 0.898/0.287 (AUC/IoU).
- 실패 사례 대부분은 AUC 는 높은데(>0.92) 임계 0.5 에서 IoU 가 무너지는
  캘리브레이션 문제 — 임계 적응으로 개선 여지.

대표 예측 (unseen 씬, scene_split):

![scene_split good](figures_seeds/scene_split_good1_ds_v2_seed1__blue_bathroom_v51.png)
![scene_split bad](figures_seeds/scene_split_bad_ds_v2_seed1__minimal_interior_v02.png)

---

## 5. R4 — 거울평면 복원: 학습 없는 반사대칭 최적화

거울 안 FS 점 P′ 와 방 점 Q 의 반사대칭(`P = reflect_M(P′)`)을 이용:
(p′,q) 쌍의 수직이등분면을 RANSAC 가설로, trimmed-NN 정합 스코어 + ICP 정제.
**학습·미세조정 없음, FS 출력만 사용. GT 평면은 평가에만.**

### 5.1 베이스라인 비교 — 1,536뷰 (GT mask)

| 방법 | 법선각 중앙값 | perp 거리오차 중앙값 | 성공률 (<10° & <0.5 m) |
|---|---|---|---|
| 거울 내부 점 평면피팅 (RANSAC, naive) | 87.5° | 2.52 m | **0.1%** |
| **반사대칭 최적화 (제안)** | **2.30°** | **0.137 m** | **71.7%** |

naive 베이스라인은 반사된 씬(바닥/벽)의 평면을 잡아 방향·거리 모두 실패 —
거울 내부 깊이로는 거울면을 직접 회복할 수 없음을 정량 확인.

제안 방법 세부: 법선 <5°: 76%, <10°: **92%** · perp <0.2 m: 57%, <0.5 m: 74%.

![scatter + CDF](figures_seeds/sym_scatter.png)

### 5.2 씬별 성공률 (<10° & <0.5 m)

| 씬 | 성공률 | 씬 | 성공률 |
|---|---|---|---|
| computer_room | **100%** | living_room_contemp | 75% |
| livingroom | 98% | blue_bathroom | 74% |
| minigym | 96% | bedroom | 71% |
| sunlight | 88% | terrazzo | 64% |
| gym | 82% | green_house | 40% |
| minimal_interior | 76% | scandinavian | **2%** |

실패 모드는 기하학적으로 규명됨: **거울이 카메라 뒤(관측 밖) 공간을 비추면**
반사점의 대응이 점군에 존재하지 않아 정합 스코어가 신호를 잃는 퇴화(d-슬라이딩).
scandinavian/green_house 가 이 경우. (개선 방향: mask 경계 링 통과 제약,
가시성 인지 스코어 — §6)

![success](figures_seeds/sym_topdown_good.png)
![failure](figures_seeds/sym_topdown_bad.png)

### 5.3 End-to-end 예비 결과 (예측 mask, 파일럿 n=15)

GT mask 를 학습 검출기의 예측 mask 로 교체(완전 end-to-end: 스테레오 입력만 사용):
3씬 × 5뷰에서 법선각 중앙값 1.50°, perp 오차 중앙값 0.243 m, 성공률 67%.
법선 방향은 mask 품질에 둔감하고, 열화는 주로 거리(mask 오염이 평면을 카메라
쪽으로 편향)에서 발생. ※ 표본이 작아(n=15) 예비 결과로만 제시.

---

## 6. 한계

1. 합성 데이터 단일 도메인(실내, 평면 거울 1개 중심). 실사 검증 미수행.
2. 평면 복원의 퇴화 케이스(거울↔관측영역 겹침 부족): 12씬 중 2씬에서 지배적.
   mask 경계 링(프레임=평면 근방 실기하) 제약, 가시성 인지 스코어, 멀티뷰 합의가
   직접적 개선 후보.
3. End-to-end(예측 mask) 평가는 파일럿 규모(n=15) — 전수 평가 필요.
4. 검출 IoU 는 임계 0.5 고정 기준 — 캘리브레이션/적응 임계 미적용 수치.
5. FS 비거울 영역 발산 뷰 36/1,537 존재(원인 미조사) — R1 은 중앙값으로 보고.

## 부록 — 수치 출처

| 결과 | 원본 파일 |
|---|---|
| R1 | `experiments/<group>/_eval_gt_depth.{csv,json}` (뷰별 중앙값 재계산: `_paper_rq1_medians.json`) |
| R2 | 뷰별 `features/report.json` × 1,537 집계 (`_paper_numbers.json`) |
| R3 | `experiments/_splits/{scene_split,view_split,rep_view}/summary.json` + `applied/metrics_*.csv` |
| R4 | `experiments/_eval_sym.{csv,json}`, 뷰별 `mirror_geometry{,_sym,_sym_submodel}/result.json`, `mirror_gt/gt.json` |
| 검증 스크립트 | `scripts/verify_paper_numbers.py` → `experiments/_paper_numbers.json` |
