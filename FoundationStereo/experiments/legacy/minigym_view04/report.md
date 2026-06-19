# Mirror Geometry Report — `minigym_view04`

FoundationStereo prior 기반 거울 검출 · AHCF 신호 분석 · 거울 평면 추정 (단일 scene)

| 지표 | 값 |
|---|---|
| 검출 AUC (단일 scene) | **0.998** |
| 검출 AUC (범용모델 적용) | **0.947** |
| 거울 마스크 비율 | 44.0% |
| 평면 RANSAC rms | 0.055 m |

## 1. 입력
스테레오 쌍(baseline 0.08m) + 거울 마스크 GT. `minigym` 장면의 **view 04** 카메라(베이스라인과 다른 시점) — 교차-시점 일반화 평가용.

![left](input/left.png) ![mask](input/mirror_mask.png)

## 2. FoundationStereo 추론
FS 1회 추론 → disparity/depth + prior(side-tuning 특징, AHCF 전/후 cost-volume 통계).

![disp](fs/vis_disp.png)

## 3. AHCF 신호 분석 (RQ)
| 순위 | 신호 | AUC | polarity |
|---|---|---|---|
| 1 | feat_left_l2 | **0.628** | + |
| 2 | feat_left_pc1 | 0.621 | − |
| 3 | delta_argdisp | 0.612 | + |
| 4 | post_conf | 0.609 | − |

이 시점에서는 side-tuning 특징이 가장 강한 거울 신호. 검출 헤드가 이를 결합해 고성능.

![ahcf](features/panels.png)

## 4. 거울 검출 — FS prior 위 경량 헤드
| 설정 | AUC | IoU@0.5 | F1@0.5 |
|---|---|---|---|
| 단일 scene (70/30 분할) | **0.998** | 0.953 | 0.976 |
| 범용 모델 적용 (baseline 12 scene 학습 → 평가) | **0.947** | 0.737 | 0.849 |

범용 모델은 12개 baseline scene으로만 학습. 전혀 다른 시점 view04에서 AUC 0.947 → FS prior가 시점 무관 "거울다움"을 인코딩(교차-시점 일반화).

![submodel single](submodel/panel.png) ![submodel applied](submodel_applied/panel.png)

## 5. 거울 평면 추정 (RANSAC)
거울 마스크 영역 3D 점(255,521) 평면 적합. 좌측 카메라 OpenCV, `n·x + d = 0`.

| 항목 | 값 |
|---|---|
| 법선 n (RANSAC) | [0.002, −0.990, −0.138] |
| 수직거리 | 1.985 m |
| 잔차 RMS — PCA vs RANSAC | 0.920 m vs **0.055 m** |
| 평면 내 bbox | 11.23 × 6.12 m |

**주의 — 반사면 ≠ 거울면.** 적합 평면은 반사면(거리·bbox는 반사 깊이). 거울면 GT는 `minigym.blend`+view04 카메라로 생성했으나, minigym 거울이 큐브(`Cube.001~003`)라 평면 GT planarity_rms 1.0m로 **신뢰 불가**(알려진 한계).

![plane viz](mirror_geometry/viz.png) ![ransac inliers](mirror_geometry/ransac_inliers.png)

## 6. 요약
- 검출 매우 강함 — 단일 0.998, 범용 적용 0.947(교차-시점).
- AHCF 신호는 이 시점에서 약하나(최고 0.63) 존재.
- 평면 RANSAC은 강건(0.055m)하나 반사면 — 거울면은 정공법 + GT 필요(큐브 거울이라 GT 불안정).
