# 반사대칭 기반 거울평면 추정 (정공법) — 결과 보고서

작성일: 2026-06-06 · 프로젝트: Find Mirror Geometry Inside FoundationStereo's Prior
구현: `scripts/estimate_mirror_plane_sym.py` · 평가 데이터: ds_v1_seed0 + ds_v2_seed1 (12씬 × 2seed)

---

## 1. 문제 — RANSAC 평면은 거울면이 아니다

FoundationStereo(FS)는 거울을 "창문"처럼 통과해 **반사된 씬의 깊이**를 본다.
그래서 거울 마스크 안 3D 점들에 평면을 피팅하면 거울 유리면이 아니라 거울 *뒤*
가상공간의 면("반사면")이 잡힌다 — 실측으로 진짜 거울면보다 **약 2배 깊다**.

| 예: computer_room_v00 | 수직거리(perp) |
|---|---|
| 거울 안 점 RANSAC (기존) | 6.96 m ← 반사면 |
| **반사대칭 정공법 (본 방법)** | **3.23 m** |
| GT 거울면 | 3.24 m |

## 2. 방법 — 반사 대칭 + 수직이등분면 RANSAC

거울 안 가상점 P′ 와 방의 실제점 P 는 거울평면 M 에 대해 완벽한 대칭이다:

```
P = reflect_M(P′) = P′ − 2(n·P′ + d)·n        (Householder 반사)
```

핵심 성질: **대응쌍 (P′, P) 하나를 알면 평면이 닫힌형으로 결정**된다
(두 점의 수직이등분면: `n=(P−P′)/‖P−P′‖`, `d=−n·(P+P′)/2`).

알고리즘 (학습 불필요, FS 출력만 사용):

1. **입력 분리**: mask 침식 후 거울 안 점 P′(≤4,000), mask 팽창 밖 방 점 Q(≤80,000, KD-tree)
2. **가설 생성**: 랜덤 (p′∈P′, q∈Q) 쌍 4,000개 → 각 쌍의 수직이등분면
   (perp 범위 제약: 0.3m ~ 가상점 깊이 90퍼센타일)
3. **2단계 스코어**: P′ 샘플을 가설로 반사 → Q 와의 NN 거리 **하위 50% 평균(trimmed)**
   — 거울은 카메라 뒤(관측 밖)도 비추므로 대응 없는 점 절사가 필수.
   150pt로 전수 평가 → 상위 40개만 600pt 재평가 (7배 가속, 뷰당 ~10초)
4. **ICP식 정제**: 상위 5개 가설에 대해 8회 반복 — 반사 → NN 대응(하위 60%) →
   닫힌형 재추정(방향 주성분 + 중점 median)
5. **평가**: GT 평면 대비 법선 각도(°), 수직거리 오차(m) — GT는 평가에만 사용

검증: 단위테스트 6/6 (`test_estimate_mirror_plane_sym.py` — 반사 involution,
이등분면 복원, 닫힌형 정제 정밀도, 합성 방 복원: 노이즈 2cm + 무대응 40%에서도 <3°).

## 3. 전체 결과 — 1,536뷰 (GT mask)

| 지표 | 값 |
|---|---|
| 법선각 중앙값 | **2.3°** (<5°: 76%, <10°: **92%**) |
| perp 거리오차 중앙값 | **0.14 m** (<0.2m: 57%, <0.5m: 74%) |
| 동시 충족 (<10° & <0.5m) | **72%** |
| 처리 | 8샤드 병렬, 뷰당 ~10초, 실패 1뷰(점 부족 스킵) |

![추정 vs GT 산점도 + 법선각 CDF](figures_seeds/sym_scatter.png)

### 씬별 성능

| 씬 | n | 각도 med | perp med | 성공률(<10°&<0.5m) |
|---|---|---|---|---|
| computer_room | 132 | 1.4° | 0.04m | **100%** |
| livingroom | 129 | 1.0° | 0.05m | 98% |
| minigym | 111 | 2.2° | 0.08m | 96% |
| sunlight | 132 | 1.1° | 0.13m | 88% |
| gym | 132 | 2.3° | 0.07m | 82% |
| minimal_interior | 130 | 1.8° | 0.10m | 76% |
| living_room_contemp | 132 | 2.2° | 0.16m | 75% |
| blue_bathroom | 116 | 0.9° | 0.19m | 74% |
| bedroom | 129 | 2.9° | 0.14m | 71% |
| terrazzo | 132 | 4.2° | 0.40m | 64% |
| green_house | 129 | 5.1° | 0.61m | 40% |
| **scandinavian** | 132 | 7.0° | **2.09m** | **2%** |

## 4. 성공/실패 기하 분석

### 성공 — 반사점이 방 점군과 포개짐

computer_room_v00 (각도 2.2°, perp 오차 0.01m). 거울 안 가상점(빨강)을 추정 평면으로
반사하면(초록) 방 점군(회색)과 정확히 정합하고, 추정 평면(주황)이 GT(파랑)와 일치:

![성공 예 top-down](figures_seeds/sym_topdown_good.png)

### 실패 — d-슬라이딩 퇴화 (scandinavian형)

scandinavian_v10 (perp 오차 2.30m). **GT 평면으로 반사하면 점들이 카메라 뒤
(관측되지 않은 공간)로 간다** → 방 점군에 대응이 존재하지 않음 → trimmed NN 스코어가
신호를 잃고, 반사점을 "보이는 먼 벽"에 끼워 맞추는 더 깊은 평면을 선택:

![실패 예 top-down](figures_seeds/sym_topdown_bad.png)

green_house(40%)·terrazzo(64%)도 같은 계열의 약화 — 거울이 비추는 영역과 카메라
관측 영역의 겹침이 작을수록 취약하다.

## 5. 결론 및 개선 방향

**결론**: 스테레오 입력 → FS prior → 거울 mask → **반사대칭 최적화**만으로 진짜
거울평면(유리면)이 복원된다. 학습 없이 1,536뷰에서 법선 중앙값 2.3°/거리 0.14m.
프로젝트 가설("FS prior 안에 거울 기하가 들어 있다")의 마지막 단계가 입증됨.

**개선 방향** (실패 모드 대응):
1. **mask 경계 링 제약**: 거울 프레임/주변 벽의 3D 점은 거울평면 근방의 *실제* 기하
   → "평면이 경계 링을 지나야 한다"는 제약 추가 시 d-슬라이딩 차단 기대
2. **가시성 인지 스코어**: 반사점이 카메라 frustum 밖이면 벌점 대신 중립 처리
3. **멀티뷰 일관성**: 같은 씬 여러 뷰의 평면 추정을 월드 좌표에서 합의
4. **예측 mask 적용**: GT mask → submodel 예측 mask 교체(`--mask submodel` 구현됨)
   → 완전 end-to-end "스테레오 → 거울평면" 데모

## 부록 — 재현

```
# 단일 뷰
python scripts/estimate_mirror_plane_sym.py --scene ds_v1_seed0/computer_room_v00

# 전체 (8샤드 병렬 예시) + 집계
python scripts/run_sym_batch.py --shard 0/8   # ... 7/8 까지
python scripts/run_sym_batch.py --aggregate   # → experiments/_eval_sym.csv/.json

# 단위테스트 / figure 재생성
python -m pytest scripts/test_estimate_mirror_plane_sym.py -q
python scripts/build_sym_report_figs.py
```

뷰별 결과: `experiments/<group>/<scene>_<view>/mirror_geometry_sym/result.json`
(추정 normal/d/perp/score + eval_vs_gt). 종합 보고서는 [ds_seeds_report.md](ds_seeds_report.md) §6.
