# Mirror Geometry / FoundationStereo — 보고서 인덱스

> "Find Mirror Geometry inside FoundationStereo's Prior" (가반 7팀 · 김정현)
> 이 폴더(`report/`)의 모든 문서·그림을 관리하는 시작점. 작성 기준: 2026-06-05.

## 📂 문서 목록 (읽는 순서)

| # | 문서 | 형식 | 내용 | 추천 독자 |
|---|---|---|---|---|
| 1 | **`파이프라인_개요.md`** | md | 데이터 흐름 + 각 스크립트 역할 (코드 설명) | 처음 보는 사람 |
| 2 | **`index.html`** / `index.md` | html/md | **통합 기술 보고서** (검출·AHCF·평면·GT, 그림 포함) | 결과 요약이 필요한 사람 |
| 3 | **`연구기록.md`** | md | **연구 기록** (RQ1/RQ2, 시행착오·정량분석·인사이트) | 최종보고서·발표 작성 |
| 4 | **`worklog.html`** / `worklog.md` | html/md | **작업 로그** (성공·한계·실패 상태표, 의사결정) | 진행 이력 추적 |

> `.html`은 브라우저용(그림 렌더), `.md`는 동일 내용의 텍스트판. 그림은 `figures/`.

## 🖼 그림 (`figures/`)
| 파일 | 내용 |
|---|---|
| `fig_detection.png` | 거울 검출 결과 (cozy_living_room) |
| `fig_ahcf.png` | AHCF 전/후 신호 분석 패널 (computer_room) |
| `fig_ransac.png` | RANSAC 거울 평면 (반사면) |
| `fig_gt_computer.png` / `fig_gt_bedroom.png` / `fig_gt_livingroom.png` | GT 거울평면 재투영 오버레이 |
| `fig_gt_archiviz.png` / `fig_gt_cozy.png` / `fig_gt_minigym_view04.png` | (추가 장면) |

## 📊 핵심 결과 한눈에
- **거울 검출** (FS prior + 경량 헤드): 미지 장면 LOSO **AUC 0.911**, 교차-시점 0.947.
- **AHCF가 거울 신호 생성**: AUC 0.51(이전) → 최대 0.80(이후).
- **거울 평면**: RANSAC은 강건하나 "반사면"을 잡음(거울면 아님). GT 거울평면 16장면 구축.
- **평면 평가**(법선각/모서리거리): RANSAC 반사면 vs GT = 중앙값 77.8° / 4.9m (= 반사면≠거울면 정량).
- **K 가정 오류 발견**: 데이터셋 50mm 가정 ≠ 실제 25–60mm(절대 거리 척도에만 영향).

## 🗂 데이터·코드 위치
- 코드: `../scripts/`  (파이프라인·평가·GT 스크립트)
- 실험 출력: `../experiments/`
  - `legacy/` — 기존 17개 장면 결과 (이 보고서의 주 대상)
  - `archive_views/` — 신규 데이터셋(ds_v1) 결과
  - `_model/ _loso/ _summary_table.md` — 공유 인프라
- 설계·계획: `../docs/superpowers/`

## ⚠️ 알려진 한계 / 다음 과제
- **정공법(반사 대칭 거울평면 추정) 미구현** — 현재 평면 평가의 추정값은 RANSAC 반사면(baseline). 구현 후 GT로 점수.
- **절대 스케일** — 장면별 실제 lens로 K/points 재생성 필요(정공법 절대 거리 평가 선행).
- **일부 장면 거울 미가시** — terrazzo/minigym(벽 뒤·edge-on), modern_living_room(화면 밖).
- **검출 수치 갱신 대기** — bedroom 마스크 교정(31.7%) 후 LOSO/summary 재실행 필요(현재 수치는 교정 전).
- **Archive(ds_v1) 결과 미반영** — 통합 보고서는 legacy 중심. Archive 섹션 추가 예정.

## ▶ 빠른 실행
```bash
# 한 장면 전체 파이프라인 (입력 준비 후)
python ../scripts/run_pipeline_full.py --scene legacy/computer_room_baseline080
# 평면 평가 (법선각/모서리거리)
python ../scripts/eval_mirror_plane.py --root ../experiments/legacy
```
브라우저로 보고서 열기: `index.html`, `worklog.html`.
