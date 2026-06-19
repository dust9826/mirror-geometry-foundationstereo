# Colab 실행 패키지 — Mirror Geometry / FoundationStereo

스테레오 입력 → FS prior → 거울 검출 · **반사대칭 거울평면(정공법)** · LOSO 까지 Colab에서 돌리는 패키지.
**`files.upload()` 위젯 없이, zip을 파일 패널에 드래그**해서 쓴다(빠름).

> 최종 갱신: 2026-06-07 — IPIU 2026 논문(*스테레오 Prior에 내재된 거울 기하의 복원*) 최종 결과 반영.
> 신규 포함: `estimate_mirror_plane_sym.py`(반사대칭 평면), 전체 스크립트 최신화(30종), 논문 수치 요약 셀.

## 📦 파일
| 파일 | 크기 | 내용 |
|---|---|---|
| `mirror_pipeline_colab_drag.ipynb` | — | 실행 노트북 (Colab에서 열기) |
| `colab_all.zip` | 239MB | scripts(최신 30종) + 샘플 input + **ViT-S 모델**(250MB) |
| `colab_all_vitl.zip` | 1.4GB | 위와 동일하나 **ViT-L 모델**(최고 성능) |
| `colab_loso.zip` | 97MB | **LOSO용 12장면 features+mask** (FS 재추론 불필요) |

> `colab_all*.zip` 은 **하나만** 선택해서 올린다(ViT-S=가볍고 빠름 / ViT-L=품질).
> `repo`(NVlabs FoundationStereo)와 `open3d` 등 패키지는 노트북이 알아서 clone·설치한다.

## 🚀 사용법
1. `mirror_pipeline_colab_drag.ipynb` 를 Colab에서 연다.
2. **런타임 → 런타임 유형 변경 → GPU(T4)**.
3. 좌측 **파일 패널**(기본 위치 `/content`)에 드래그:
   - 필수: `colab_all.zip` (또는 `colab_all_vitl.zip`)
   - LOSO(§7~9)까지 보려면: `colab_loso.zip` 도 함께
4. 셀을 위에서부터 순서대로 실행.

## 📑 노트북 섹션
| § | 내용 |
|---|---|
| 1 | GPU 확인 + 경로 |
| 2 | `/content/*.zip` 전부 압축해제 + repo clone + 의존성(**open3d 포함**) |
| 3 | 압축해제 결과 확인 (scripts/모델/input 존재) |
| 4 | FS 추론 (실패 시 STDERR 자동 출력) |
| 5 | (선택) 평면·AHCF·검출 + GT depth 평가 |
| **5.5** | **반사대칭 거울평면 복원 (정공법, 학습 없음)** — naive 반사면 vs 진짜 거울면 + GT 평가 |
| 6 | 결과 종합 출력 (수치 + 그림, 정공법 포함) |
| 7 | 여러 scene **LOSO** 실행 (12-fold) |
| 8 | LOSO 결과 (표 + 막대그래프) |
| 9 | LOSO fold별 예측 패널(panel.png) |
| **10** | **논문 최종 수치 요약** (1,537뷰: 거울 41× / 검출 4체제 / 평면 2.3°·0.14m·72%) |

## ⚠️ 자주 막히는 곳
- **`BadZipFile`**: zip 업로드가 아직 안 끝남 → 파일 패널 진행표시 사라질 때까지 대기 후 재실행.
- **`No module named 'open3d'` / FS 추론 실패**: ② 셀이 open3d 설치 → 다시 실행. numpy 에러 시 런타임 재시작 후 ②부터.
- **체크포인트 `[]`**: `colab_all*.zip` 이 제대로 풀렸는지(③ 셀) 확인.
- **LOSO에서 scene<3**: `colab_loso.zip` 을 올리고 ②(압축해제) 재실행.

## 재생성 (로컬)
- 노트북: `python build_colab_drag.py` → `colab/mirror_pipeline_colab_drag.ipynb`
- zip 갱신: `python build_colab_zips.py` (모델·demo 입력 보존, scripts 만 최신 교체)
- 로컬 데모 노트북: `python build_notebook.py` → `mirror_pipeline_demo.ipynb`
- 모델은 추론 전용 슬림화(ViT-S 752→250MB, ViT-L 3.1G→1.4G).
