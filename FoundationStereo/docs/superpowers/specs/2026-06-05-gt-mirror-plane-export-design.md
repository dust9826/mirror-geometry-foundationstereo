# GT 거울평면 추출 (GT Mirror Plane Export) — 설계 문서

- 날짜: 2026-06-05
- 프로젝트: "Find Mirror Geometry inside FoundationStereo's prior" (가반 7팀, 김정현)
- 접근: **A — 2단계 분리** (Blender 기하 추출 → numpy 좌표 변환/검증)

## 1. 목적과 배경

거울 평면 **정공법**(반사 대칭으로 거울면을 *찾는* 최적화)을 평가하려면, 비교 기준이 되는
**GT(정답) 거울 평면**이 필요하다. GT는 추정에 쓰지 않고 **평가용으로만** 사용한다
(평가 지표: 법선 각도 / 수직거리 / center 거리 / bbox IoU).

거울 본체 메쉬의 정점은 `.blend` 안에서만 접근 가능하므로, GT 평면은 Blender를 headless로
열어 메쉬에서 직접 추출해야 한다. 현재 `stereo_meta.json` 에는 카메라 extrinsic 과 거울
이름만 있고 평면 기하는 없다.

핵심 한계 재확인(왜 GT가 필요한가): 거울 안쪽 점들의 RANSAC 평면은 거울면이 아니라
"반사면"이다(거울면 = 진짜점↔반사점의 수직이등분면이라 반사점만으론 복원 불가).
따라서 거울면 자체의 GT는 점군이 아니라 Blender 메쉬에서 와야 한다.

## 2. 범위 (Scope)

- 먼저 **검증된 핵심 scene**(computer_room_baseline080, cozy_living_room_baseline080)으로
  파이프라인을 검증한 뒤 나머지로 확장한다.
- 스크립트는 `--scene <name>` 과 `--all` 을 모두 지원한다.
- GT 평면 추출은 마스크와 무관(메쉬에서 직접)하므로, 마스크가 빈 scene
  (minigym/living_room/terrazzo/modern_living_room)도 GT 추출 자체는 가능하다.
- **거울이 여러 개인 scene**(예: archiviz = 2개)은 **거울별 평면 리스트**로 저장한다.
  평가 시 추정 평면과 가장 가까운 GT 거울을 매칭한다.

### 비범위 (YAGNI)
- 기존 데이터셋 재렌더링 (접근 C 기각).
- views 데이터셋(minigym_view04 등 카메라 포즈가 다른 셋)은 이번 범위 밖.
  (baseline080 stereo 셋의 기본 카메라 CM 기준만 다룬다.)
- 곡면/비평면 거울의 정밀 모델링(평면 근사 + planarity 경고로 충분).

## 3. 구성 요소 & 데이터 흐름

```
사용자: python scripts/make_mirror_gt.py --scene computer_room_baseline080
         │  (FoundationStereo/scripts/make_mirror_gt.py = 오케스트레이터, numpy)
         │
         ├─1─► blender.exe --background --python export_mirror_gt.py -- --scene ... --out <tmp>
         │        │  (MakeStereoDataset/scripts/export_mirror_gt.py = Blender 전용)
         │        │  · scene → .blend 경로 해석 (computer_room_baseline080 → computer_room.blend)
         │        │  · find_mirrors() 재사용 (render_reflect3r_stereo 에서 import 또는 복제)
         │        │  · 거울별: 평가(depsgraph)메쉬 정점 matrix_world 변환 → 최소제곱 평면(world)
         │        │  · CM(left) matrix_world 기록
         │        └─► gt_mirror_world.json  (world 좌표, 거울 리스트)
         │
         ├─2─► (numpy) world → left-camera OpenCV 좌표 변환 (축 플립 포함)
         ├─3─► experiments/<scene>/mirror_gt/gt.json   (result.json 과 같은 스키마)
         └─4─► experiments/<scene>/mirror_gt/overlay.png  (검증: GT 폴리곤을 left.png + mask 위에)
```

### 책임 분리
- `export_mirror_gt.py` (Blender 전용, `MakeStereoDataset/scripts/`): 순수 기하 추출 → world json.
- `make_mirror_gt.py` (numpy, `FoundationStereo/scripts/`): Blender subprocess 호출 +
  좌표 변환 + 검증 산출물. 사용자는 **명령 하나**로 실행.
- `--no-blender` 플래그: 기존 `gt_mirror_world.json` 만 재변환(축 플립 등 빠른 디버깅, Blender 불필요).
- Blender 경로 기본값: `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`
  (없으면 명확한 에러로 경로 안내).

### scene → .blend 경로 해석
- scene 이름에서 `_baseline080` 접미사 제거 → base 이름.
- `blender_source_files/<base>.blend` 가 있으면 사용.
- 없으면 폴더형 `blender_source_files/<base>/` 안의 `*.blend` 를 탐색(여럿이면 경고/로그).
- 핵심 scene(computer_room.blend, cozy_living_room.blend)은 .blend 직접 존재.

## 4. 좌표 변환 (가장 오류 많은 핵심부)

거울 평면 = world 좌표의 (법선 `n_w`(단위), 중심점 `c_w`).
left 카메라 `C` = `left_camera_to_world` (cam→world, **Blender 규약: -Z 전방 / +Y 상 / +X 우**).
목표 프레임 = **left 카메라 OpenCV 규약: +Z 전방 / +Y 하 / +X 우** (`points.npy` 와 동일).

```
F = diag(1, -1, -1)         # Blender 카메라축 → OpenCV 카메라축 (y, z 부호 플립)
M = F · C⁻¹                 # world → OpenCV 카메라 (4x4)

c_cv  = (M · [c_w, 1])[:3]              # 중심점: 점 변환
R     = M[:3,:3]                        # rigid(직교)
n_cv  = R · n_w                         # 법선: 회전부만 (직교라 inverse-transpose 불필요)
n_cv /= |n_cv|
d_cv  = -(n_cv · c_cv)                  # 평면: n·x + d = 0
perp_distance = |d_cv|
corners_cv = M · corners_w              # 4 모서리도 점 변환 → bbox
```

- **rigid 성질:** `C⁻¹` 와 `F` 모두 직교 → `M` 직교. 따라서 법선은 회전만 적용하면 되고
  평면 변환의 inverse-transpose 미묘함을 피한다.
- **부호 규약:** `estimate_mirror_plane`/`mirror_geometry` 의 출력 부호 규약과 맞춘다.
  단 평면 *각도* 평가는 `angle = min(θ, 180°−θ)` 로 법선 방향 모호성을 제거한다.

## 5. 평면 피팅 (Blender 측, world 좌표)

- 거울 메쉬의 **평가(depsgraph) 메쉬** 정점을 `matrix_world` 로 world 변환
  (모디파이어 반영 위해 evaluated mesh 사용).
- 최소제곱 평면: 정점 중심 `c_w` = 평균, 공분산의 최소 고유벡터(SVD 최소 특이벡터) = `n_w`.
- 평면 bbox: 정점을 평면 기저(u, v)에 투영 → u/v 범위로 4 모서리 + width/height.
- `planarity_rms` = 정점들의 평면까지 수직거리 RMS (문짝/곡면 거울 경고용).

## 6. 출력 스키마 (`experiments/<scene>/mirror_gt/gt.json`)

`estimate_mirror_plane` 의 `result.json` 과 필드명을 정렬하여 평가가 1:1로 읽도록 한다.

```json
{
  "scene": "computer_room_baseline080",
  "frame": "left_camera_opencv",
  "mirrors": [
    {
      "name": "Mirror",
      "normal": [nx, ny, nz],
      "d": d,
      "perp_distance": perp,
      "centroid": [cx, cy, cz],
      "corners_3d": [[...], [...], [...], [...]],
      "plane_bbox": { "width_m": w, "height_m": h, "center_3d": [...] },
      "planarity_rms": rms,
      "n_verts": n
    }
  ],
  "extrinsic_check": { "max_abs_diff_vs_prepare_meta": e },
  "world": { "mirrors": [ { "name":..., "normal":..., "d":..., "centroid":..., "corners":... } ],
             "left_camera_to_world": [[...]] }
}
```

## 7. 검증 & 에러 처리 (= 성공 기준)

- **시각 검증(핵심):** GT 평면의 4모서리 + 중심을 K로 이미지에 재투영하여
  `left.png` 와 `mirror_mask.png` 위에 폴리곤을 오버레이(`overlay.png`).
  **폴리곤이 마스크와 겹치면 좌표 플립이 옳다는 증거.** 플립 실수 시 엉뚱한 위치에 찍혀 즉시 발견.
- **수치 교차검증:** .blend 의 CM extrinsic vs `prepare_meta.json` 의 `left_camera_to_world`
  비교 → `max_abs_diff_vs_prepare_meta` 가 작아야 함(올바른 scene/카메라 확인).
- 에러 처리:
  - blend 없음 / 거울 없음 → 건너뛰고 명확한 메시지.
  - `planarity_rms` 큰 거울 → 출력하되 경고 로그.
  - Blender 실행 파일 없음 → 알려진 경로 안내 후 종료.
  - Windows cp949 콘솔 유니코드 이슈 → 기존 `common.py` 패턴(stdout UTF-8 재설정) 따름.

## 8. 테스트

- **순수 변환부 TDD:** 합성 평면 + 합성 카메라로 world→cam 변환 후 재투영하여
  모서리가 기대 픽셀에 오는지 단위 테스트(Blender 불필요).
  대칭 케이스로 F 플립의 정확성(부호) 검증.
- **통합:** computer_room_baseline080 에서 `overlay.png` 가 마스크와 겹치는지(육안/IoU)
  + `extrinsic_check` 가 작은지 확인.

## 9. 산출물 정리

| 파일 | 위치 | 역할 |
|---|---|---|
| `export_mirror_gt.py` | `MakeStereoDataset/scripts/` | Blender 전용 기하 추출 → world json |
| `make_mirror_gt.py` | `FoundationStereo/scripts/` | 오케스트레이션 + 변환 + 검증 (단일 진입점) |
| `gt_mirror_world.json` | (임시/중간) | world 좌표 거울 평면 리스트 |
| `mirror_gt/gt.json` | `experiments/<scene>/` | left-cam OpenCV GT (평가 입력) |
| `mirror_gt/overlay.png` | `experiments/<scene>/` | 시각 검증 오버레이 |

## 10. 후속 (이 spec 범위 밖, 다음 단계)

이 GT가 준비되면 **정공법**(reflect_M 로 거울 안 P' ↔ 방 P 정합, RANSAC+ICP 또는 3대응
closed-form)을 구현하고, 본 GT로 법선각도/수직거리/center거리/bbox IoU 를 평가한다.
