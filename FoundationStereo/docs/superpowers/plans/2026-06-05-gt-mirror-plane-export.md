# GT 거울평면 추출 (GT Mirror Plane Export) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `.blend` 거울 본체 메쉬에서 GT 거울 평면을 추출해 left-camera OpenCV 좌표로 변환하고, `experiments/<scene>/mirror_gt/gt.json` + 검증 오버레이를 만든다 (정공법 평가용 정답).

**Architecture:** 접근 A(2단계 분리). Blender 전용 스크립트(`export_mirror_gt.py`)가 거울 메쉬 정점을 world 좌표 최소제곱 평면으로 피팅해 중간 json 을 쓰고, numpy 오케스트레이터(`make_mirror_gt.py`)가 그 평면을 left-camera OpenCV 좌표로 변환(축 플립)해 최종 `gt.json` 과 검증 `overlay.png` 를 만든다. 좌표 변환은 순수 함수로 분리해 Blender 없이 TDD 한다.

**Tech Stack:** Python 3.12, numpy, matplotlib, imageio (FoundationStereo `.venv`). Blender 5.1 (headless, 번들 numpy). 기존 `common.py`, `mirror_geometry.py`, `render_reflect3r_stereo.find_mirrors` 재사용.

> **NOTE — git:** 이 프로젝트 폴더(`FoundationStereo/`)는 git 저장소가 아니다(`repo/` 만 git). 각 Task 끝의 "Commit" 단계는 **저장소가 git 으로 초기화돼 있을 때만** 수행한다. 아니면 그 단계는 "변경 파일 확인 후 다음 Task 로" 체크포인트로 취급한다.

> **경로 표기:** 명령은 `FoundationStereo/` 를 작업 디렉터리로 가정한다(`cd D:\Users\dust\Documents\학교\2026년도1학기\컴퓨터비전\실습\FoundationStereo`). 테스트는 pytest 의존성을 피하기 위해 `if __name__ == "__main__"` 로 직접 실행한다.

---

## File Structure

| 파일 | 역할 | Task |
|---|---|---|
| `FoundationStereo/scripts/make_mirror_gt.py` | 오케스트레이터: 순수 변환 함수 + Blender 호출 + gt.json/overlay 생성 | 1,2,4,5 |
| `FoundationStereo/scripts/test_make_mirror_gt.py` | 순수 함수 단위 테스트 (Blender 불필요) | 1,2 |
| `MakeStereoDataset/scripts/export_mirror_gt.py` | Blender 전용: 거울 메쉬 → world 평면 json | 3 |
| `FoundationStereo/experiments/<scene>/mirror_gt/gt_mirror_world.json` | (중간 산출물) world 좌표 거울 평면 | 3,4 |
| `FoundationStereo/experiments/<scene>/mirror_gt/gt.json` | (최종) left-cam OpenCV GT | 4 |
| `FoundationStereo/experiments/<scene>/mirror_gt/overlay.png` | (검증) GT 폴리곤 오버레이 | 4 |

설계 결정: 좌표 변환·재투영을 `make_mirror_gt.py` 안의 **순수 함수**(`world_plane_to_cam`, `project_points`, `blend_base`)로 분리한다 → Blender 없이 단위 테스트 가능하고, 가장 오류가 잦은 축 플립을 빠르게 검증한다.

---

## Task 1: 좌표 변환 순수 함수 `world_plane_to_cam`

world 평면(Blender)을 left-camera OpenCV 좌표로 변환하는 핵심 함수. Blender 불필요, TDD.

**Files:**
- Create: `FoundationStereo/scripts/make_mirror_gt.py`
- Test: `FoundationStereo/scripts/test_make_mirror_gt.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`FoundationStereo/scripts/test_make_mirror_gt.py` 생성:

```python
"""make_mirror_gt.py 순수 함수 단위 테스트 (Blender 불필요).
실행: python scripts/test_make_mirror_gt.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from make_mirror_gt import world_plane_to_cam


def test_identity_camera_z_forward():
    # Blender 카메라 = world identity (–Z 전방). 거울이 5m 앞(-Z)에서 카메라를 향함.
    cam = np.eye(4)
    normal_w = np.array([0.0, 0.0, 1.0])        # +Z(world) = 카메라 쪽
    centroid_w = np.array([0.0, 0.0, -5.0])     # –Z 5m 전방
    corners_w = np.array([[-1, -1, -5], [1, -1, -5], [1, 1, -5], [-1, 1, -5]], float)
    out = world_plane_to_cam(normal_w, centroid_w, corners_w, cam)
    # OpenCV: +Z 전방 → centroid 가 +5
    assert np.allclose(out["centroid"], [0, 0, 5], atol=1e-6), out["centroid"]
    # 법선: F=diag(1,-1,-1) 적용 → (0,0,-1)
    assert np.allclose(out["normal"], [0, 0, -1], atol=1e-6), out["normal"]
    # 평면식 d = -n·c = -((0,0,-1)·(0,0,5)) = 5
    assert abs(out["d"] - 5.0) < 1e-6, out["d"]
    print("OK test_identity_camera_z_forward")


def test_y_flip_world_up_is_image_up():
    # world +Y(Blender 위)는 OpenCV 에서 -Y(위=작은 row) 가 돼야 한다 — y 플립 검증.
    cam = np.eye(4)
    out = world_plane_to_cam([0, 0, 1.0], [0.0, 1.0, -5.0],
                             [[0, 1, -5]] * 4, cam)
    assert out["centroid"][1] < 0, out["centroid"]
    print("OK test_y_flip_world_up_is_image_up")


def test_x_preserved():
    # world +X 는 OpenCV +X 로 보존(플립 없음).
    cam = np.eye(4)
    out = world_plane_to_cam([0, 0, 1.0], [1.0, 0.0, -5.0],
                             [[1, 0, -5]] * 4, cam)
    assert out["centroid"][0] > 0, out["centroid"]
    print("OK test_x_preserved")


if __name__ == "__main__":
    test_identity_camera_z_forward()
    test_y_flip_world_up_is_image_up()
    test_x_preserved()
    print("ALL PASS (Task 1)")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python scripts/test_make_mirror_gt.py`
Expected: FAIL — `ImportError: cannot import name 'world_plane_to_cam'` (또는 `ModuleNotFoundError: make_mirror_gt`)

- [ ] **Step 3: 최소 구현 작성**

`FoundationStereo/scripts/make_mirror_gt.py` 생성 (이 Task 에서는 헤더 + 순수 함수만):

```python
"""make_mirror_gt.py — GT 거울평면을 left-camera OpenCV 좌표로 추출/검증.

1) Blender(export_mirror_gt.py) 를 subprocess 로 호출해 world 평면 추출
2) world -> left-camera OpenCV 변환 (축 플립 F=diag(1,-1,-1))
3) experiments/<scene>/mirror_gt/gt.json + overlay.png 저장

사용:
  python scripts/make_mirror_gt.py --scene computer_room_baseline080
  python scripts/make_mirror_gt.py --all
  python scripts/make_mirror_gt.py --scene X --no-blender   # 기존 world json 재변환
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C


# --------------------------------------------------------------- 순수 변환 함수
def world_plane_to_cam(normal_w, centroid_w, corners_w, cam_to_world):
    """Blender world 평면을 left-camera OpenCV 좌표로 변환.

    normal_w     : (3,) world 단위 법선
    centroid_w   : (3,) world 평면 중심
    corners_w    : (4,3) world 평면 모서리
    cam_to_world : (4,4) Blender 카메라 matrix_world (cam->world, -Z 전방)
    반환 dict(normal(3,), d(float), centroid(3,), corners(4,3))  — OpenCV cam 좌표.
    """
    C_ = np.asarray(cam_to_world, dtype=np.float64).reshape(4, 4)
    F = np.diag([1.0, -1.0, -1.0, 1.0])      # Blender(-Z전방/+Y상) → OpenCV(+Z전방/+Y하)
    M = F @ np.linalg.inv(C_)                # world → OpenCV 카메라
    R = M[:3, :3]

    def xf_pt(p):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        return (M @ np.array([p[0], p[1], p[2], 1.0]))[:3]

    n = R @ np.asarray(normal_w, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    c = xf_pt(centroid_w)
    corners = np.array([xf_pt(p) for p in np.asarray(corners_w, float).reshape(-1, 3)])
    d = -float(n @ c)
    return {"normal": n, "d": d, "centroid": c, "corners": corners}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python scripts/test_make_mirror_gt.py`
Expected: PASS — `OK test_identity_camera_z_forward` / `OK test_y_flip_world_up_is_image_up` / `OK test_x_preserved` / `ALL PASS (Task 1)`

- [ ] **Step 5: Commit** (git 저장소일 때만)

```bash
git add scripts/make_mirror_gt.py scripts/test_make_mirror_gt.py
git commit -m "feat(gt): add world->camera plane transform with axis flip + tests"
```

---

## Task 2: 재투영 함수 `project_points` + `blend_base`

OpenCV 3D 점을 픽셀로 재투영(오버레이용)하는 함수와, scene 이름 → .blend base 이름 헬퍼.

**Files:**
- Modify: `FoundationStereo/scripts/make_mirror_gt.py`
- Test: `FoundationStereo/scripts/test_make_mirror_gt.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`test_make_mirror_gt.py` 의 import 줄에 함수 두 개를 추가하고, 테스트 함수와 `__main__` 호출을 추가한다.

import 줄 수정:

```python
from make_mirror_gt import world_plane_to_cam, project_points, blend_base
```

테스트 함수 추가 (`if __name__` 블록 위에):

```python
def test_projection_principal_point():
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 288], [0, 0, 1.0]])
    uv = project_points(K, [[0.0, 0.0, 5.0]])     # 광축 위의 점 → 주점
    assert np.allclose(uv[0], [512, 288], atol=1e-6), uv
    print("OK test_projection_principal_point")


def test_projection_offset():
    K = np.array([[1000.0, 0, 512], [0, 1000.0, 288], [0, 0, 1.0]])
    uv = project_points(K, [[1.0, 0.0, 5.0]])     # x=1,z=5 → u = 1000*1/5 + 512
    assert abs(uv[0, 0] - (200 + 512)) < 1e-6, uv
    assert abs(uv[0, 1] - 288) < 1e-6, uv
    print("OK test_projection_offset")


def test_blend_base_strips_suffix():
    assert blend_base("computer_room_baseline080") == "computer_room"
    assert blend_base("archiviz_baseline080") == "archiviz"
    assert blend_base("minigym_view04") == "minigym_view04"   # 접미사 없으면 그대로
    print("OK test_blend_base_strips_suffix")
```

`if __name__ == "__main__":` 블록에 호출 추가 (기존 3줄 뒤, `print("ALL PASS...")` 앞):

```python
    test_projection_principal_point()
    test_projection_offset()
    test_blend_base_strips_suffix()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python scripts/test_make_mirror_gt.py`
Expected: FAIL — `ImportError: cannot import name 'project_points'`

- [ ] **Step 3: 최소 구현 추가**

`make_mirror_gt.py` 의 `world_plane_to_cam` 아래에 추가:

```python
def project_points(K, pts):
    """(N,3) OpenCV 카메라좌표 점 → (N,2) 픽셀 (z>0 가정, z 클램프)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    z = np.clip(pts[:, 2], 1e-6, None)
    u = K[0, 0] * pts[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1)


def blend_base(scene):
    """experiments scene 이름 → .blend base 이름 (baseline 접미사 제거)."""
    suffix = "_baseline080"
    return scene[:-len(suffix)] if scene.endswith(suffix) else scene
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python scripts/test_make_mirror_gt.py`
Expected: PASS — 6개 `OK ...` 줄 + `ALL PASS (Task 1)` (마지막 print 문구는 그대로 둔다)

- [ ] **Step 5: Commit** (git 저장소일 때만)

```bash
git add scripts/make_mirror_gt.py scripts/test_make_mirror_gt.py
git commit -m "feat(gt): add project_points and blend_base helpers + tests"
```

---

## Task 3: Blender 전용 추출기 `export_mirror_gt.py`

`.blend` 를 열어 거울 메쉬 정점 → world 좌표 최소제곱 평면을 뽑아 중간 json 을 쓴다. Blender 안에서 실행. `render_reflect3r_stereo.find_mirrors` 재사용(DRY).

**Files:**
- Create: `MakeStereoDataset/scripts/export_mirror_gt.py`

- [ ] **Step 1: 스크립트 작성**

`MakeStereoDataset/scripts/export_mirror_gt.py` 생성:

```python
"""export_mirror_gt.py — (Blender 전용) .blend 거울 본체 메쉬의 GT 평면(world)을 추출.

거울 메쉬 정점을 matrix_world 로 world 좌표 변환 후 최소제곱 평면을 피팅한다.
좌표 변환(OpenCV)·시각화는 하지 않는다(그건 make_mirror_gt.py 담당).

실행:
  blender --background scene.blend --python export_mirror_gt.py -- \
      --out gt_mirror_world.json --camera CM
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

# find_mirrors 재사용 (같은 폴더의 렌더 스크립트). Blender 안이라 bpy import 안전.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_reflect3r_stereo import find_mirrors


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="CM")
    p.add_argument("--mirror_object", default="")
    return p.parse_args(argv)


def world_vertices(obj):
    """거울 메쉬의 world 좌표 정점 (N,3). 모디파이어 반영 evaluated mesh, 실패 시 raw."""
    mw = obj.matrix_world
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        oe = obj.evaluated_get(dg)
        me = oe.to_mesh()
        verts = np.array([(mw @ v.co)[:] for v in me.vertices], dtype=np.float64)
        oe.to_mesh_clear()
    except Exception:
        verts = np.array([(mw @ v.co)[:] for v in obj.data.vertices], dtype=np.float64)
    return verts


def fit_world_plane(obj):
    """거울 메쉬 → world 평면 dict(name, normal, centroid, d, corners, planarity_rms, n_verts)."""
    verts = world_vertices(obj)
    c = verts.mean(axis=0)
    # SVD: 최소 특이벡터 = 법선, 첫 두 벡터 = 평면 내 기저(u,v)
    _, _, vt = np.linalg.svd(verts - c)
    n = vt[-1] / (np.linalg.norm(vt[-1]) + 1e-12)
    u, v = vt[0], vt[1]
    resid = (verts - c) @ n
    rms = float(np.sqrt(np.mean(resid ** 2)))
    su, sv = (verts - c) @ u, (verts - c) @ v
    umin, umax, vmin, vmax = float(su.min()), float(su.max()), float(sv.min()), float(sv.max())
    corners = [(c + cu * u + cv * v).tolist()
               for cu, cv in [(umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax)]]
    return {
        "name": obj.name,
        "normal": n.tolist(),
        "centroid": c.tolist(),
        "d": float(-(n @ c)),
        "corners": corners,
        "planarity_rms": rms,
        "n_verts": int(len(verts)),
    }


def mat4(m):
    return [list(row) for row in m]


def main():
    a = parse_args()
    mirrors = find_mirrors(a.mirror_object)
    cam = bpy.data.objects.get(a.camera)
    out = {
        "camera": a.camera,
        "left_camera_to_world": mat4(cam.matrix_world) if cam else None,
        "mirror_names": [o.name for o in mirrors],
        "mirrors": [fit_world_plane(o) for o in mirrors],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("WROTE", a.out, "mirrors:", out["mirror_names"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: computer_room 으로 실행 검증**

먼저 출력 폴더를 만든다(없으면 Blender 가 못 씀):

Run:
```bash
python -c "import os; os.makedirs(r'D:\Users\dust\Documents\학교\2026년도1학기\컴퓨터비전\실습\FoundationStereo\experiments\computer_room_baseline080\mirror_gt', exist_ok=True)"
```

Blender headless 실행 (한 줄):
```bash
"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background "D:\Users\dust\Documents\학교\2026년도1학기\컴퓨터비전\실습\MakeStereoDataset\data\source_scenes\blender_source_files\computer_room.blend" --python "D:\Users\dust\Documents\학교\2026년도1학기\컴퓨터비전\실습\MakeStereoDataset\scripts\export_mirror_gt.py" -- --out "D:\Users\dust\Documents\학교\2026년도1학기\컴퓨터비전\실습\FoundationStereo\experiments\computer_room_baseline080\mirror_gt\gt_mirror_world.json" --camera CM
```
Expected: 콘솔 끝에 `WROTE ... mirrors: ['Mirror']`. `gt_mirror_world.json` 생성.

- [ ] **Step 3: 출력 JSON 정합성 확인**

Run:
```bash
python -c "import json;d=json.load(open(r'experiments\computer_room_baseline080\mirror_gt\gt_mirror_world.json',encoding='utf-8'));m=d['mirrors'][0];print('names',d['mirror_names']);print('n_verts',m['n_verts'],'rms',round(m['planarity_rms'],5));print('normal',[round(x,3) for x in m['normal']]);print('cam set?',d['left_camera_to_world'] is not None)"
```
Expected:
- `names ['Mirror']`
- `n_verts` > 0, `rms` 작음(평면 거울이면 < 0.01 정도)
- `normal` 단위벡터, `cam set? True`

- [ ] **Step 4: Commit** (git 저장소일 때만)

```bash
git -C "D:\Users\dust\Documents\학교\2026년도1학기\컴퓨터비전\실습\MakeStereoDataset" add scripts/export_mirror_gt.py
git -C "D:\Users\dust\Documents\학교\2026년도1학기\컴퓨터비전\실습\MakeStereoDataset" commit -m "feat: add Blender mirror GT world-plane extractor"
```

---

## Task 4: 오케스트레이터 완성 (변환 + gt.json + overlay) — computer_room 통합 검증

`make_mirror_gt.py` 에 Blender 호출/변환/검증 산출물 로직을 채우고 단일 scene 으로 end-to-end 검증.

**Files:**
- Modify: `FoundationStereo/scripts/make_mirror_gt.py`

- [ ] **Step 1: 변환/저장/오버레이/메인 로직 추가**

`make_mirror_gt.py` 의 `blend_base` 아래에 추가 (`mirror_geometry` import 는 함수 내부에서 한다 — 순수 함수 테스트가 numpy 외 의존 없이 돌도록):

```python
# --------------------------------------------------------------- 경로 상수
BLENDER = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
_MSD = os.path.normpath(os.path.join(C.PROJECT_ROOT, "..", "MakeStereoDataset"))
BLEND_ROOT = os.path.join(_MSD, "data", "source_scenes", "blender_source_files")
EXPORT_SCRIPT = os.path.join(_MSD, "scripts", "export_mirror_gt.py")


def resolve_blend(scene):
    """scene → .blend 경로 (없으면 None). 폴더형 blend 도 탐색."""
    base = blend_base(scene)
    cand = os.path.join(BLEND_ROOT, base + ".blend")
    if os.path.isfile(cand):
        return cand
    d = os.path.join(BLEND_ROOT, base)
    if os.path.isdir(d):
        blends = sorted(glob.glob(os.path.join(d, "*.blend")))
        if len(blends) > 1:
            print(f"[warn] {scene}: 폴더에 .blend 여러개 {[os.path.basename(b) for b in blends]} → 첫번째 사용")
        if blends:
            return blends[0]
    return None


def run_blender_export(blend, out_json):
    """Blender headless 로 export_mirror_gt.py 실행 → out_json 생성."""
    if not os.path.isfile(BLENDER):
        raise SystemExit(f"Blender 실행파일 없음: {BLENDER}")
    C.ensure_dir(os.path.dirname(out_json))
    cmd = [BLENDER, "--background", blend, "--python", EXPORT_SCRIPT, "--",
           "--out", out_json, "--camera", "CM"]
    print("RUN:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 or not os.path.isfile(out_json):
        raise SystemExit(f"Blender export 실패 (rc={rc}, out={out_json})")


def build_gt(scene, world_json, run_dir):
    """world json + prepare_meta → left-cam OpenCV gt dict (result.json 스키마 정렬)."""
    import mirror_geometry as MG

    meta = C.load_meta(run_dir)
    cam_to_world = np.asarray(world_json["left_camera_to_world"], dtype=np.float64)
    pm = np.asarray(meta["left_camera_to_world"], dtype=np.float64)
    ext_diff = float(np.max(np.abs(cam_to_world - pm)))

    mirrors_out = []
    for m in world_json.get("mirrors", []):
        t = world_plane_to_cam(m["normal"], m["centroid"], m["corners"], cam_to_world)
        n, d = MG.canonicalize_normal(t["normal"], t["corners"], t["d"])  # 카메라쪽으로 부호 정렬
        bbox = MG.plane_bbox(t["corners"], n)
        depth = MG.mirror_depth(n, d, t["corners"])
        mirrors_out.append({
            "name": m["name"],
            "normal": [float(x) for x in n],
            "d": float(d),
            "perp_distance": depth["perp_distance"],
            "centroid": [float(x) for x in t["centroid"]],
            "corners_3d": t["corners"].tolist(),
            "plane_bbox": bbox,
            "planarity_rms": m.get("planarity_rms"),
            "n_verts": m.get("n_verts"),
        })
    return {
        "scene": scene,
        "frame": "left_camera_opencv",
        "mirrors": mirrors_out,
        "extrinsic_check": {"max_abs_diff_vs_prepare_meta": ext_diff},
        "world": world_json,
    }


def save_overlay(path, run_dir, gt):
    """GT 평면 4모서리+중심을 K로 재투영해 left.png / mirror_mask.png 위에 오버레이."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    K, _ = C.read_K_txt(os.path.join(run_dir, "input", "K.txt"))
    left = imageio.imread(os.path.join(run_dir, "input", "left.png"))[..., :3]
    mask = imageio.imread(os.path.join(run_dir, "input", "mirror_mask.png"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    panels = [(ax[0], left, None, "left + GT"),
              (ax[1], mask, "gray", "mask + GT")]
    for a_, img, cmap, title in panels:
        a_.imshow(img, cmap=cmap)
        a_.set_title(title)
        a_.axis("off")
        for m in gt["mirrors"]:
            uv = project_points(K, m["corners_3d"])
            poly = np.vstack([uv, uv[0]])
            a_.plot(poly[:, 0], poly[:, 1], "-", lw=2, color="lime")
            cuv = project_points(K, [m["centroid"]])[0]
            a_.plot(cuv[0], cuv[1], "x", color="red", ms=10, mew=3)
    fig.suptitle(f"GT mirror plane — {gt['scene']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def process_scene(scene, args):
    """단일 scene: (옵션)Blender 추출 → 변환 → gt.json + overlay.png."""
    run_dir = C.run_dir_for(scene)
    if not os.path.isdir(run_dir):
        print(f"[skip] {scene}: experiments 폴더 없음")
        return False
    out_dir = C.ensure_dir(os.path.join(run_dir, "mirror_gt"))
    world_json_path = os.path.join(out_dir, "gt_mirror_world.json")

    if not args.no_blender:
        blend = resolve_blend(scene)
        if not blend:
            print(f"[skip] {scene}: .blend 못찾음 (base={blend_base(scene)})")
            return False
        run_blender_export(blend, world_json_path)

    if not os.path.isfile(world_json_path):
        print(f"[skip] {scene}: world json 없음 ({world_json_path})")
        return False
    with open(world_json_path, encoding="utf-8") as f:
        world_json = json.load(f)
    if not world_json.get("mirrors"):
        print(f"[warn] {scene}: 거울 0개 (gt.json 은 빈 mirrors 로 저장)")

    gt = build_gt(scene, world_json, run_dir)
    with open(os.path.join(out_dir, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    save_overlay(os.path.join(out_dir, "overlay.png"), run_dir, gt)

    for m in gt["mirrors"]:
        rms = m.get("planarity_rms") or 0.0
        if rms > 0.02:
            print(f"[warn] {scene}/{m['name']}: planarity_rms={rms:.3f}m (비평면 거울 의심)")
    ext = gt["extrinsic_check"]["max_abs_diff_vs_prepare_meta"]
    print(f"[ok] {scene}: mirrors={len(gt['mirrors'])} ext_diff={ext:.2e} → {out_dir}")
    if ext > 1e-3:
        print(f"[warn] {scene}: extrinsic 불일치 {ext:.4f} (scene/카메라 확인 필요)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-blender", action="store_true",
                    help="기존 gt_mirror_world.json 만 재변환(Blender 미실행)")
    args = ap.parse_args()

    if args.all:
        scenes = sorted(d for d in os.listdir(C.EXPERIMENTS_DIR)
                        if os.path.isdir(os.path.join(C.EXPERIMENTS_DIR, d))
                        and not d.startswith("_"))
    elif args.scene:
        scenes = [args.scene]
    else:
        raise SystemExit("--scene <name> 또는 --all 필요")

    results = [(s, process_scene(s, args)) for s in scenes]
    print("\n#### SUMMARY")
    for s, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {s}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 순수 함수 테스트 회귀 확인 (깨지지 않았는지)**

Run: `python scripts/test_make_mirror_gt.py`
Expected: PASS — 6개 `OK` + `ALL PASS (Task 1)` (import 가 늘어난 상수/함수 때문에 깨지지 않아야 함)

- [ ] **Step 3: computer_room end-to-end 실행 (Blender 재사용, --no-blender)**

Task 3 에서 이미 `gt_mirror_world.json` 을 만들었으므로 Blender 없이 변환만:

Run: `python scripts/make_mirror_gt.py --scene computer_room_baseline080 --no-blender`
Expected: `[ok] computer_room_baseline080: mirrors=1 ext_diff=...e-... → ...\mirror_gt`
- `ext_diff` 가 작아야 함(< 1e-3). 크면 Task 3 의 카메라/ scene 이 데이터셋과 다른 것.

- [ ] **Step 4: gt.json 스키마/값 확인**

Run:
```bash
python -c "import json;d=json.load(open(r'experiments\computer_room_baseline080\mirror_gt\gt.json',encoding='utf-8'));m=d['mirrors'][0];print('frame',d['frame']);print('ext_diff',d['extrinsic_check']);print('perp',round(m['perp_distance'],3),'centroid_z',round(m['centroid'][2],3));print('normal',[round(x,3) for x in m['normal']]);print('bbox',round(m['plane_bbox']['width_m'],2),round(m['plane_bbox']['height_m'],2))"
```
Expected:
- `frame left_camera_opencv`
- `centroid_z` 양수(+Z 전방), `perp` 양수
- **거울면 GT 의 `perp`/`centroid_z` 는 기존 RANSAC "반사면"(perp≈9.5, z≈10.65)보다 작아야 한다** (거울 테두리 깊이 ≈ 6m 안팎 예상). 이 차이가 "반사면 ≠ 거울면" 한계를 정량 확인해 준다.

- [ ] **Step 5: overlay.png 육안 검증 (성공 기준)**

`experiments/computer_room_baseline080/mirror_gt/overlay.png` 를 연다.
Expected: **초록 폴리곤(GT 거울 평면 투영)이 오른쪽 패널의 흰 거울 마스크와 겹쳐야 한다.** 빨간 ×(중심)도 거울 안. 엉뚱한 곳(상하 반전 등)이면 Task 1 의 F 플립/부호 재점검.

- [ ] **Step 6: Commit** (git 저장소일 때만)

```bash
git add scripts/make_mirror_gt.py
git commit -m "feat(gt): orchestrate Blender export, world->cam transform, gt.json + overlay"
```

---

## Task 5: 핵심 scene 확장 + 다중 거울 검증

cozy_living_room(검증된 scene)과 archiviz(거울 2개)로 일반화 확인. Blender 풀 경로 실행 포함.

**Files:** (코드 변경 없음 — 실행/검증만. 문제 발견 시 해당 Task 로 회귀)

- [ ] **Step 1: cozy_living_room 풀 실행 (Blender 포함)**

Run: `python scripts/make_mirror_gt.py --scene cozy_living_room_baseline080`
Expected: `RUN: ...blender.exe...` 로그 후 `[ok] cozy_living_room_baseline080: mirrors>=1 ext_diff=...`
`overlay.png` 의 초록 폴리곤이 마스크와 겹침.

- [ ] **Step 2: archiviz 다중 거울 실행**

Run: `python scripts/make_mirror_gt.py --scene archiviz_baseline080`
Expected: `[ok] archiviz_baseline080: mirrors=2 ...` — `gt.json` 의 `mirrors` 가 **2개**(거울별 리스트), overlay 에 폴리곤 2개.

Run (확인):
```bash
python -c "import json;d=json.load(open(r'experiments\archiviz_baseline080\mirror_gt\gt.json',encoding='utf-8'));print('n_mirrors',len(d['mirrors']));[print(m['name'],'perp',round(m['perp_distance'],2),'rms',round(m['planarity_rms'],4)) for m in d['mirrors']]"
```
Expected: `n_mirrors 2`, 두 거울 이름(`main door middle.001`, `main door right.001`) 각각 perp/rms 출력.

- [ ] **Step 3: 자체 검증 결과 정리**

세 scene(computer_room, cozy_living_room, archiviz)의 `overlay.png` 와 `ext_diff`/`perp`/`planarity_rms` 를 한눈에 표로 정리해 사용자에게 보고:
- overlay 폴리곤-마스크 정합 여부(육안)
- `ext_diff` 모두 < 1e-3 인지(올바른 카메라)
- GT `perp` vs 기존 RANSAC 반사면 `perp` 비교(거울면 ≠ 반사면 정량 확인)

- [ ] **Step 4: Commit** (git 저장소일 때만 — 산출물 커밋이 필요하면)

```bash
git add scripts/make_mirror_gt.py
git commit -m "test(gt): verify GT export on computer_room, cozy_living_room, archiviz (multi-mirror)"
```

---

## Self-Review (작성자 체크리스트)

**1. Spec coverage:**
- §3 구성요소(2파일 분리) → Task 3(export_mirror_gt) + Task 1/2/4(make_mirror_gt). ✅
- §3 scene→blend 해석 → Task 2 `blend_base` + Task 4 `resolve_blend`(폴더형 처리). ✅
- §4 좌표 변환(F 플립, rigid 법선) → Task 1 `world_plane_to_cam`. ✅
- §5 평면 피팅(evaluated mesh, SVD, planarity_rms, bbox) → Task 3 `fit_world_plane`/`world_vertices`. ✅
- §6 출력 스키마(result.json 정렬) → Task 4 `build_gt`. ✅
- §7 시각 검증(overlay) → Task 4 `save_overlay` + Task 4 Step5 육안. ✅
- §7 수치 교차검증(extrinsic_check) → Task 4 `build_gt` ext_diff + 경고. ✅
- §7 에러처리(blend/거울 없음, planarity 경고, Blender 없음) → Task 4 `process_scene`/`run_blender_export`. ✅
- §7 cp949 유니코드 → `make_mirror_gt` 가 `common` import(stdout UTF-8 재설정). export_mirror_gt 는 Blender Python(UTF-8 기본)이라 무관. ✅
- §2 범위(--scene/--all, 핵심 scene 먼저, 거울별 리스트) → Task 4 main + Task 5. ✅
- §8 테스트(순수 변환 TDD + 통합) → Task 1/2 단위, Task 4/5 통합. ✅

**2. Placeholder scan:** "TBD/적절히 처리" 류 없음. 모든 코드 단계에 실제 코드 포함. ✅

**3. Type consistency:**
- `world_plane_to_cam` 반환 dict 키(`normal/d/centroid/corners`) → `build_gt`/테스트에서 동일하게 사용. ✅
- `project_points(K, pts)` 시그니처 → `save_overlay`/테스트 동일. ✅
- `blend_base(scene)` → `resolve_blend` 에서 사용. ✅
- 재사용 함수 시그니처 확인: `MG.canonicalize_normal(normal, points, d)`→(normal,d) / `MG.plane_bbox(points, normal)`→dict / `MG.mirror_depth(normal, d, points)`→dict, `C.read_K_txt`→(K,baseline), `C.load_meta`→prepare_meta dict, `C.run_dir_for/ensure_dir/EXPERIMENTS_DIR/PROJECT_ROOT` — 모두 기존 코드에서 검증됨. ✅
- `find_mirrors(requested="")` (render_reflect3r_stereo) 시그니처 일치. ✅
