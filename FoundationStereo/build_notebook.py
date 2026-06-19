# -*- coding: utf-8 -*-
"""mirror_pipeline_demo.ipynb 생성기. 한 번 실행 후 삭제해도 됨."""
import json, os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}

def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(
"# Finding Mirror Geometry inside FoundationStereo's Prior — 재현 노트북",
"",
"스테레오 입력과 FoundationStereo(FS) prior 만으로 **거울 검출 / AHCF 신호 분석 / 거울 평면(RANSAC)**을",
"한 장면에 대해 재현한다. (선택) Blender 가 있으면 **거울 평면 GT**까지.",
"",
"## 입력이 뭐가 필요한가?",
"| 단계 | 필요 입력 |",
"|---|---|",
"| 1~5 (검출·AHCF·평면) | **left.png + right.png + mirror_mask.png** (+ baseline·렌즈 가정) |",
"| 6 (GT 평면) | **Blender + .blend 원본** — 이미지만으론 불가(거울 메쉬가 .blend 에만 있음) |",
"",
"`mirror_mask.png` 는 평가(AUC/IoU)와 서브모델 학습 라벨에 쓰인다. 없으면 1~3 단계의 *예측*은 보지만 정량 평가는 건너뛴다.",
"",
"## 전제",
"- 이 노트북을 **프로젝트 루트(`FoundationStereo/`)** 에서 연다.",
"- 커널 = 프로젝트 `.venv` (torch 2.7.1+cu128 등). 다른 커널이면 FS 추론이 안 된다.",
"- `pretrained_models/` 에 체크포인트 존재. FS 추론은 **GPU** 필요(수십 초~분).",
))

cells.append(md("## 0. 설정 — 입력 폴더와 파라미터", "",
"`INPUT_DIR` 에 `left.png`, `right.png`, `mirror_mask.png`(또는 `mirror_mask_left.png`) 를 둔다.",
"기본값은 이 환경의 샘플 장면(computer_room)을 가리킨다. 직접 데이터를 쓰려면 경로만 바꾸면 된다."))

cells.append(code(
"import os, sys, subprocess, json",
"import numpy as np",
"import matplotlib.pyplot as plt",
"import imageio.v2 as imageio",
"",
"# --- 프로젝트 루트 자동 탐색 (scripts/common.py 가 보이는 곳) ---",
"PROJECT_ROOT = os.path.abspath('')",
"if not os.path.isfile(os.path.join(PROJECT_ROOT, 'scripts', 'common.py')):",
"    raise SystemExit('이 노트북을 FoundationStereo/ 루트에서 실행하세요 (scripts/common.py 가 안 보임)')",
"sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))",
"import common as C",
"PY = sys.executable  # 현재 커널의 .venv 파이썬으로 스크립트 실행",
"print('PROJECT_ROOT =', PROJECT_ROOT)",
"print('python       =', PY)",
"",
"# --- 사용자 설정 ---",
"SCENE     = 'demo_computer_room'   # 출력 experiments/<SCENE>/ 이름 (자유)",
"INPUT_DIR = os.path.normpath(os.path.join(",
"    PROJECT_ROOT, '..', 'MakeStereoDataset', 'data', 'stereo_dataset', 'computer_room_baseline080'))",
"BASELINE_M = 0.08    # 스테레오 베이스라인(m)",
"LENS_MM    = 50.0    # 가정 렌즈 초점거리(mm) — 절대 깊이 스케일에만 영향(§ 참고)",
"SENSOR_MM  = 36.0    # 센서 가로(mm)",
"print('INPUT_DIR =', INPUT_DIR)",
))

cells.append(md("## 1. 입력 확인 및 미리보기"))

cells.append(code(
"def find_one(d, names):",
"    for n in names:",
"        p = os.path.join(d, n)",
"        if os.path.isfile(p): return p",
"    return None",
"",
"left_src  = find_one(INPUT_DIR, ['left.png'])",
"right_src = find_one(INPUT_DIR, ['right.png'])",
"mask_src  = find_one(INPUT_DIR, ['mirror_mask.png', 'mirror_mask_left.png'])",
"assert left_src and right_src, f'left.png/right.png 가 {INPUT_DIR} 에 필요합니다'",
"print('left :', left_src)",
"print('right:', right_src)",
"print('mask :', mask_src if mask_src else '(없음 — 평가/서브모델 라벨 생략)')",
"",
"imgs = [('left', left_src), ('right', right_src)] + ([('mirror_mask', mask_src)] if mask_src else [])",
"fig, ax = plt.subplots(1, len(imgs), figsize=(5*len(imgs), 4))",
"ax = np.atleast_1d(ax)",
"for a, (t, p) in zip(ax, imgs):",
"    a.imshow(imageio.imread(p)); a.set_title(t); a.axis('off')",
"plt.tight_layout(); plt.show()",
))

cells.append(md("## 2. 입력 준비 — `experiments/<SCENE>/input/` 생성", "",
"`left/right/mirror_mask` 만으로 입력을 구성한다(원본 `prepare_data.py` 와 달리 `stereo_meta.json` 불필요).",
"K 는 렌즈/센서 가정으로 계산하고, baseline 은 위 설정값을 쓴다. extrinsic 은 (있으면) stereo_meta 에서 가져오되 없으면 생략(6단계 GT 에서만 필요)."))

cells.append(code(
"import shutil",
"run_dir = os.path.join(PROJECT_ROOT, 'experiments', SCENE)",
"in_dir  = C.ensure_dir(os.path.join(run_dir, 'input'))",
"h, w = imageio.imread(left_src).shape[:2]",
"K = C.intrinsics_from_assumption(w, h, lens_mm=LENS_MM, sensor_mm=SENSOR_MM)",
"shutil.copyfile(left_src,  os.path.join(in_dir, 'left.png'))",
"shutil.copyfile(right_src, os.path.join(in_dir, 'right.png'))",
"if mask_src: shutil.copyfile(mask_src, os.path.join(in_dir, 'mirror_mask.png'))",
"C.write_K_txt(os.path.join(in_dir, 'K.txt'), K, BASELINE_M)",
"",
"# stereo_meta 가 있으면 extrinsic 등 보존(없어도 OK)",
"meta_src = os.path.join(INPUT_DIR, 'stereo_meta.json')",
"sm = json.load(open(meta_src)) if os.path.isfile(meta_src) else {}",
"prep = dict(scene=SCENE, source_dir=INPUT_DIR, width=w, height=h, baseline_m=BASELINE_M,",
"            K=K.tolist(), K_source='lens/sensor', lens_mm=LENS_MM, sensor_mm=SENSOR_MM,",
"            K_is_assumed=True,",
"            mirror_objects=sm.get('mirror_objects', sm.get('mirror_object')),",
"            left_camera_to_world=sm.get('left_camera_to_world'),",
"            right_camera_to_world=sm.get('right_camera_to_world'))",
"json.dump(prep, open(os.path.join(in_dir, 'prepare_meta.json'), 'w'), indent=2)",
"print('prepared:', run_dir, '| size', (w, h), '| baseline', BASELINE_M)",
"print('K=\\n', K)",
))

cells.append(md("## 3. FoundationStereo 추론 + prior 추출 (무거움, GPU)", "",
"`run_foundation_stereo.py` 가 FS 를 1회 forward 하며 hook 으로 AHCF 전/후 cost-volume 통계와 side-tuning 특징을 뽑아 `fs/features.npz` 등에 저장한다.",
"GPU/체크포인트가 없으면 여기서 에러가 난다(그 경우 메시지 확인)."))

cells.append(code(
"cmd = [PY, os.path.join('scripts', 'run_foundation_stereo.py'), '--scene', SCENE]",
"print('RUN:', ' '.join(cmd)); print('-'*60)",
"rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode",
"print('-'*60, '\\nreturn code:', rc)",
"assert rc == 0, 'FS 추론 실패 — GPU/torch/pretrained_models 확인'",
))

cells.append(code(
"# disparity 시각화 + depth 통계",
"vis = os.path.join(run_dir, 'fs', 'vis_disp.png')",
"depth = np.load(os.path.join(run_dir, 'fs', 'depth_meter.npy'))",
"fig, ax = plt.subplots(1, 2, figsize=(15, 4))",
"ax[0].imshow(imageio.imread(vis)); ax[0].set_title('left | disparity'); ax[0].axis('off')",
"valid = depth > 0",
"im = ax[1].imshow(np.where(valid, depth, np.nan), cmap='turbo'); ax[1].set_title('depth (m)'); ax[1].axis('off')",
"fig.colorbar(im, ax=ax[1], fraction=0.046)",
"plt.tight_layout(); plt.show()",
"print('depth(m): valid', int(valid.sum()), 'min %.2f max %.2f'%(depth[valid].min(), depth[valid].max()))",
))

cells.append(md("## 4. AHCF 신호 분석 (RQ: AHCF 가 거울 신호를 만드는가?)", "",
"`analyze_features.py` 가 `features.npz` 의 AHCF 전/후 신호를 거울 마스크와 대조해 신호별 ROC AUC 를 매긴다.",
"`pre_*`(우연≈0.51) → `post_*`/`delta_*`(상승) 이면 AHCF 가 거울 신호를 강화한 것."))

cells.append(code(
"rc = subprocess.run([PY, os.path.join('scripts','analyze_features.py'), '--scene', SCENE],",
"                    cwd=PROJECT_ROOT).returncode",
"assert rc == 0",
"rep = json.load(open(os.path.join(run_dir, 'features', 'report.json'), encoding='utf-8'))",
"print('best signal:', rep['best_signal'], 'AUC', rep['best_auc'])",
"print('rank  signal              AUC   pol')",
"for i, r in enumerate(rep['ranked'][:8], 1):",
"    print(f\"{i:>3}  {r['signal']:<18} {str(r['auc']):>5}  {r['polarity']}\")",
"plt.figure(figsize=(15, 9))",
"plt.imshow(imageio.imread(os.path.join(run_dir, 'features', 'panels.png'))); plt.axis('off'); plt.show()",
))

cells.append(md("## 5. 거울 평면 적합 (RANSAC) — 그리고 그 한계", "",
"`estimate_mirror_plane.py` 가 거울 점군에 PCA/RANSAC/IRLS 평면을 적합한다. RANSAC 은 강건하지만,",
"적합되는 평면은 거울면이 아니라 **반사면**이다(반사점만으론 거울면 복원 불가). → 6단계 GT 로 정량 확인."))

cells.append(code(
"if not mask_src:",
"    print('mirror_mask 없음 → 평면 적합 생략'); res = None",
"else:",
"    rc = subprocess.run([PY, os.path.join('scripts','estimate_mirror_plane.py'), '--scene', SCENE],",
"                        cwd=PROJECT_ROOT).returncode",
"    assert rc == 0",
"    res = json.load(open(os.path.join(run_dir,'mirror_geometry','result.json'), encoding='utf-8'))",
"    rr = res['recommended_result']",
"    print('recommended:', res['recommended'])",
"    print('  normal', [round(x,3) for x in rr['normal']], 'perp_distance %.2f m'%res['mirror_depth']['perp_distance'])",
"    print('  rms: PCA %.3f vs RANSAC %.3f m'%(res['methods']['pca']['rms_residual'], res['methods']['ransac']['rms_residual']))",
"    viz = os.path.join(run_dir, 'mirror_geometry', 'viz.png')",
"    if os.path.isfile(viz):",
"        plt.figure(figsize=(13,5)); plt.imshow(imageio.imread(viz)); plt.axis('off'); plt.show()",
))

cells.append(md("## 6. 거울 검출 — FS prior 위 경량 헤드 (접근 #3)", "",
"`submodel_head.py` 가 픽셀당 136차원 FS 특징 위에 작은 MLP 를 학습(이 셀은 단일 장면 70/30 분할 데모).",
"여러 장면 일반화(LOSO)는 `apply_submodel.py` 참고. 마스크가 있어야 학습/평가 가능."))

cells.append(code(
"if not mask_src:",
"    print('mirror_mask 없음 → 서브모델 학습 생략')",
"else:",
"    rc = subprocess.run([PY, os.path.join('scripts','submodel_head.py'), '--scene', SCENE],",
"                        cwd=PROJECT_ROOT).returncode",
"    assert rc == 0",
"    mt = json.load(open(os.path.join(run_dir,'submodel','metrics.json'), encoding='utf-8'))['metrics']",
"    print('AUC %.3f  IoU %.3f  F1 %.3f'%(mt['auc'], mt['iou'], mt['f1']))",
"    plt.figure(figsize=(15,5)); plt.imshow(imageio.imread(os.path.join(run_dir,'submodel','panel.png'))); plt.axis('off'); plt.show()",
))

cells.append(md("## 6.5 반사대칭 거울평면 복원 (정공법, 학습 없음)", "",
"5단계의 RANSAC 평면은 거울면이 아니라 **반사면**(약 2배 깊음)이다. `estimate_mirror_plane_sym.py` 는",
"반사 대칭 — 거울 안 가상점 P′와 방의 실제점 P 가 거울평면에 대칭 — 으로 진짜 거울면을 복원한다.",
"대응쌍 하나의 수직이등분면이 평면을 닫힌형으로 결정 → RANSAC + trimmed-NN 점수 + ICP 정제.",
"(IPIU 2026 논문: 1,536뷰에서 법선각 중앙값 2.3°, 수직거리 오차 중앙값 0.14 m, 성공률 72%)"))

cells.append(code(
"if not mask_src:",
"    print('mirror_mask 없음 → 정공법 생략')",
"else:",
"    rc = subprocess.run([PY, os.path.join('scripts','estimate_mirror_plane_sym.py'), '--scene', SCENE, '--quiet'],",
"                        cwd=PROJECT_ROOT).returncode",
"    sp = os.path.join(run_dir,'mirror_geometry_sym','result.json')",
"    if rc == 0 and os.path.isfile(sp):",
"        sy = json.load(open(sp, encoding='utf-8'))",
"        print('반사대칭 평면: perp %.2f m  normal %s'%(sy['perp'], [round(x,3) for x in sy['normal']]))",
"        if res: print('  (비교) 단순 RANSAC 반사면: perp %.2f m'%res['mirror_depth']['perp_distance'])",
"        ev = sy.get('eval_vs_gt')",
"        if ev: print('  GT 대비: 각도 %.1f°, perp 오차 %.2f m (GT %.2f m)'%(ev['angle_deg'], ev['perp_err_m'], ev['gt_perp']))",
"    else:",
"        print('정공법 실패/스킵 (rc=%s)'%rc)",
))

cells.append(md("## 7. (선택) 거울 평면 GT — Blender 필요", "",
"이미지만으로는 불가하다. `.blend` 원본 + Blender 설치가 있어야 거울 메쉬에서 평면 GT 를 뽑는다.",
"`make_mirror_gt.py` 가 Blender 를 headless 로 호출 → world 평면 → 좌측 카메라 OpenCV 좌표 변환 → `mirror_gt/gt.json` + `overlay.png`.",
"",
"전제: `make_mirror_gt.py` 의 `BLENDER` 경로가 맞아야 하고, `<scene>` 이름이 `_baseline080` 규약이어야 `.blend` 를 찾는다.",
"여기 데모 SCENE 은 임의 이름이라 자동 매칭이 안 될 수 있으니, 실제 baseline 장면명으로 시도한다."))

cells.append(code(
"GT_SCENE = 'computer_room_baseline080'   # .blend 가 있는 실제 장면명",
"gt_ok = os.path.isdir(os.path.join(PROJECT_ROOT, 'experiments', GT_SCENE))",
"if not gt_ok:",
"    print(f'experiments/{GT_SCENE} 없음 → 먼저 그 장면으로 1~3단계를 돌리거나, GT 단계를 건너뜀')",
"else:",
"    cmd = [PY, os.path.join('scripts','make_mirror_gt.py'), '--scene', GT_SCENE]",
"    print('RUN:', ' '.join(cmd)); print('-'*60)",
"    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode",
"    print('-'*60, 'rc', rc)",
"    gtj = os.path.join(PROJECT_ROOT,'experiments',GT_SCENE,'mirror_gt','gt.json')",
"    if rc == 0 and os.path.isfile(gtj):",
"        g = json.load(open(gtj, encoding='utf-8'))",
"        for m in g['mirrors']:",
"            print('  %s: perp %.2f m, planarity_rms %.4f'%(m['name'], m['perp_distance'], m['planarity_rms'] or 0))",
"        ov = os.path.join(PROJECT_ROOT,'experiments',GT_SCENE,'mirror_gt','overlay.png')",
"        plt.figure(figsize=(15,5)); plt.imshow(imageio.imread(ov)); plt.axis('off'); plt.show()",
"    else:",
"        print('GT 생성 실패 — Blender 경로/.blend 존재 확인 (make_mirror_gt.py 의 BLENDER 상수)')",
))

cells.append(md("## 요약", "",
"- **1~6.5단계**: `left/right/mirror_mask` 만으로 재현 — 검출, AHCF RQ, RANSAC 반사면, **반사대칭 거울평면(정공법)**.",
"- **7단계 GT**: Blender + `.blend` 필요 (ds_v1/v2 신규 데이터셋은 GT depth 역투영 `make_mirror_gt_from_depth.py` 로 대체 가능).",
"- **주의(스케일)**: K 는 렌즈 가정값이라 *절대 깊이/거리* 가 (가정/실제)배 어긋날 수 있다. disparity·검출·AHCF AUC 는 K 무관이라 영향 없음. (신규 ds_v1/v2 는 실제 intrinsic 동봉이라 무관)",
"- **논문 최종 수치(1,537뷰)**: 거울 disp 오차 ~41× · 미지 장면 검출 AUC 0.928 · 정공법 평면 2.3°/0.14 m/성공률 72% (naive 0.1%).",
))

for i, c in enumerate(cells):          # 각 셀에 고유 id 부여(nbformat 5.x 권장)
    c["id"] = f"cell{i:02d}"

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mirror_pipeline_demo.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", out, "cells:", len(cells))
