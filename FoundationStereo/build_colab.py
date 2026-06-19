# -*- coding: utf-8 -*-
"""Colab용 산출물 생성:
   1) colab/colab_bundle.zip  — scripts/ + 샘플 입력(computer_room) 묶음 (Colab 업로드용)
   2) colab/mirror_pipeline_colab.ipynb — Colab 전용 노트북
한 번 실행 후 build_colab.py 는 지워도 됨."""
import json, os, glob, zipfile, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.normpath(os.path.join(
    ROOT, "..", "MakeStereoDataset", "data", "stereo_dataset", "computer_room_baseline080"))
OUT = os.path.join(ROOT, "colab")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- 1) 번들 zip
zip_path = os.path.join(OUT, "colab_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for py in glob.glob(os.path.join(ROOT, "scripts", "*.py")):
        z.write(py, os.path.join("scripts", os.path.basename(py)))
    # 샘플 입력 (left/right/mask + meta). mirror_mask_left -> mirror_mask 로 이름 통일.
    pairs = [("left.png", "left.png"), ("right.png", "right.png"),
             ("mirror_mask_left.png", "mirror_mask.png"), ("stereo_meta.json", "stereo_meta.json")]
    for src_name, dst_name in pairs:
        sp = os.path.join(DATASET, src_name)
        if os.path.isfile(sp):
            z.write(sp, os.path.join("sample", dst_name))
print("WROTE", zip_path, "size %.1f MB" % (os.path.getsize(zip_path) / 1e6))

# ----------------------------------------------------------------- 2) 노트북
def md(*l): return {"cell_type": "markdown", "metadata": {}, "source": _s(l)}
def code(*l): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _s(l)}
def _s(lines):
    t = "\n".join(lines).split("\n")
    return [p + "\n" for p in t[:-1]] + [t[-1]]

cells = []

cells.append(md(
"# Mirror Geometry / FoundationStereo — **Google Colab** 재현 노트북",
"",
"FS prior 만으로 **거울 검출 / AHCF 신호 분석 / 거울 평면(RANSAC)** 을 한 장면에 재현한다.",
"",
"## 사용법 (위에서 아래로 순서대로 실행)",
"1. **런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU (T4)** 로 설정.",
"2. 셀들을 순서대로 실행.",
"3. '번들 업로드' 셀에서 `colab_bundle.zip`(≈6MB, scripts+샘플) 을 올린다.",
"4. '모델 업로드' 셀에서 `colab_model_vitl.zip`(≈1.5GB, **ViT-L** 추론용 슬림 체크포인트) 을 올린다.",
"   → **Google Drive 연결 불필요.** 단 1.5GB라 업로드가 느릴 수 있다(수~수십 분, 회선 속도에 따라).",
"",
"## 입력",
"- 번들에 **샘플 장면(computer_room)** 의 `left/right/mirror_mask` 가 들어있어 그대로 실행된다.",
"- 본인 데이터로 바꾸려면: `sample/` 안의 `left.png / right.png / mirror_mask.png` 를 교체(업로드)하면 된다.",
"",
"## 안 되는 것",
"- **거울 평면 GT(gt.json/overlay)** 는 Blender + `.blend` 원본이 필요 → Colab 에서는 생략.",
))

cells.append(md("## 1. GPU 확인"))
cells.append(code(
"import torch, subprocess",
"print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())",
"assert torch.cuda.is_available(), '런타임 > 런타임 유형 변경 > GPU 로 바꾸세요'",
"print(torch.cuda.get_device_name(0))",
))

cells.append(md("## 2. 번들 업로드 & 압축 해제",
"`colab_bundle.zip`(scripts + 샘플) 을 올린다. 이미 풀려 있으면 건너뛴다."))
cells.append(code(
"import os, zipfile, sys",
"PROJECT_ROOT = '/content/FoundationStereo'",
"if not os.path.isdir(os.path.join(PROJECT_ROOT, 'scripts')):",
"    from google.colab import files",
"    print('colab_bundle.zip 을 선택하세요...')",
"    up = files.upload()",
"    zname = next((k for k in up if k.endswith('.zip')), None)",
"    assert zname, 'zip 파일이 필요합니다'",
"    os.makedirs(PROJECT_ROOT, exist_ok=True)",
"    with zipfile.ZipFile(zname) as z: z.extractall(PROJECT_ROOT)",
"    print('extracted ->', PROJECT_ROOT)",
"else:",
"    print('이미 존재:', PROJECT_ROOT)",
"assert os.path.isfile(os.path.join(PROJECT_ROOT,'scripts','common.py')), 'scripts/common.py 가 안 보임'",
"print(sorted(os.listdir(PROJECT_ROOT)))",
))

cells.append(md("## 3. FoundationStereo repo clone (GitHub)"))
cells.append(code(
"repo = os.path.join(PROJECT_ROOT, 'repo')",
"if not os.path.isfile(os.path.join(repo, 'core', 'foundation_stereo.py')):",
"    !git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git {repo}",
"else:",
"    print('repo 이미 존재')",
"print('repo ok:', os.path.isfile(os.path.join(repo,'core','foundation_stereo.py')))",
))

cells.append(md("## 4. 의존성 설치",
"torch 는 Colab 내장(이미 CUDA 빌드)을 쓰고, repo 가 필요한 나머지를 설치한다.",
"`open3d` 는 repo 의 `Utils.py` 가 상단에서 import 하므로 **반드시 필요**하다(없으면 FS 추론이 import 단계에서 실패)."))
cells.append(code(
"!pip install -q open3d pandas omegaconf timm einops \"ruamel.yaml\" huggingface-hub opencv-contrib-python scikit-image trimesh gdown",
"print('deps installed')",
"# open3d 가 numpy 를 내릴 수 있음 → 'numpy' 관련 에러가 나면: 런타임 > 세션 다시 시작 후 이 셀부터 재실행",
))

cells.append(md("## 5. 모델 업로드 (`colab_model_vitl.zip`, ≈1.5GB)",
"",
"이 노트북과 함께 받은 **`colab_model_vitl.zip`** 을 올린다. **ViT-L(23-51-11, 최고 성능)** 체크포인트에서 추론에",
"불필요한 부분(옵티마이저·백업 가중치)을 떼어내 **3.1GB → ~1.5GB** 로 줄인 추론 전용 슬림 버전이다.",
"**Google Drive 연결이 필요 없다.** 단 1.5GB라 업로드가 느릴 수 있다(회선에 따라 수~수십 분)."))
cells.append(code(
"import os, zipfile",
"pre = os.path.join(PROJECT_ROOT, 'pretrained_models')",
"def have_ckpt():",
"    return [os.path.join(r,f) for r,_,fs in os.walk(pre) for f in fs if f.endswith('.pth')]",
"",
"if not have_ckpt():",
"    from google.colab import files",
"    print('colab_model_vitl.zip (약 1.5GB) 를 선택하세요... (업로드에 수~수십 분 걸릴 수 있음)')",
"    up = files.upload()",
"    zname = next((k for k in up if k.endswith('.zip')), None)",
"    assert zname, 'zip 파일이 필요합니다'",
"    os.makedirs(pre, exist_ok=True)",
"    with zipfile.ZipFile(zname) as z: z.extractall(pre)   # 23-51-11/ 가 pretrained_models/ 아래로",
"    print('extracted ->', pre)",
"else:",
"    print('모델 이미 존재')",
"",
"ck = have_ckpt()",
"print('checkpoints:', ck)",
"assert ck, '체크포인트 없음 — colab_model_vitl.zip 을 업로드했는지 확인'",
))

cells.append(md("## 6. 설정 & 입력 미리보기"))
cells.append(code(
"sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))",
"import importlib, common as C; importlib.reload(C)",
"import numpy as np, json, shutil",
"import matplotlib.pyplot as plt, imageio.v2 as imageio",
"PY = sys.executable",
"",
"SCENE     = 'demo_computer_room'",
"INPUT_DIR = os.path.join(PROJECT_ROOT, 'sample')   # left.png/right.png/mirror_mask.png",
"BASELINE_M, LENS_MM, SENSOR_MM = 0.08, 50.0, 36.0  # 실제 lens 알면 바꾸기(절대 깊이 스케일)",
"",
"def find_one(d, names):",
"    for n in names:",
"        p=os.path.join(d,n)",
"        if os.path.isfile(p): return p",
"    return None",
"left_src  = find_one(INPUT_DIR, ['left.png'])",
"right_src = find_one(INPUT_DIR, ['right.png'])",
"mask_src  = find_one(INPUT_DIR, ['mirror_mask.png','mirror_mask_left.png'])",
"assert left_src and right_src, 'sample/ 에 left.png, right.png 필요'",
"print('mask:', mask_src if mask_src else '(없음)')",
"imgs=[('left',left_src),('right',right_src)]+([('mask',mask_src)] if mask_src else [])",
"fig,ax=plt.subplots(1,len(imgs),figsize=(5*len(imgs),4)); ax=np.atleast_1d(ax)",
"for a,(t,p) in zip(ax,imgs): a.imshow(imageio.imread(p)); a.set_title(t); a.axis('off')",
"plt.tight_layout(); plt.show()",
))

cells.append(md("## 7. 입력 준비 (left/right/mask 만으로 K.txt 생성)"))
cells.append(code(
"run_dir=os.path.join(PROJECT_ROOT,'experiments',SCENE); in_dir=C.ensure_dir(os.path.join(run_dir,'input'))",
"h,w=imageio.imread(left_src).shape[:2]",
"K=C.intrinsics_from_assumption(w,h,lens_mm=LENS_MM,sensor_mm=SENSOR_MM)",
"shutil.copyfile(left_src,os.path.join(in_dir,'left.png')); shutil.copyfile(right_src,os.path.join(in_dir,'right.png'))",
"if mask_src: shutil.copyfile(mask_src,os.path.join(in_dir,'mirror_mask.png'))",
"C.write_K_txt(os.path.join(in_dir,'K.txt'),K,BASELINE_M)",
"sm_path=os.path.join(INPUT_DIR,'stereo_meta.json'); sm=json.load(open(sm_path)) if os.path.isfile(sm_path) else {}",
"json.dump(dict(scene=SCENE,width=w,height=h,baseline_m=BASELINE_M,K=K.tolist(),K_is_assumed=True,",
"               lens_mm=LENS_MM,sensor_mm=SENSOR_MM,",
"               left_camera_to_world=sm.get('left_camera_to_world'),",
"               right_camera_to_world=sm.get('right_camera_to_world'),",
"               mirror_objects=sm.get('mirror_objects',sm.get('mirror_object'))),",
"          open(os.path.join(in_dir,'prepare_meta.json'),'w'),indent=2)",
"print('prepared',run_dir,'size',(w,h))",
))

cells.append(md("## 8. FoundationStereo 추론 + prior 추출 (GPU, 수십초~분)"))
cells.append(code(
"r=subprocess.run([PY,'scripts/run_foundation_stereo.py','--scene',SCENE],cwd=PROJECT_ROOT)",
"assert r.returncode==0, 'FS 추론 실패 (위 로그 확인)'",
"vis=os.path.join(run_dir,'fs','vis_disp.png'); depth=np.load(os.path.join(run_dir,'fs','depth_meter.npy'))",
"fig,ax=plt.subplots(1,2,figsize=(15,4))",
"ax[0].imshow(imageio.imread(vis)); ax[0].set_title('left | disparity'); ax[0].axis('off')",
"v=depth>0; im=ax[1].imshow(np.where(v,depth,np.nan),cmap='turbo'); ax[1].set_title('depth(m)'); ax[1].axis('off')",
"fig.colorbar(im,ax=ax[1],fraction=0.046); plt.tight_layout(); plt.show()",
))

cells.append(md("## 9. AHCF 신호 분석 (RQ)",
"`pre_*`(우연≈0.51) → `post_*`/`delta_*` 상승이면 AHCF 가 거울 신호를 만든 것."))
cells.append(code(
"r=subprocess.run([PY,'scripts/analyze_features.py','--scene',SCENE],cwd=PROJECT_ROOT); assert r.returncode==0",
"rep=json.load(open(os.path.join(run_dir,'features','report.json'),encoding='utf-8'))",
"print('best:',rep['best_signal'],'AUC',rep['best_auc'])",
"for i,rw in enumerate(rep['ranked'][:8],1): print(f\"{i:>2} {rw['signal']:<18} AUC {rw['auc']} ({rw['polarity']})\")",
"plt.figure(figsize=(15,9)); plt.imshow(imageio.imread(os.path.join(run_dir,'features','panels.png'))); plt.axis('off'); plt.show()",
))

cells.append(md("## 10. 거울 평면 RANSAC (한계: '반사면')"))
cells.append(code(
"if not mask_src: print('mask 없음 → 생략')",
"else:",
"    r=subprocess.run([PY,'scripts/estimate_mirror_plane.py','--scene',SCENE],cwd=PROJECT_ROOT); assert r.returncode==0",
"    res=json.load(open(os.path.join(run_dir,'mirror_geometry','result.json'),encoding='utf-8'))",
"    rr=res['recommended_result']",
"    print('normal',[round(x,3) for x in rr['normal']],'perp %.2f m'%res['mirror_depth']['perp_distance'])",
"    print('rms PCA %.3f vs RANSAC %.3f'%(res['methods']['pca']['rms_residual'],res['methods']['ransac']['rms_residual']))",
"    viz=os.path.join(run_dir,'mirror_geometry','viz.png')",
"    if os.path.isfile(viz): plt.figure(figsize=(13,5)); plt.imshow(imageio.imread(viz)); plt.axis('off'); plt.show()",
))

cells.append(md("## 11. 거울 검출 — FS prior 위 경량 헤드 (접근 #3)"))
cells.append(code(
"if not mask_src: print('mask 없음 → 생략')",
"else:",
"    r=subprocess.run([PY,'scripts/submodel_head.py','--scene',SCENE],cwd=PROJECT_ROOT); assert r.returncode==0",
"    mt=json.load(open(os.path.join(run_dir,'submodel','metrics.json'),encoding='utf-8'))['metrics']",
"    print('AUC %.3f IoU %.3f F1 %.3f'%(mt['auc'],mt['iou'],mt['f1']))",
"    plt.figure(figsize=(15,5)); plt.imshow(imageio.imread(os.path.join(run_dir,'submodel','panel.png'))); plt.axis('off'); plt.show()",
))

cells.append(md("## 요약 / 주의",
"- `left/right/mirror_mask` 만으로 **검출·AHCF·평면** 재현 완료.",
"- **GT 평면**은 Blender+`.blend` 필요 → Colab 생략.",
"- **스케일 주의**: K 는 렌즈 가정값이라 *절대 깊이/거리* 가 (가정/실제)배 어긋날 수 있음. disparity·검출·AHCF AUC 는 K 무관. 정확한 절대 거리가 필요하면 실제 lens 를 `LENS_MM` 에 넣어 7단계부터 재실행.",
))

for i, c in enumerate(cells): c["id"] = f"cell{i:02d}"
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
nb_path = os.path.join(OUT, "mirror_pipeline_colab.ipynb")
json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", nb_path, "cells:", len(cells))
