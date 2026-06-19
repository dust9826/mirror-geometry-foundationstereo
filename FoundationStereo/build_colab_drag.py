# -*- coding: utf-8 -*-
"""드래그-업로드(zip)용 Colab 노트북 생성. colab/mirror_pipeline_colab_drag.ipynb
   files.upload() 위젯 대신, colab_all.zip 을 파일 패널에 드래그 → 압축해제 → 실행."""
import json, os

def md(*l): return {"cell_type":"markdown","metadata":{},"source":_s(l)}
def code(*l): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":_s(l)}
def _s(lines):
    t="\n".join(lines).split("\n"); return [p+"\n" for p in t[:-1]]+[t[-1]]

cells=[]
cells.append(md(
"# Mirror Geometry / FoundationStereo — Colab (zip 드래그판)",
"",
"`files.upload()` 위젯 없이, **`colab_all.zip` 을 Colab 좌측 파일 패널에 드래그**해서 올린 뒤 실행한다(빠름).",
"",
"## 0. 먼저 (1회)",
"1. **런타임 → 런타임 유형 변경 → GPU(T4)**.",
"2. 좌측 **파일 패널**(폴더 아이콘)을 열고 — 기본 위치 `/content` — 거기에 **`colab_all.zip`** 을 드래그.",
"   (이 zip 안에 `scripts/` + 샘플 input(`experiments/demo/input/`) + 모델(`pretrained_models/11-33-40/`, ViT-S 250MB) 전부 들어있음.)",
"3. 업로드 다 된 뒤(파일 패널에 `colab_all.zip` 보이면) 아래 셀들을 순서대로 실행.",
"",
"> `repo/`(NVlabs FoundationStereo)만 아래에서 GitHub clone(드래그보다 빠름).",
"> 모델은 `colab_all.zip`(ViT-S) 또는 `colab_all_vitl.zip`(ViT-L) **하나** 선택해 드래그.",
"> **LOSO**(여러 장면 일반화, §7)까지 하려면 `colab_loso.zip`(12장면 features, ~100MB)도 함께 드래그.",
"",
"파이프라인: FS 추론 → 거울 검출(경량 헤드) → **반사대칭 거울평면 복원(§5.5, 학습 없음)** → GT 평가.",
))

cells.append(md("## 1. GPU 확인 + 경로"))
cells.append(code(
"import torch, os, sys, subprocess, glob, zipfile",
"print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())",
"assert torch.cuda.is_available(), '런타임>유형변경>GPU 로 설정'",
"print(torch.cuda.get_device_name(0))",
"PROJECT_ROOT='/content'",
))

cells.append(md("## 2. zip 압축 해제 (+ repo clone + 의존성)",
"`/content` 의 모든 `*.zip` 을 푼다. open3d 는 repo Utils.py 가 import 하므로 **필수**."))
cells.append(code(
"zips=glob.glob('/content/*.zip')",
"assert zips, '/content 에 colab_all.zip 을 드래그하세요 (파일 패널)'",
"for z in zips:",
"    with zipfile.ZipFile(z) as f: f.extractall(PROJECT_ROOT)",
"    print('압축 해제:', os.path.basename(z))",
"repo=os.path.join(PROJECT_ROOT,'repo')",
"if not os.path.isfile(os.path.join(repo,'core','foundation_stereo.py')):",
"    !git clone --depth 1 https://github.com/NVlabs/FoundationStereo.git {repo}",
"else: print('repo 이미 있음')",
"!pip install -q open3d pandas omegaconf timm einops \"ruamel.yaml\" huggingface-hub opencv-contrib-python scikit-image trimesh gdown",
"print('repo ok:', os.path.isfile(os.path.join(repo,'core','foundation_stereo.py')))",
"# numpy 관련 에러 시: 런타임>세션 다시 시작 후 이 셀부터 재실행",
))

cells.append(md("## 3. 압축 해제 결과 확인"))
cells.append(code(
"sys.path.insert(0, os.path.join(PROJECT_ROOT,'scripts'))",
"ok_scripts=os.path.isfile(os.path.join(PROJECT_ROOT,'scripts','common.py'))",
"ckpts=glob.glob(os.path.join(PROJECT_ROOT,'pretrained_models','**','*.pth'), recursive=True)",
"inputs=glob.glob(os.path.join(PROJECT_ROOT,'experiments','*','input','left.png'))",
"print('scripts/common.py :', ok_scripts)",
"print('checkpoints       :', ckpts)",
"print('input scene       :', [p.split(os.sep)[-3] for p in inputs])",
"assert ok_scripts and ckpts and inputs, 'zip 압축 해제 확인 — colab_all.zip 이 제대로 풀렸는지'",
))

cells.append(md("## 4. FS 추론 (에러가 보이도록 출력 캡처)"))
cells.append(code(
"SCENE=[p.split(os.sep)[-3] for p in inputs][0]; print('SCENE =', SCENE)",
"r=subprocess.run([sys.executable,'scripts/run_foundation_stereo.py','--scene',SCENE],",
"                 cwd=PROJECT_ROOT, capture_output=True, text=True)",
"print('RC =', r.returncode); print(r.stdout[-2000:])",
"if r.returncode!=0:",
"    print('===== STDERR =====\\n', r.stderr[-6000:]); raise SystemExit('FS 추론 실패 — 위 STDERR 확인')",
))
cells.append(code(
"import numpy as np, imageio.v2 as imageio, matplotlib.pyplot as plt",
"run_dir=os.path.join(PROJECT_ROOT,'experiments',SCENE)",
"depth=np.load(os.path.join(run_dir,'fs','depth_meter.npy'))",
"fig,ax=plt.subplots(1,2,figsize=(15,4))",
"ax[0].imshow(imageio.imread(os.path.join(run_dir,'fs','vis_disp.png'))); ax[0].set_title('left | disparity'); ax[0].axis('off')",
"v=depth>0; im=ax[1].imshow(np.where(v,depth,np.nan),cmap='turbo'); ax[1].set_title('depth(m)'); ax[1].axis('off')",
"fig.colorbar(im,ax=ax[1],fraction=0.046); plt.tight_layout(); plt.show()",
))

cells.append(md("## 5. (선택) 나머지 단계 (평면·AHCF·검출 + GT depth 평가)"))
cells.append(code(
"for s in ['estimate_mirror_plane.py','viz_ransac_inliers.py','analyze_features.py','submodel_head.py']:",
"    r=subprocess.run([sys.executable,'scripts/'+s,'--scene',SCENE], cwd=PROJECT_ROOT, capture_output=True, text=True)",
"    print(f'[{s}] RC={r.returncode}')",
"    if r.returncode!=0: print(r.stderr[-1500:])",
"if os.path.isfile(os.path.join(run_dir,'input','depth_gt.npy')):",
"    subprocess.run([sys.executable,'scripts/make_mirror_gt_from_depth.py','--scene',SCENE], cwd=PROJECT_ROOT)",
"    subprocess.run([sys.executable,'scripts/eval_mirror_plane.py','--scene',SCENE], cwd=PROJECT_ROOT)",
))

cells.append(md("## 5.5 반사대칭 거울평면 복원 (정공법, 학습 없음)",
"",
"거울 안 점군의 단순 평면 적합은 거울면이 아니라 **반사면**(거울 뒤 가상공간, 약 2배 깊음)을 잡는다.",
"이를 반사 대칭으로 교정한다: 거울 안 가상점 P′와 방의 실제점 P는 거울평면에 대해 대칭이므로,",
"대응쌍 하나의 **수직이등분면**이 평면을 닫힌형으로 결정한다 → RANSAC + trimmed-NN 점수 + ICP 정제.",
"학습이 전혀 없고 FS 출력만 사용한다. (논문 §3.3 / 1,536뷰에서 법선 2.3°·거리오차 0.14 m)"))
cells.append(code(
"import json",
"r=subprocess.run([sys.executable,'scripts/estimate_mirror_plane_sym.py','--scene',SCENE,'--quiet'],",
"                 cwd=PROJECT_ROOT, capture_output=True, text=True)",
"print(r.stdout[-800:])",
"if r.returncode!=0: print('===== STDERR =====\\n', r.stderr[-2000:])",
"sy=json.load(open(os.path.join(run_dir,'mirror_geometry_sym','result.json'),encoding='utf-8')) \\",
"   if os.path.isfile(os.path.join(run_dir,'mirror_geometry_sym','result.json')) else None",
"g=json.load(open(os.path.join(run_dir,'mirror_geometry','result.json'),encoding='utf-8')) \\",
"  if os.path.isfile(os.path.join(run_dir,'mirror_geometry','result.json')) else None",
"if sy:",
"    print('\\n===== 비교: 단순 평면 적합 vs 반사대칭 =====')",
"    if g and g.get('mirror_depth'):",
"        print(f\"  단순 적합(반사면)  perp = {g['mirror_depth']['perp_distance']:.2f} m   <- 거울 뒤 가상공간\")",
"    print(f\"  반사대칭(정공법)   perp = {sy['perp']:.2f} m   normal {[round(x,3) for x in sy['normal']]}\")",
"    ev=sy.get('eval_vs_gt')",
"    if ev: print(f\"  GT 대비           각도 {ev['angle_deg']:.1f}°, perp 오차 {ev['perp_err_m']:.2f} m (GT {ev['gt_perp']:.2f} m)\")",
))

cells.append(md("## 6. 결과 종합 출력",
"이 장면(`SCENE`)의 모든 산출물(수치 + 그림)을 한 셀에서 모아 본다. 있는 것만 표시."))
cells.append(code(
"import json, matplotlib.pyplot as plt, imageio.v2 as imageio",
"def _show(p, title):",
"    if os.path.isfile(p):",
"        plt.figure(figsize=(14,5)); plt.imshow(imageio.imread(p)); plt.title(title); plt.axis('off'); plt.show()",
"def _j(p):",
"    return json.load(open(p,encoding='utf-8')) if os.path.isfile(p) else None",
"",
"print('='*56); print('결과 요약 —', SCENE); print('='*56)",
"# --- 수치 ---",
"r=_j(os.path.join(run_dir,'features','report.json'))",
"if r: print(f\"AHCF 최고 신호 : {r['best_signal']}  AUC {r['best_auc']}\")",
"g=_j(os.path.join(run_dir,'mirror_geometry','result.json'))",
"if g: print(f\"RANSAC 평면    : perp {g['mirror_depth']['perp_distance']:.2f} m, \"",
"            f\"rms {g['methods']['ransac']['rms_residual']:.3f} (PCA {g['methods']['pca']['rms_residual']:.3f})\")",
"for sub,name in [('submodel','검출(단일)'),('submodel_applied','검출(범용적용)')]:",
"    m=_j(os.path.join(run_dir,sub,'metrics.json'))",
"    if m:",
"        m=m.get('metrics',m); print(f\"{name:14s}: AUC {m['auc']:.3f}  IoU {m['iou']:.3f}  F1 {m['f1']:.3f}\")",
"gt=_j(os.path.join(run_dir,'mirror_gt','gt.json'))",
"if gt and gt.get('mirrors'):",
"    mm=gt['mirrors'][0]",
"    print(f\"GT 거울평면    : perp {mm['perp_distance']:.2f} m, planarity_rms {mm['planarity_rms']:.4f} m\")",
"sy=_j(os.path.join(run_dir,'mirror_geometry_sym','result.json'))",
"if sy:",
"    ev=sy.get('eval_vs_gt') or {}",
"    print(f\"반사대칭 평면  : perp {sy['perp']:.2f} m\" +",
"          (f\"  (GT 대비 각도 {ev['angle_deg']:.1f}°, perp 오차 {ev['perp_err_m']:.2f} m)\" if ev else ''))",
"# --- GT depth/disparity 평가(있으면) ---",
"import numpy as np",
"dg=os.path.join(run_dir,'input','disp_gt.npy'); fsd=os.path.join(run_dir,'fs','disp.npy')",
"if os.path.isfile(dg) and os.path.isfile(fsd):",
"    d=np.load(fsd); dgt=np.load(dg)",
"    mk=imageio.imread(os.path.join(run_dir,'input','mirror_mask.png')); mk=(mk if mk.ndim==2 else mk[...,:3].mean(-1))>127",
"    v=(dgt>0)&np.isfinite(dgt)&(d>0)",
"    for nm,sel in [('비거울',v&~mk),('거울',v&mk)]:",
"        if sel.sum(): print(f\"  disp MAE({nm}): {np.abs(d[sel]-dgt[sel]).mean():.2f} px\")",
"",
"# --- 그림 ---",
"_show(os.path.join(run_dir,'fs','vis_disp.png'), 'disparity')",
"_show(os.path.join(run_dir,'features','panels.png'), 'AHCF 전/후 신호 분석')",
"_show(os.path.join(run_dir,'mirror_geometry','viz.png'), '거울 평면 (RANSAC)')",
"_show(os.path.join(run_dir,'mirror_geometry','ransac_inliers.png'), 'RANSAC inlier/outlier')",
"_show(os.path.join(run_dir,'submodel','panel.png'), '검출 (단일 scene)')",
"_show(os.path.join(run_dir,'submodel_applied','panel.png'), '검출 (범용모델 적용)')",
"_show(os.path.join(run_dir,'mirror_gt','overlay.png'), 'GT 거울평면 overlay')",
))

cells.append(md("## 7. 여러 scene 검출 일반화 (LOSO)",
"",
"미지 장면 일반화를 LOSO(leave-one-scene-out)로 측정한다. **FS 재추론 불필요** — 미리 뽑은 12개 장면의",
"`features.npz`+마스크만 있으면 된다.",
"",
"→ **`colab_loso.zip`**(12 scene features, ~100MB)을 파일 패널에 추가로 드래그하고 ②(압축해제) 셀을 다시 실행한 뒤,",
"아래를 실행. (각 fold = 11개로 학습→나머지 1개 평가, 작은 MLP 12회 학습이라 GPU에서 수 분.)"))
cells.append(code(
"import glob, json, numpy as np",
"loso=sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))",
"             for p in glob.glob(f'{PROJECT_ROOT}/experiments/*/fs/features.npz') if 'baseline080' in p})",
"print('LOSO scenes:', len(loso))",
"assert len(loso)>=3, 'colab_loso.zip 을 드래그하고 ②셀(압축해제) 재실행하세요'",
"sc=','.join(loso); res={}",
"for val in loso:",
"    out=f'experiments/_loso/{val}'",
"    r=subprocess.run([sys.executable,'scripts/submodel_head.py','--scenes',sc,'--val_scene',val,'--out',out],",
"                     cwd=PROJECT_ROOT, capture_output=True, text=True)",
"    try:",
"        m=json.load(open(os.path.join(PROJECT_ROOT,out,'metrics.json')))['metrics']",
"        res[val]=(m['auc'],m['iou']); print(f\"  {val:30s} AUC {m['auc']:.3f}  IoU {m['iou']:.3f}\")",
"    except Exception:",
"        print(f'  {val}: 실패\\n', r.stderr[-400:])",
"a=[v[0] for v in res.values()]; iou=[v[1] for v in res.values()]",
"print(f'\\n=== LOSO {len(res)}-fold: 평균 AUC {np.mean(a):.3f} (중앙값 {np.median(a):.3f}) / IoU {np.mean(iou):.3f} ===')",
))

cells.append(md("## 8. LOSO 결과 출력 (표 + 그래프)",
"§7 의 fold 결과(`experiments/_loso/*/metrics.json`)를 표·막대그래프로 정리. §7 실행 후 사용."))
cells.append(code(
"import json, glob, numpy as np, matplotlib.pyplot as plt",
"rows=[]",
"for mp in sorted(glob.glob(f'{PROJECT_ROOT}/experiments/_loso/*/metrics.json')):",
"    sc=os.path.basename(os.path.dirname(mp)); m=json.load(open(mp,encoding='utf-8'))['metrics']",
"    rows.append((sc, m['auc'], m['iou']))",
"assert rows, '먼저 §7 LOSO 셀을 실행하세요'",
"rows.sort(key=lambda r:-r[1])",
"print(f\"{'scene':32s} {'AUC':>6s} {'IoU':>6s}\")",
"for sc,au,io in rows: print(f'{sc[:32]:32s} {au:6.3f} {io:6.3f}')",
"aucs=[r[1] for r in rows]; ious=[r[2] for r in rows]",
"print(f'\\nLOSO {len(rows)}-fold  평균 AUC {np.mean(aucs):.3f} (중앙값 {np.median(aucs):.3f})  /  IoU 평균 {np.mean(ious):.3f}')",
"labels=[r[0].replace('_baseline080','') for r in rows]; x=np.arange(len(labels))",
"fig,ax=plt.subplots(figsize=(max(7,1.1*len(labels)),4))",
"ax.bar(x-0.2,aucs,0.4,label='AUC',color='#1f5f8b'); ax.bar(x+0.2,ious,0.4,label='IoU',color='#c47f00')",
"ax.axhline(np.mean(aucs),ls='--',c='#1f5f8b',lw=1)",
"ax.set_xticks(x); ax.set_xticklabels(labels,rotation=40,ha='right',fontsize=8)",
"ax.set_ylim(0,1); ax.set_ylabel('score'); ax.legend()",
"ax.set_title(f'LOSO per-scene  (mean AUC {np.mean(aucs):.3f}, IoU {np.mean(ious):.3f})')",
"plt.tight_layout(); plt.show()",
))

cells.append(md("## 9. LOSO fold별 예측 패널 (panel.png)",
"각 fold(미지 장면)의 예측 결과 이미지. 좌=입력 / 중=예측 거울확률 / 우=GT 마스크. §7 실행 후 사용."))
cells.append(code(
"import json, glob, os, imageio.v2 as imageio, matplotlib.pyplot as plt",
"mps=sorted(glob.glob(f'{PROJECT_ROOT}/experiments/_loso/*/metrics.json'))",
"assert mps, '먼저 §7 LOSO 셀을 실행하세요'",
"for mp in mps:",
"    sc=os.path.basename(os.path.dirname(mp)); m=json.load(open(mp,encoding='utf-8'))['metrics']",
"    pp=os.path.join(os.path.dirname(mp),'panel.png')",
"    if os.path.isfile(pp):",
"        plt.figure(figsize=(14,4)); plt.imshow(imageio.imread(pp))",
"        plt.title(f\"{sc}   AUC {m['auc']:.3f}  IoU {m['iou']:.3f}\"); plt.axis('off'); plt.show()",
))

cells.append(md("## 10. (참고) 논문 최종 수치 — 12장면 × 2시드 = 1,537뷰 전수 평가",
"",
"이 노트북의 데모는 1개 장면이지만, 동일 파이프라인을 전체 벤치마크에 돌린 논문 수치는 다음과 같다",
"(IPIU 2026, *스테레오 Prior에 내재된 거울 기하의 복원*).",
"",
"**R1. FoundationStereo의 거울 실패 (뷰별 중앙값)**",
"",
"| 지표 | 비거울 | 거울 | 배율 |",
"|---|---|---|---|",
"| disp MAE (px) | 0.21 | 8.72 | ~41× |",
"| depth MAE (m) | 0.151 | 2.57 | ~17× |",
"",
"**R2. 거울 검출 (픽셀당 136-d prior + MLP 136→64→1)**",
"",
"| 체제 | AUC | IoU |",
"|---|---|---|",
"| single_scene (자기 장면) | 0.985 | 0.715 |",
"| view_split (새 시점) | 0.958 | 0.546 |",
"| rep_view (씬당 1뷰=12장 학습) | 0.955 | 0.540 |",
"| **scene_split (미지 장면)** | **0.928** | **0.492** |",
"",
"**R3. 반사대칭 거울평면 복원 (1,536뷰, GT mask, 학습 없음)**",
"",
"| 지표 | 값 |",
"|---|---|",
"| 법선각 중앙값 | **2.3°** (<10°: 92%) |",
"| 수직거리 오차 중앙값 | **0.14 m** (<0.5m: 74%) |",
"| 동시 충족(<10° & <0.5m) | **72%** |",
"| (비교) 단순 평면 적합 | 87.5° / 2.52 m / 0.1% |",
"",
"실패 모드는 *d-슬라이딩* 퇴화(거울이 관측 밖을 비춰 반사점의 대응이 없는 경우)로 규명되었다."))

for i,c in enumerate(cells): c["id"]=f"cell{i:02d}"
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU","colab":{"provenance":[]}},
    "nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"colab","mirror_pipeline_colab_drag.ipynb")
json.dump(nb, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("WROTE", out, "cells", len(cells))
