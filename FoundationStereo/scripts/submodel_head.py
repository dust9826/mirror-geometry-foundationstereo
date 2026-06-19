"""
submodel_head.py — FoundationStereo prior 특징 위에 얹는 경량 거울 검출 헤드 (제안서 접근 #3).

제안서 접근 #3 "서브 모델 학습":
  FoundationStereo 자체는 재학습하지 않는다. 대신 FS 가 이미 뽑아 둔 prior 특징
  (side-tuning feature `feat_left`, cost-volume 통계 pre/post conf·entropy·argdisp)을
  입력으로 받아, 픽셀별로 (a) 거울 확률 마스크를 예측하는 아주 작은 헤드(sub-model)를
  학습한다. 무거운 stereo backbone 은 동결한 채 헤드만 학습하므로 가볍다.

무엇을 검증하는가:
  "FS 의 prior 특징이 거울-ness 를 인코딩하는가?" — 작은 선형/MLP 헤드만으로도
  거울 영역이 분리되면, FS 특징이 거울 신호를 담고 있다는 직접 증거가 된다.
  제안서가 요구하는 기하 방법(RANSAC/closed-form) 대비 비교의 학습 기반 baseline.

데이터 사정:
  여러 scene 에 걸친 라벨 학습셋이 아직 없으므로, 깔끔하게 돌아가는 SCAFFOLD 를 만든다.
    - 단일 scene: 그 scene 자신의 GT mirror_mask 를 라벨로 써서 픽셀 70/30 분할
      train/val (작은 overfit/cross-val 데모). FS 특징이 선형/MLP 로 분리 가능한지 검사.
    - 다중 scene: train scene 들의 픽셀 union 으로 학습하고 held-out val scene 으로 평가
      (일반화 수치).

입력 계약 (experiments/<scene>/):
  fs/features.npz : pre_conf, pre_entropy, pre_argdisp, post_conf, post_entropy,
                    post_argdisp, scale_to_full, (옵션) feat_left (C,h,w) float16.
                    모두 feature 해상도 (h,w).
  input/mirror_mask.png : GT 마스크(흰=거울). common.load_mask 로 로드 후
                          cv2.INTER_NEAREST 로 (h,w) 다운샘플하여 라벨로 사용.

출력 (experiments/<scene>/submodel/  또는 --out):
  pred_mask.png : 예측 거울 확률을 전체 해상도로 업샘플 + colormap.
  panel.png     : pred vs GT side-by-side (matplotlib Agg).
  metrics.json  : AUC/IoU/F1/accuracy + feature 목록 + config.
  head.pt       : torch.save 한 헤드 가중치 + 표준화 통계(mean/std).

사용:
  python scripts/submodel_head.py --scene cozy_living_room_baseline080     # 단일 scene 데모
  python scripts/submodel_head.py --scenes a,b,c --val_scene c             # 다중 scene
  python scripts/submodel_head.py --scene a --linear                       # 로지스틱 회귀 baseline
"""

import argparse
import json
import os
import sys

# scripts/ 자신을 path 에 넣어 프로젝트 루트에서 실행해도 import common 가능.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C


# --------------------------------------------------------------------- features
# cost-volume 통계 기반 feature 이름(항상 존재). feat_left 채널은 앞에 붙는다.
COST_FEATURE_NAMES = [
    "pre_conf", "post_conf", "pre_entropy", "post_entropy",
    "pre_argdisp", "post_argdisp", "delta_conf", "delta_entropy",
]


def build_features(scene, out_root=C.EXPERIMENTS_DIR):
    """scene 의 features.npz + GT mask 로 (X, y, (h,w), feature_names) 생성.

    X : (N_pixels, F) float32  픽셀별 feature 벡터
        = [feat_left 채널들(있으면)] + [pre_conf, post_conf, pre_entropy,
           post_entropy, pre_argdisp, post_argdisp, delta_conf, delta_entropy]
    y : (N_pixels,) float32  GT 거울 라벨(0/1), (h,w) 로 INTER_NEAREST 다운샘플.
    features.npz 가 없으면 None 반환(호출부에서 skip).
    """
    import cv2

    run_dir = os.path.join(out_root, scene)
    feat_path = os.path.join(run_dir, "fs", "features.npz")
    mask_path = os.path.join(run_dir, "input", "mirror_mask.png")
    if not os.path.exists(feat_path):
        print(f"[skip] {scene}: features.npz 없음 ({feat_path})")
        return None
    if not os.path.exists(mask_path):
        print(f"[skip] {scene}: mirror_mask.png 없음 ({mask_path})")
        return None

    z = np.load(feat_path)
    h, w = z["pre_conf"].shape[:2]

    # cost-volume 통계 채널 (h,w) 들을 쌓는다. delta = post - pre.
    delta_conf = z["post_conf"] - z["pre_conf"]
    delta_entropy = z["post_entropy"] - z["pre_entropy"]
    cost_stack = np.stack([
        z["pre_conf"], z["post_conf"], z["pre_entropy"], z["post_entropy"],
        z["pre_argdisp"], z["post_argdisp"], delta_conf, delta_entropy,
    ], axis=0).astype(np.float32)                       # (8,h,w)

    feature_names = list(COST_FEATURE_NAMES)
    chans = [cost_stack]

    # feat_left 가 있으면 채널 앞쪽에 붙인다 (없어도 robust).
    if "feat_left" in z.files:
        fl = z["feat_left"].astype(np.float32)          # (C,h,w)
        if fl.ndim == 3 and fl.shape[1:] == (h, w):
            chans = [fl, cost_stack]
            feature_names = [f"feat_left[{i}]" for i in range(fl.shape[0])] + feature_names
        else:
            print(f"[warn] {scene}: feat_left shape {fl.shape} != (C,{h},{w}) — 무시")

    feat = np.concatenate(chans, axis=0)                # (F,h,w)
    F = feat.shape[0]
    X = feat.reshape(F, h * w).T.astype(np.float32)     # (N,F)

    # GT mask 로드 후 (h,w) 로 INTER_NEAREST 다운샘플.
    mask_full = C.load_mask(mask_path).astype(np.uint8)  # (H,W) bool->0/1
    mask_small = cv2.resize(mask_full, (w, h), interpolation=cv2.INTER_NEAREST)
    y = mask_small.reshape(-1).astype(np.float32)        # (N,)

    return dict(X=X, y=y, hw=(h, w), feature_names=feature_names)


def standardize_fit(X):
    """feature mean/std 계산 (std==0 보호)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X, mean, std):
    return (X - mean) / std


# --------------------------------------------------------------------- model
def make_head(in_dim, linear=False, hidden=64):
    """경량 헤드. linear=True 면 순수 로지스틱 회귀, 아니면 작은 MLP."""
    import torch.nn as nn
    if linear:
        return nn.Linear(in_dim, 1)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 1),
    )


def train_head(Xtr, ytr, in_dim, linear=False, hidden=64, epochs=300,
               lr=1e-3, device="cpu", seed=0):
    """BCEWithLogitsLoss + Adam, class-imbalance 는 pos_weight 로 보정."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    head = make_head(in_dim, linear=linear, hidden=hidden).to(device)

    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=device).view(-1, 1)

    # pos_weight = neg/pos (양성 픽셀이 드물면 가중치를 키운다).
    n_pos = float(yt.sum().item())
    n_neg = float(yt.numel() - n_pos)
    pos_weight = torch.as_tensor([n_neg / max(n_pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    head.train()
    for ep in range(epochs):
        opt.zero_grad()
        logit = head(Xt)
        loss = crit(logit, yt)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  [train] epoch {ep + 1}/{epochs}  loss={loss.item():.4f}")
    return head


def predict_proba(head, X, device="cpu", batch=200000):
    """logits -> sigmoid 확률 (h*w 가 클 수 있어 배치 처리)."""
    import torch
    head.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            p = torch.sigmoid(head(xb)).view(-1).cpu().numpy()
            out.append(p)
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


# --------------------------------------------------------------------- metrics
def roc_auc(y, p):
    """ROC AUC. sklearn 있으면 사용, 없으면 numpy rank 기반 fallback."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        # Mann-Whitney U 통계 = AUC. 평균 rank 로 tie 처리.
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=np.float64)
        sp = p[order]
        i = 0
        while i < len(sp):
            j = i
            while j + 1 < len(sp) and sp[j + 1] == sp[i]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # 1-based 평균 rank
            i = j + 1
        sum_pos = ranks[y == 1].sum()
        auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc)


def binary_metrics(y, p, thr=0.5):
    """thr 기준 IoU/F1/accuracy/precision/recall."""
    y = np.asarray(y).astype(np.int64)
    pred = (np.asarray(p) >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    inter = tp
    union = tp + fp + fn
    iou = inter / union if union > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + fp + fn + tn))
    return dict(iou=float(iou), f1=float(f1), accuracy=float(acc),
                precision=float(prec), recall=float(rec),
                tp=tp, fp=fp, fn=fn, tn=tn)


# --------------------------------------------------------------------- outputs
def save_pred_mask(path, prob_small, hw_small, full_hw):
    """예측 확률 (h,w) -> 전체 해상도 업샘플 + colormap PNG 저장."""
    import cv2
    h, w = hw_small
    H, W = full_hw
    prob_img = prob_small.reshape(h, w).astype(np.float32)
    prob_full = cv2.resize(prob_img, (W, H), interpolation=cv2.INTER_LINEAR)
    u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)  # BGR
    import imageio.v2 as imageio
    imageio.imwrite(path, color[..., ::-1])          # RGB 로 저장


def save_panel(path, scene, prob_small, y_small, hw_small, left_path=None):
    """pred vs GT side-by-side 패널 (matplotlib Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = hw_small
    prob_img = prob_small.reshape(h, w)
    gt_img = y_small.reshape(h, w)

    ncol = 3 if left_path and os.path.exists(left_path) else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5 * ncol, 5))
    ax = list(np.atleast_1d(axes))
    idx = 0
    if ncol == 3:
        import imageio.v2 as imageio
        img = imageio.imread(left_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        ax[idx].imshow(img[..., :3]); ax[idx].set_title("left"); ax[idx].axis("off")
        idx += 1
    im0 = ax[idx].imshow(prob_img, cmap="jet", vmin=0, vmax=1)
    ax[idx].set_title("pred mirror prob"); ax[idx].axis("off")
    fig.colorbar(im0, ax=ax[idx], fraction=0.046)
    idx += 1
    ax[idx].imshow(gt_img, cmap="gray", vmin=0, vmax=1)
    ax[idx].set_title("GT mirror mask"); ax[idx].axis("off")

    fig.suptitle(f"submodel head — {scene}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _full_hw_for(scene, hw_small, out_root):
    """scale_to_full 로 전체 해상도 추정 (없으면 small 그대로)."""
    h, w = hw_small
    feat_path = os.path.join(out_root, scene, "fs", "features.npz")
    try:
        z = np.load(feat_path)
        s = float(z["scale_to_full"]) if "scale_to_full" in z.files else 1.0
    except Exception:
        s = 1.0
    return (int(round(h * s)), int(round(w * s)))


# --------------------------------------------------------------------- runs
def run_single(args, device):
    """단일 scene: 픽셀 70/30 분할 train/val 데모."""
    data = build_features(args.scene, out_root=args.out_root)
    if data is None:
        raise SystemExit(f"{args.scene}: features.npz/mask 없음 — 먼저 run_foundation_stereo.py 실행")

    X, y = data["X"], data["y"]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    n_tr = int(round(0.70 * len(perm)))
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]

    mean, std = standardize_fit(X[tr_idx])
    Xs = standardize_apply(X, mean, std)

    head = train_head(Xs[tr_idx], y[tr_idx], in_dim=X.shape[1],
                      linear=args.linear, hidden=args.hidden, epochs=args.epochs,
                      lr=args.lr, device=device, seed=args.seed)

    p_va = predict_proba(head, Xs[va_idx], device=device)
    metrics = _eval_block(y[va_idx], p_va)
    print(f"\n[VAL] scene={args.scene}  (pixel 70/30 split)")
    _print_metrics(metrics)

    # 전체 픽셀 예측으로 시각화 저장.
    p_all = predict_proba(head, Xs, device=device)
    config = dict(mode="single", scene=args.scene, linear=bool(args.linear),
                  hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed,
                  n_train=int(len(tr_idx)), n_val=int(len(va_idx)),
                  n_pos=int(y.sum()), n_total=int(len(y)))
    _save_outputs(args, head, mean, std, data, p_all, metrics, config)


def run_multi(args, device):
    """다중 scene: train scene union 학습, held-out val scene 평가."""
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    val_scene = args.val_scene
    if val_scene not in scenes:
        raise SystemExit(f"--val_scene {val_scene} 가 --scenes 목록에 없음")
    train_scenes = [s for s in scenes if s != val_scene]
    if not train_scenes:
        raise SystemExit("학습용 scene 이 0개 — val_scene 외 scene 이 필요")

    # train 데이터 누적 (feature 차원 일치 확인).
    blocks, feat_names, F = [], None, None
    for s in train_scenes:
        d = build_features(s, out_root=args.out_root)
        if d is None:
            continue
        if F is None:
            F, feat_names = d["X"].shape[1], d["feature_names"]
        elif d["X"].shape[1] != F:
            print(f"[skip] {s}: feature 차원 {d['X'].shape[1]} != {F} (feat_left 유무 불일치)")
            continue
        blocks.append((d["X"], d["y"]))
    if not blocks:
        raise SystemExit("유효한 train scene 이 없음")

    Xtr = np.concatenate([b[0] for b in blocks], axis=0)
    ytr = np.concatenate([b[1] for b in blocks], axis=0)

    dva = build_features(val_scene, out_root=args.out_root)
    if dva is None:
        raise SystemExit(f"val_scene {val_scene}: features.npz/mask 없음")
    if dva["X"].shape[1] != F:
        raise SystemExit(f"val_scene feature 차원 {dva['X'].shape[1]} != train {F}")

    mean, std = standardize_fit(Xtr)
    Xtr_s = standardize_apply(Xtr, mean, std)
    Xva_s = standardize_apply(dva["X"], mean, std)

    head = train_head(Xtr_s, ytr, in_dim=F, linear=args.linear, hidden=args.hidden,
                      epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    p_va = predict_proba(head, Xva_s, device=device)
    metrics = _eval_block(dva["y"], p_va)
    print(f"\n[VAL] held-out scene={val_scene}  train={train_scenes}")
    _print_metrics(metrics)

    config = dict(mode="multi", train_scenes=train_scenes, val_scene=val_scene,
                  linear=bool(args.linear), hidden=args.hidden, epochs=args.epochs,
                  lr=args.lr, seed=args.seed, n_train=int(Xtr.shape[0]),
                  n_val=int(dva["X"].shape[0]), feature_dim=int(F))
    # 출력은 val_scene 기준으로 저장.
    args_out_scene = val_scene
    _save_outputs(args, head, mean, std, dva, p_va, metrics, config,
                  scene_for_out=args_out_scene)


def _eval_block(y, p):
    m = binary_metrics(y, p, thr=0.5)
    m["auc"] = roc_auc(y, p)
    return m


def _print_metrics(m):
    print(f"  AUC      = {m['auc']:.4f}")
    print(f"  IoU@0.5  = {m['iou']:.4f}")
    print(f"  F1@0.5   = {m['f1']:.4f}")
    print(f"  accuracy = {m['accuracy']:.4f}  (prec={m['precision']:.4f}, rec={m['recall']:.4f})")
    print(f"  tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")


def _save_outputs(args, head, mean, std, data, p_all, metrics, config,
                  scene_for_out=None):
    """pred_mask.png / panel.png / metrics.json / head.pt 저장."""
    import torch

    scene = scene_for_out or args.scene
    out_dir = args.out or os.path.join(args.out_root, scene, "submodel")
    C.ensure_dir(out_dir)

    hw = data["hw"]
    full_hw = _full_hw_for(scene, hw, args.out_root)
    left_path = os.path.join(args.out_root, scene, "input", "left.png")

    save_pred_mask(os.path.join(out_dir, "pred_mask.png"), p_all, hw, full_hw)
    save_panel(os.path.join(out_dir, "panel.png"), scene, p_all, data["y"], hw, left_path)

    metrics_out = dict(metrics=metrics, config=config,
                       feature_names=data["feature_names"],
                       feature_dim=int(len(data["feature_names"])),
                       feature_res=[int(hw[0]), int(hw[1])])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)

    torch.save(dict(state_dict=head.state_dict(),
                    mean=mean, std=std,
                    linear=bool(args.linear), hidden=int(args.hidden),
                    in_dim=int(len(data["feature_names"])),
                    feature_names=data["feature_names"]),
               os.path.join(out_dir, "head.pt"))

    print(f"\n[save] {out_dir}")
    print("  pred_mask.png, panel.png, metrics.json, head.pt")
    print("DONE:", out_dir)


# --------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="FS prior 특징 위 경량 거울 검출 헤드 (제안서 #3)")
    ap.add_argument("--scene", default=None, help="단일 scene 데모 (experiments/<scene>)")
    ap.add_argument("--scenes", default=None, help="다중 scene 쉼표목록 (a,b,c)")
    ap.add_argument("--val_scene", default=None, help="다중 scene 모드의 held-out val scene")
    ap.add_argument("--linear", action="store_true", help="MLP 대신 순수 로지스틱 회귀 baseline")
    ap.add_argument("--hidden", type=int, default=64, help="MLP hidden 차원")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="출력 폴더 (기본 experiments/<scene>/submodel)")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    ap.add_argument("--cpu", action="store_true", help="GPU 가 있어도 CPU 강제")
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[device] {device}")

    if args.scenes:
        if not args.val_scene:
            raise SystemExit("--scenes 사용 시 --val_scene 필요")
        run_multi(args, device)
    elif args.scene:
        run_single(args, device)
    else:
        raise SystemExit("--scene 또는 --scenes/--val_scene 중 하나를 지정")


if __name__ == "__main__":
    main()
