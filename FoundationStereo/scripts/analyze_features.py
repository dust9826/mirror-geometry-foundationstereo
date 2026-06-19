"""
analyze_features.py — AHCF 전/후 prior 텐서 분석 (거울 신호 검출).

목적:
  run_foundation_stereo.py 가 저장한 features.npz(AHCF 전/후 cost-volume 통계 +
  side-tuning feature)를 GT 거울 마스크와 대조해, 어떤 신호가 거울 영역을 가장 잘
  분리하는지 정량 평가한다.

답하는 RQ (제안서):
  FoundationStereo 의 AHCF(Attentive Hybrid Cost Filtering = corr_feature_att +
  cost_agg) 단계 전후 특징을 비교했을 때 "거울 마스크 신호"가 생기는가?
  - 거울은 photometric consistency 를 깨므로 stereo matching 이 불안정 ->
    낮은 confidence / 높은 entropy / 불안정한 disparity 가 기대된다.
  - AHCF 가 분포를 거울 안/밖에서 다르게 sharpen 하는지 (delta_* 맵)를 본다.
  - side-tuning feature(feat_left)가 이미 거울성을 인코딩하는지 (norm/PCA)를 본다.

각 신호를 per-pixel score 로 보고 GT 마스크에 대한 ROC AUC / 분리도(separability)로
순위를 매겨, "거울을 가장 잘 검출하는 신호와 그 AUC" 를 report.json 에 기록한다.
가장 잘 분리한 신호는 mirror_signal.npy (full-res) 로 저장해 약한 거울 prior 로 재사용한다.

사용:
  python scripts/analyze_features.py --scene cozy_living_room_baseline080
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import common
import common as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------- AUC
def roc_auc(score, label):
    """score(1d float), label(1d bool: True=거울) -> ROC AUC.

    sklearn 이 있으면 사용, 없으면 rank 통계 기반 numpy AUC 로 폴백.
    label 이 한 클래스뿐이면 nan.
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label).ravel().astype(bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(label.astype(np.int32), score))
    except Exception:
        # Mann-Whitney U 기반: AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        order = np.argsort(score, kind="mergesort")
        ranks = np.empty(len(score), dtype=np.float64)
        sorted_score = score[order]
        # 동점 평균 순위 처리
        i = 0
        n = len(score)
        r = np.arange(1, n + 1, dtype=np.float64)
        while i < n:
            j = i
            while j + 1 < n and sorted_score[j + 1] == sorted_score[i]:
                j += 1
            ranks[order[i:j + 1]] = r[i:j + 1].mean()
            i = j + 1
        sum_ranks_pos = ranks[label].sum()
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def directional_auc(score, label):
    """방향(polarity) 무관 AUC: 원본/부호반전 중 큰 쪽을 택해 (auc, polarity) 반환.

    polarity '+' = 거울에서 값이 큼, '-' = 거울에서 값이 작음(부호반전이 더 분리).
    """
    a = roc_auc(score, label)
    if not np.isfinite(a):
        return float("nan"), "+"
    if a >= 0.5:
        return a, "+"
    return 1.0 - a, "-"


def separability(score, label):
    """|mean_in - mean_out| / (std_in + std_out + eps) 와 mean_in, mean_out 반환."""
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label).ravel().astype(bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    inside = score[label]
    outside = score[~label]
    if inside.size == 0 or outside.size == 0:
        return float("nan"), float("nan"), float("nan")
    mi, mo = float(inside.mean()), float(outside.mean())
    sep = abs(mi - mo) / (float(inside.std()) + float(outside.std()) + 1e-6)
    return sep, mi, mo


# --------------------------------------------------------------------- helpers
def resize_nearest(mask_bool, hw):
    """bool 마스크를 (h,w) 로 nearest 다운샘플."""
    h, w = hw
    try:
        import cv2
        m = cv2.resize(mask_bool.astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)
    except Exception:
        H, W = mask_bool.shape[:2]
        ys = (np.linspace(0, H - 1, h)).round().astype(int)
        xs = (np.linspace(0, W - 1, w)).round().astype(int)
        return mask_bool[np.ix_(ys, xs)]


def resize_bilinear(arr, hw):
    """float 맵을 (H,W) 로 bilinear 업샘플 (full-res 재사용용)."""
    H, W = hw
    try:
        import cv2
        return cv2.resize(arr.astype(np.float32), (W, H),
                          interpolation=cv2.INTER_LINEAR)
    except Exception:
        h, w = arr.shape[:2]
        ys = np.linspace(0, h - 1, H).round().astype(int)
        xs = np.linspace(0, w - 1, W).round().astype(int)
        return arr[np.ix_(ys, xs)]


def norm01(a):
    """디스플레이용 [0,1] 정규화 (유한값 기준)."""
    a = np.asarray(a, dtype=np.float32)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(fin, 1), np.percentile(fin, 99)
    if hi <= lo:
        lo, hi = float(fin.min()), float(fin.max())
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def pca_maps(feat_chw, n_comp=3):
    """feat_left (C,h,w) -> 픽셀 PCA. (l2_norm(h,w), [PC1,PC2,PC3](h,w) 리스트) 반환."""
    C_, h, w = feat_chw.shape
    X = feat_chw.reshape(C_, h * w).T.astype(np.float64)  # (N, C)
    l2 = np.sqrt((X ** 2).sum(axis=1)).reshape(h, w).astype(np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        # 경제적 SVD: components = Vt
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(n_comp, Vt.shape[0])
        proj = Xc @ Vt[:k].T  # (N, k)
        comps = [proj[:, i].reshape(h, w).astype(np.float32) for i in range(k)]
    except Exception:
        comps = []
    return l2, comps


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="AHCF 전/후 특징의 거울 신호 분석")
    ap.add_argument("--scene", required=True, help="experiments/<scene>")
    ap.add_argument("--out_root", default=C.EXPERIMENTS_DIR)
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.scene)
    fs_dir = os.path.join(run_dir, "fs")
    feat_dir = C.ensure_dir(os.path.join(run_dir, "features"))

    npz_path = os.path.join(fs_dir, "features.npz")
    if not os.path.isfile(npz_path):
        raise SystemExit(f"features.npz 없음: {npz_path}\n"
                         f"먼저: python scripts/run_foundation_stereo.py --scene {args.scene}")
    npz = np.load(npz_path)

    pre_conf = npz["pre_conf"].astype(np.float32)
    post_conf = npz["post_conf"].astype(np.float32)
    pre_ent = npz["pre_entropy"].astype(np.float32)
    post_ent = npz["post_entropy"].astype(np.float32)
    pre_arg = npz["pre_argdisp"].astype(np.float32)
    post_arg = npz["post_argdisp"].astype(np.float32)
    scale_to_full = float(npz["scale_to_full"]) if "scale_to_full" in npz.files else None
    feat_left = npz["feat_left"].astype(np.float32) if "feat_left" in npz.files else None

    h, w = post_conf.shape[:2]

    # --- AHCF 효과 맵 (feature 해상도) ---
    delta_conf = post_conf - pre_conf            # AHCF 후 신뢰도 변화
    delta_entropy = post_ent - pre_ent           # 엔트로피 변화
    delta_argdisp = np.abs(post_arg - pre_arg)   # argmax disparity 이동량

    # --- GT 거울 마스크 (feature 해상도로 nearest 다운샘플) ---
    mask_path = os.path.join(run_dir, "input", "mirror_mask.png")
    have_mask = os.path.isfile(mask_path)
    if have_mask:
        mask_full = C.load_mask(mask_path)
        mask = resize_nearest(mask_full, (h, w))
    else:
        mask_full = None
        mask = np.zeros((h, w), dtype=bool)
        print("WARN: mirror_mask.png 없음 -> AUC=nan (정성 패널만 생성)")

    # --- 분석 대상 신호 모음 ---
    # 기대 방향: conf 는 거울 밖에서 큼(-), entropy 는 거울에서 큼(+),
    # delta_*/argdisp 은 데이터로 판단(directional_auc 가 자동 처리).
    signals = {
        "pre_conf": pre_conf,
        "post_conf": post_conf,
        "pre_entropy": pre_ent,
        "post_entropy": post_ent,
        "delta_conf": delta_conf,
        "delta_entropy": delta_entropy,
        "delta_argdisp": delta_argdisp,
    }

    feat_l2 = None
    feat_pcs = []
    if feat_left is not None:
        feat_l2, feat_pcs = pca_maps(feat_left, n_comp=3)
        signals["feat_left_l2"] = feat_l2
        for i, pc in enumerate(feat_pcs, start=1):
            signals[f"feat_left_pc{i}"] = np.abs(pc)  # |PC| 를 score 로

    # --- 정량 평가 ---
    rows = []
    for name, sig in signals.items():
        auc, pol = directional_auc(sig, mask)
        sep, mi, mo = separability(sig, mask)
        rows.append(dict(
            signal=name,
            auc=None if not np.isfinite(auc) else round(float(auc), 4),
            polarity=pol,
            separability=None if not np.isfinite(sep) else round(float(sep), 4),
            mean_inside=None if not np.isfinite(mi) else round(float(mi), 6),
            mean_outside=None if not np.isfinite(mo) else round(float(mo), 6),
        ))

    # AUC 내림차순 정렬 (nan 은 뒤로)
    def _auc_key(r):
        return r["auc"] if r["auc"] is not None else -1.0
    rows.sort(key=_auc_key, reverse=True)

    best = rows[0] if rows and rows[0]["auc"] is not None else None

    # --- report.json ---
    report = dict(
        scene=args.scene,
        feature_hw=[int(h), int(w)],
        scale_to_full=scale_to_full,
        have_mask=bool(have_mask),
        mirror_frac=float(mask.mean()) if have_mask else None,
        has_feat_left=feat_left is not None,
        feat_left_channels=int(feat_left.shape[0]) if feat_left is not None else None,
        ranked=rows,
        best_signal=best["signal"] if best else None,
        best_auc=best["auc"] if best else None,
        best_polarity=best["polarity"] if best else None,
        note=("AUC>0.5 이면 거울 분리 신호 존재. polarity '+' = 거울에서 값 큼, "
              "'-' = 거울에서 값 작음(부호반전이 더 분리). best_signal 이 "
              "AHCF 전(pre_*)보다 후(post_*)/delta_* 면 AHCF 가 거울 신호를 강화한 것."),
    )
    with open(os.path.join(feat_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- mirror_signal.npy: 최고 분리 신호를 full-res 로 업샘플 ---
    # 사용할 score 의 polarity 가 '-' 면 부호 반전(항상 "거울=큰값" 방향으로).
    if best is not None:
        best_name = best["signal"]
        best_map = signals[best_name].astype(np.float32)
        if best["polarity"] == "-":
            best_map = -best_map
        if scale_to_full is not None:
            H, W = int(round(h * scale_to_full)), int(round(w * scale_to_full))
        else:
            H, W = h, w
        if mask_full is not None:  # full-res 마스크 크기에 맞춤(있으면 가장 정확)
            H, W = mask_full.shape[:2]
        mirror_signal = resize_bilinear(best_map, (H, W))
        np.save(os.path.join(feat_dir, "mirror_signal.npy"),
                mirror_signal.astype(np.float32))
    else:
        # 마스크 없으면 post_entropy 를 기본 prior 로 저장
        H, W = (mask_full.shape[:2] if mask_full is not None
                else (int(round(h * (scale_to_full or 1))), int(round(w * (scale_to_full or 1)))))
        np.save(os.path.join(feat_dir, "mirror_signal.npy"),
                resize_bilinear(post_ent, (H, W)).astype(np.float32))

    # --- panels.png ---
    panels = []
    if have_mask:
        panels.append(("GT mirror mask", mask.astype(np.float32), None))
    panel_specs = [
        ("pre_conf", pre_conf), ("post_conf", post_conf),
        ("delta_conf", delta_conf),
        ("pre_entropy", pre_ent), ("post_entropy", post_ent),
        ("delta_entropy", delta_entropy),
        ("delta_argdisp", delta_argdisp),
    ]
    if feat_l2 is not None:
        panel_specs.append(("feat_left_l2", feat_l2))
    if feat_pcs:
        panel_specs.append(("feat_left_pc1", feat_pcs[0]))

    auc_lookup = {r["signal"]: r for r in rows}
    for name, arr in panel_specs:
        r = auc_lookup.get(name)
        if r is None and name == "feat_left_pc1":
            r = auc_lookup.get("feat_left_pc1")
        title = name
        if r is not None and r["auc"] is not None:
            title = f"{name}\nAUC={r['auc']:.3f} ({r['polarity']})"
        panels.append((title, arr, None))

    n = len(panels)
    cols = 4
    nrows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, arr, _) in zip(axes, panels):
        im = ax.imshow(norm01(arr), cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(panels):]:
        ax.axis("off")
    sup = f"{args.scene}  |  AHCF pre/post mirror-signal analysis"
    if best:
        sup += f"  |  best={best['signal']} AUC={best['auc']}"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(feat_dir, "panels.png"), dpi=110)
    plt.close(fig)

    # --- 콘솔 요약 ---
    print(f"[{args.scene}] feature {h}x{w}  mirror_frac="
          f"{report['mirror_frac']}  feat_left={feat_left is not None}")
    print("rank  signal              AUC    pol  sep     mean_in     mean_out")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {r['signal']:<18} {str(r['auc']):>6}  {r['polarity']:>2}  "
              f"{str(r['separability']):>6}  {str(r['mean_inside']):>10}  "
              f"{str(r['mean_outside']):>10}")
    if best:
        print(f"BEST: {best['signal']} (AUC={best['auc']}, polarity={best['polarity']})")
    print("저장:", os.path.join(feat_dir, "report.json"),
          "/ panels.png / mirror_signal.npy")


if __name__ == "__main__":
    main()
