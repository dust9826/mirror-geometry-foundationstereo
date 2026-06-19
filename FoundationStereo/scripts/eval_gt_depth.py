"""eval_gt_depth.py — FoundationStereo 의 disparity/depth 를 GT 와 비교 (거울 vs 비거울).

ds_v1(Archive) 뷰는 input/ 에 GT(disp_gt.npy, depth_gt.npy)와 거울 마스크가 있다.
처리된 모든 뷰(fs/disp.npy + input/disp_gt.npy 존재)를 스캔해 픽셀 오차를 집계한다.

핵심 가설 검증: FS 는 정상면에선 정확하지만 거울에선 (반사 깊이를 보므로) 크게 틀린다.

지표(거울/비거울 각각): disp MAE, bad@1px/3px(%), depth MAE(m), AbsRel.

출력(--root 아래): _eval_gt_depth.csv (뷰별), _eval_gt_depth.json (집계), _eval_gt_depth.png (요약)

사용:
  python scripts/eval_gt_depth.py --root experiments/archive_views
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C


def _load_mask(p):
    import imageio.v2 as imageio
    m = imageio.imread(p)
    m = m if m.ndim == 2 else m[..., :3].mean(axis=-1)
    return m > 127


def eval_view(run_dir):
    """한 뷰의 거울/비거울 오차 dict 반환 (필수 파일 없으면 None)."""
    fs, inp = os.path.join(run_dir, "fs"), os.path.join(run_dir, "input")
    need = [os.path.join(fs, "disp.npy"), os.path.join(fs, "depth_meter.npy"),
            os.path.join(inp, "disp_gt.npy"), os.path.join(inp, "depth_gt.npy"),
            os.path.join(inp, "mirror_mask.png")]
    if not all(os.path.isfile(p) for p in need):
        return None
    disp = np.load(need[0]).astype(np.float64)
    depth = np.load(need[1]).astype(np.float64)
    dg = np.load(need[2]).astype(np.float64)
    depg = np.load(need[3]).astype(np.float64)
    mir = _load_mask(need[4])
    if disp.shape != dg.shape:        # 해상도 불일치 방어
        return None

    valid = np.isfinite(dg) & (dg > 0) & np.isfinite(depg) & (depg > 0) & (disp > 0)
    out = {}
    for region, sel in [("nonmirror", valid & ~mir), ("mirror", valid & mir)]:
        n = int(sel.sum())
        if n == 0:
            out[region] = dict(n=0)
            continue
        de = np.abs(disp[sel] - dg[sel])
        ze = np.abs(depth[sel] - depg[sel])
        out[region] = dict(
            n=n,
            disp_mae=float(de.mean()),
            bad1=float((de > 1).mean() * 100),
            bad3=float((de > 3).mean() * 100),
            depth_mae=float(ze.mean()),
            absrel=float((ze / depg[sel]).mean()),
        )
    return out


def _agg(rows, region, key):
    """뷰별 결과를 픽셀수 가중 평균 (집계)."""
    num = den = 0.0
    for r in rows:
        d = r[region]
        if d.get("n", 0) and key in d:
            num += d[key] * d["n"]
            den += d["n"]
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(C.EXPERIMENTS_DIR, "archive_views"),
                    help="뷰 폴더들이 들어있는 루트")
    args = ap.parse_args()

    view_dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                       if os.path.isdir(d))
    rows = []
    for vd in view_dirs:
        r = eval_view(vd)
        if r is None:
            continue
        r["view"] = os.path.basename(vd)
        r["scene"] = r["view"].rsplit("_v", 1)[0]
        rows.append(r)

    if not rows:
        raise SystemExit(f"평가할 뷰 없음(fs/disp.npy + input/disp_gt.npy 필요): {args.root}")

    # --- CSV (뷰별) ---
    csv_path = os.path.join(args.root, "_eval_gt_depth.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "view", "region", "n", "disp_mae", "bad1px", "bad3px",
                    "depth_mae_m", "absrel"])
        for r in rows:
            for reg in ("nonmirror", "mirror"):
                d = r[reg]
                if d.get("n", 0):
                    w.writerow([r["scene"], r["view"], reg, d["n"],
                                round(d["disp_mae"], 4), round(d["bad1"], 2),
                                round(d["bad3"], 2), round(d["depth_mae"], 4),
                                round(d["absrel"], 4)])

    # --- 집계 (전체 + scene별) ---
    def block(subset):
        return {reg: {k: _agg(subset, reg, k)
                      for k in ("disp_mae", "bad1", "bad3", "depth_mae", "absrel")}
                for reg in ("nonmirror", "mirror")}

    scenes = sorted(set(r["scene"] for r in rows))
    agg = {"n_views": len(rows), "overall": block(rows),
           "per_scene": {sc: block([r for r in rows if r["scene"] == sc]) for sc in scenes}}
    json_path = os.path.join(args.root, "_eval_gt_depth.json")
    json.dump(agg, open(json_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # --- 요약 그래프 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = scenes
        nm = [agg["per_scene"][s]["nonmirror"]["disp_mae"] for s in labels]
        mr = [agg["per_scene"][s]["mirror"]["disp_mae"] for s in labels]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4.5))
        ax.bar(x - 0.2, nm, 0.4, label="non-mirror", color="#1a7f37")
        ax.bar(x + 0.2, mr, 0.4, label="mirror", color="#b3261e")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("disparity MAE (px)")
        ax.set_title("FS disparity error vs GT — mirror breaks stereo")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.root, "_eval_gt_depth.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print("그래프 건너뜀:", e)

    # --- 콘솔 요약 ---
    print(f"\n평가 뷰 수: {len(rows)}  (scene {len(scenes)}개)")
    # robust: 뷰별 disp_MAE 의 중앙값 (픽셀가중 평균은 발산 뷰에 오염되므로 중앙값을 헤드라인으로)
    def _med(reg, k):
        vals = [r[reg][k] for r in rows if r[reg].get("n", 0) and k in r[reg]]
        return float(np.median(vals)) if vals else float("nan")
    print("[robust] 뷰별 중앙값 — disp_MAE / depth_MAE:")
    for reg in ("nonmirror", "mirror"):
        print(f"  {reg:10s} disp {_med(reg,'disp_mae'):6.2f} px   depth {_med(reg,'depth_mae'):.3f} m")
    bad = [r["view"] for r in rows if r["nonmirror"].get("disp_mae", 0) > 5]
    if bad:
        print(f"[주의] 비거울 disp_MAE>5px (FS 발산/데이터 이상) {len(bad)}뷰: {bad[:8]}{'...' if len(bad)>8 else ''}")
    print("--- (참고) 픽셀가중 평균 ---")
    o = agg["overall"]
    print("region      disp_MAE  bad1px  bad3px  depth_MAE(m)  AbsRel")
    for reg in ("nonmirror", "mirror"):
        d = o[reg]
        print(f"  {reg:10s} {d['disp_mae']:7.2f}  {d['bad1']:5.1f}%  {d['bad3']:5.1f}%  "
              f"{d['depth_mae']:9.3f}    {d['absrel']:.3f}")
    rat = o["mirror"]["disp_mae"] / max(o["nonmirror"]["disp_mae"], 1e-9)
    print(f"  → 거울 disparity 오차가 비거울의 {rat:.1f}배")
    print("저장:", csv_path, "/ _eval_gt_depth.json / _eval_gt_depth.png")


if __name__ == "__main__":
    main()
