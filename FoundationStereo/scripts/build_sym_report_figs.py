"""build_sym_report_figs.py — 반사대칭 평면(정공법) 보고서 figure 생성.

report/figures_seeds/ 에:
  sym_scatter.png       est perp vs GT perp 산점도 + 법선각 CDF
  sym_topdown_good.png  성공 예 top-down (방/가상점/반사점/GT/추정 평면)
  sym_topdown_bad.png   실패 예 (scandinavian d-슬라이딩)
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import common as C
import estimate_mirror_plane_sym as SY

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

EXP = C.EXPERIMENTS_DIR
OUT = C.ensure_dir(os.path.join(C.PROJECT_ROOT, "report", "figures_seeds"))

rows = list(csv.DictReader(open(os.path.join(EXP, "_eval_sym.csv"), encoding="utf-8")))
rows = [r for r in rows if r["angle_deg"]]

# ---- 1) 산점도 + 각도 CDF ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
scenes = sorted({r["scene"] for r in rows})
cmap = plt.cm.tab20(np.linspace(0, 1, len(scenes)))
for sc, col in zip(scenes, cmap):
    rs = [r for r in rows if r["scene"] == sc]
    ax1.scatter([float(r["gt_perp"]) for r in rs], [float(r["perp"]) for r in rs],
                s=7, alpha=0.55, color=col, label=sc)
lim = [0, 9]
ax1.plot(lim, lim, "k--", lw=1)
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("GT 거울 수직거리 (m)"); ax1.set_ylabel("추정 수직거리 (m)")
ax1.set_title("반사대칭 평면: 추정 vs GT perp (1,536뷰)")
ax1.legend(fontsize=6.5, ncol=2, loc="upper left")

ang = np.sort(np.array([float(r["angle_deg"]) for r in rows]))
ax2.plot(ang, np.arange(1, len(ang) + 1) / len(ang) * 100, lw=2)
for x in (5, 10):
    ax2.axvline(x, color="gray", ls=":", lw=1)
ax2.set_xlim(0, 45); ax2.set_ylim(0, 100)
ax2.set_xlabel("법선 각도 오차 (°)"); ax2.set_ylabel("누적 비율 (%)")
ax2.set_title(f"법선각 CDF — 중앙값 {np.median(ang):.1f}°, <10° {100*(ang<10).mean():.0f}%")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sym_scatter.png"), dpi=130)
plt.close(fig)
print("[fig] sym_scatter.png")


# ---- 2) top-down 예시 ----
def topdown(scene, out_name, title):
    run_dir = os.path.join(EXP, scene)
    data = SY.load_scene(run_dir, rng=np.random.default_rng(0))
    P, Q = data
    res = json.load(open(os.path.join(run_dir, "mirror_geometry_sym", "result.json"),
                         encoding="utf-8"))
    n, d = np.asarray(res["normal"]), res["d"]
    R = SY.reflect_points(P, n, d)
    gt = json.load(open(os.path.join(run_dir, "mirror_gt", "gt.json"),
                        encoding="utf-8"))["mirrors"][0]
    ng, dg = np.asarray(gt["normal"]), float(gt["d"])

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(Q[::4, 0], Q[::4, 2], s=2, c="#9aa5b1", label="방 점군 Q")
    ax.scatter(P[::2, 0], P[::2, 2], s=3, c="#ef4444", label="거울 안 가상점 P'")
    ax.scatter(R[::2, 0], R[::2, 2], s=3, c="#16a34a", label="reflect(P') (추정평면)")

    def draw_plane(nv, dv, color, label):
        # 평면과 y=0 단면: n·[x,0,z]+d=0 직선
        xs = np.linspace(-6, 6, 50)
        if abs(nv[2]) > 1e-6:
            zs = -(nv[0] * xs + dv) / nv[2]
            ok = (zs > -1) & (zs < 12)
            ax.plot(xs[ok], zs[ok], color=color, lw=2.2, label=label)
    draw_plane(ng, dg, "#2563eb", f"GT 평면 (perp {gt['perp_distance']:.2f}m)")
    draw_plane(n, d, "#f59e0b", f"추정 평면 (perp {res['perp']:.2f}m)")
    ax.scatter([0], [0], marker="*", s=160, c="k", label="카메라")
    ax.set_xlim(-6, 6); ax.set_ylim(-1, 12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ev = res.get("eval_vs_gt") or {}
    ax.set_title(f"{title}\n{scene} — 각도 {ev.get('angle_deg', 0):.1f}°, "
                 f"perp오차 {ev.get('perp_err_m', 0):.2f}m")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name), dpi=130)
    plt.close(fig)
    print(f"[fig] {out_name}")


topdown("ds_v1_seed0/computer_room_v00", "sym_topdown_good.png",
        "성공 예 — 반사점(초록)이 방 점군(회색)과 정합")

# scandinavian 최악 뷰
sc_rows = [r for r in rows if r["scene"] == "scandinavian"]
worst = max(sc_rows, key=lambda r: float(r["perp_err_m"]))
topdown(worst["view"], "sym_topdown_bad.png",
        "실패 예 — d-슬라이딩 퇴화 (평면 벽 반사)")
print("worst scandinavian:", worst["view"], worst["perp_err_m"])
