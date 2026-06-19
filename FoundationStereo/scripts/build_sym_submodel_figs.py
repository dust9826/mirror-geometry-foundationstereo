"""build_sym_submodel_figs.py — 예측 mask(end-to-end) 정공법 결과 top-down figure.

report/figures_seeds/sym_submodel/ 에 15뷰 각각 저장.
"""
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
OUT = C.ensure_dir(os.path.join(C.PROJECT_ROOT, "report", "figures_seeds", "sym_submodel"))

VIEWS = [f"ds_v1_seed0/{sc}_{v}"
         for sc in ("computer_room", "livingroom", "minigym")
         for v in ("v00", "v10", "v20", "v30", "v40")]


def topdown(scene, out_name):
    run_dir = os.path.join(EXP, scene)
    P, Q = SY.load_scene(run_dir, mask_src="submodel", rng=np.random.default_rng(0))
    res = json.load(open(os.path.join(run_dir, "mirror_geometry_sym_submodel",
                                      "result.json"), encoding="utf-8"))
    n, d = np.asarray(res["normal"]), res["d"]
    R = SY.reflect_points(P, n, d)
    gt = json.load(open(os.path.join(run_dir, "mirror_gt", "gt.json"),
                        encoding="utf-8"))["mirrors"][0]
    ng, dg = np.asarray(gt["normal"]), float(gt["d"])

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(Q[::4, 0], Q[::4, 2], s=2, c="#9aa5b1", label="방 점군 Q")
    ax.scatter(P[::2, 0], P[::2, 2], s=3, c="#ef4444", label="예측mask 거울점 P'")
    ax.scatter(R[::2, 0], R[::2, 2], s=3, c="#16a34a", label="reflect(P')")

    def draw_plane(nv, dv, color, label):
        xs = np.linspace(-6, 6, 50)
        if abs(nv[2]) > 1e-6:
            zs = -(nv[0] * xs + dv) / nv[2]
            ok = (zs > -1) & (zs < 12)
            ax.plot(xs[ok], zs[ok], color=color, lw=2.2, label=label)
    draw_plane(ng, dg, "#2563eb", f"GT (perp {gt['perp_distance']:.2f}m)")
    draw_plane(n, d, "#f59e0b", f"추정 (perp {res['perp']:.2f}m)")
    ax.scatter([0], [0], marker="*", s=160, c="k", label="카메라")
    ax.set_xlim(-6, 6); ax.set_ylim(-1, 12)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ev = res.get("eval_vs_gt") or {}
    ok = (ev.get("angle_deg", 99) < 10) and (ev.get("perp_err_m", 9) < 0.5)
    ax.set_title(f"[end-to-end 예측mask] {scene}\n각도 {ev.get('angle_deg', 0):.1f}°, "
                 f"perp오차 {ev.get('perp_err_m', 0):.2f}m  {'✓성공' if ok else '✗실패'}")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name), dpi=120)
    plt.close(fig)


for v in VIEWS:
    name = v.replace("/", "__") + ".png"
    try:
        topdown(v, name)
        print("[fig]", name)
    except Exception as e:
        print("[skip]", v, e)
print(f"\nDONE → {OUT}")
