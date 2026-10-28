#!/usr/bin/env python3
"""Deployment collage: six runs, one figure.

Replaces the pair of full-width figures that showed the same six runs twice, once as pool
composites and once as plotted trajectories. The composites already carry the swum path drawn
over the photograph; what they lacked were the numbers, so the start-to-final distance and the
duration are ported across from the trajectory figure and annotated onto each panel.

Six runs, laid out 2x3 so the figure is no taller than the montage it replaces:
    (a) target reaching      (b) disturbance rejection  (c) backward-velocity start
    (d) out-of-distribution  (e) lateral-rear failure   (f) moving-target pursuit

The distances are computed with the same code path as the trajectory figure, by importing its
helpers, so the numbers cannot drift from the ones that figure used to report.

Inputs
    upstream/.../analysis_20260520/report_strobes/   the composites
    upstream/.../analysis_20260520/cache/            vision caches, for the distances
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import ANALYSIS as _ANALYSIS
from paper_style import use_paper_style, size, save, CB

ANA = str(_ANALYSIS)
SRC = os.path.join(ANA, "report_strobes")
sys.path.insert(0, ANA)
import plot_nav_collage as NAV          # reuse load_run / get_scale / to_m / despike

# run key, strobe file, caption, truncate at closest approach, moving target
PANELS = [
    ("hope_didnt_work_2", "hope_didnt_work_2_highres_traceover_strobe.png",
     "(a) Target reaching", True, False),
    ("insane_run_8", "insane_run_8_middle_bg_highres_traceover_strobe.png",
     "(b) Fluid disturbance", True, False),
    ("insane_run_2", "insane_run_2_highres_traceover_strobe.png",
     "(c) Backward-velocity start", True, False),
    ("finetune_run_7", "finetune_run_7_highres_traceover_strobe.png",
     "(d) Out-of-distribution target", True, False),
    ("finetune_run_8", "finetune_run_8_highres_traceover_strobe.png",
     "(e) Lateral-rear failure", False, False),
    ("figure_eight_6", "figure_eight_6_clean_highres_traceover_strobe.png",
     "(f) Moving-target pursuit", False, True),
]


def distances(run, truncate, moving, scale):
    """start distance, final distance and duration — same computation as the trajectory figure."""
    d = NAV.load_run(run)
    fp, tp, ft = d["fish_px"], d["tgt_px"], d["frame_time"]
    fwld, twld = d["fish_world"].astype(float), d["tgt_world"].astype(float)
    world_ok = (np.sum(~np.isnan(fwld[:, 0])) > 0) and (np.sum(~np.isnan(twld[:, 0])) > 0)
    fm, tm = (fwld.copy(), twld.copy()) if world_ok else (NAV.to_m(fp, scale), NAV.to_m(tp, scale))
    vf = ~np.isnan(fm[:, 0]) & ~np.isnan(fm[:, 1])
    vt = ~np.isnan(tm[:, 0]) & ~np.isnan(tm[:, 1])
    if world_ok:
        vf = NAV.despike(fm, vf)
    fm[:, 1] *= -1.0
    tm[:, 1] *= -1.0
    g0 = fm[np.where(vf)[0][0]].copy()
    fm, tm = fm - g0, tm - g0
    if moving:
        both = vf & vt
        dist = np.full(len(fm), np.nan)
        dist[both] = np.hypot(fm[both, 0] - tm[both, 0], fm[both, 1] - tm[both, 1])
    else:
        tref = np.median(tm[vt], axis=0)
        dist = np.where(vf, np.hypot(fm[:, 0] - tref[0], fm[:, 1] - tref[1]), np.nan)
    idx = np.where(vf)[0]
    dvi = np.where(~np.isnan(dist))[0]
    if truncate:
        ci = int(np.nanargmin(dist))
        end = int(idx[np.argmin(np.abs(idx - ci))]) + 1
    else:
        end = len(fm)
    m = vf.copy(); m[end:] = False
    plotted = np.where(m)[0]
    return float(dist[dvi[0]]), float(dist[int(plotted[-1])]), float(ft[plotted[-1]] - ft[plotted[0]])


use_paper_style()
plt.rcParams.update({"axes.grid": False})
scale = NAV.get_scale()

fig, axes = plt.subplots(2, 3, figsize=size(1.0, aspect=0.437, full=True), layout="constrained")
for ax, (run, fn, title, trunc, moving) in zip(axes.flat, PANELS):
    ax.imshow(plt.imread(os.path.join(SRC, fn)))
    ax.set_title(title, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    d0, d1, ts = distances(run, trunc, moving, scale)
    note = (f"{ts:.0f}\\,s, tracking" if moving
            else f"{d0:.2f}\\,m $\\rightarrow$ {d1:.3f}\\,m, {ts:.0f}\\,s")
    ax.text(0.975, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
            color="white", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.22", fc="black", ec="none", alpha=0.55))
    print(f"  {run:22s} {note}")

fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.01, hspace=0.02)
save(fig, "fig_deploy_collage", expect="full")
