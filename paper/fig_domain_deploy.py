#!/usr/bin/env python3
"""Identification domain vs where the deployed CPG actually stroked.

Open-loop runs are stationary -- one commanded (A,f) each, R2=0.99, 100 % of
windows sinusoid-like -- so they are POINTS, and their hull is the identified
region. The deployed policy is not stationary, so it is a CLOUD: one point per
2 s window that passed the R2>=0.6 gate. Both come from the same extractor
(probe_segfit.py), fitted on the fast residual after the slow manoeuvre is split off.
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from paper_style import use_paper_style, size, save, CB, LW
from _segfit import segment_fits, summarise, R2_MIN

from _paths import FIELD as _FIELD, DEPLOY_RUNS as _DEPLOY_RUNS, DATA as _DATA
FIELD = str(_FIELD)
DEPLOY = str(_DEPLOY_RUNS)
OL = ["run_3_1.0Hz_9.0mm","run_4_2.0Hz_9.0mm","run_7_3.0Hz_7.0mm","run_10_4.0Hz_7.0mm",
      "run_11_2.0Hz_12.0mm","run_9_2.0Hz_6.0mm_-3.0mm","run_1_1.0Hz_5.0mm",
      "run_2_2.0Hz_5.0mm","run_5_1.0Hz_7.0mm","run_6_2.0Hz_7.0mm",
      "run_12_2.0Hz_12.0mm_p-1600","run_8_2.0Hz_6.0mm_3.0mm"]
FITSET = set(OL[:6])


def main():
    P, isfit = [], []
    for r in OL:
        d = pd.read_csv(os.path.join(FIELD, r, "processed", "03_field2sim_ready.csv"))
        t = d["Time"].values - d["Time"].values[0]
        s = summarise(segment_fits(d["Ghost_Alpha"].values, t))
        if s.get("kept"):
            P.append([s["A"], s["f"]]); isfit.append(r in FITSET)
    P = np.array(P); isfit = np.array(isfit)

    W = []
    for p in sorted(glob.glob(os.path.join(DEPLOY, "20260520_*"))):
        f = os.path.join(p, "raw", "run.csv")
        if not os.path.exists(f): continue
        d = pd.read_csv(f)
        if len(d) < 300: continue
        t = d["pi_time"].values.astype(float)
        ang = np.unwrap(np.arctan2(d["obs_01"].values.astype(float),
                                   d["obs_00"].values.astype(float)))
        W += [w for w in segment_fits(ang, t) if w["R2"] >= R2_MIN]
    DA = np.array([w["A"] for w in W]); DF = np.array([w["f"] for w in W])

    # simulated policies, same extractor on the MEASURED motor angle
    z = np.load(str(_DATA / "sim_rollouts.npz"), allow_pickle=True)
    SIMC = {}
    for tag in ("cpg", "da"):
        dt = float(z[f"{tag}_dt"]); ws = []
        for m in z[f"{tag}_meas"]:
            tt = np.arange(len(m)) * dt
            ws += [w for w in segment_fits(np.asarray(m, float), tt) if w["R2"] >= R2_MIN]
        SIMC[tag] = (np.array([w["A"] for w in ws]), np.array([w["f"] for w in ws]))
        print(f"  sim {tag}: {len(ws)} windows  A={np.median(SIMC[tag][0]):.3f} "
              f"f={np.median(SIMC[tag][1]):.2f}")

    use_paper_style()
    fig, ax = plt.subplots(figsize=size(1.0, aspect=0.80), layout="constrained")
    A = np.linspace(0.03, 0.80, 200)
    for c in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot(A, c / A, color="0.78", lw=0.5, ls=(0, (4, 3)), zorder=1)

    h = ConvexHull(P); poly = P[np.append(h.vertices, h.vertices[0])]
    ax.fill(poly[:, 0], poly[:, 1], color=CB["blue"], alpha=0.12, zorder=2, lw=0)
    ax.plot(poly[:, 0], poly[:, 1], color=CB["blue"], lw=0.9, zorder=2)

    def cloud(A, F, col, z0, lab):
        ax.scatter(A, F, s=4, color=col, alpha=0.16, lw=0, zorder=z0)
        mA, mF = np.median(A), np.median(F)
        ax.plot(np.percentile(A, [25, 75]), [mF, mF], color=col, lw=1.3, zorder=z0 + 2)
        ax.plot([mA, mA], np.percentile(F, [25, 75]), color=col, lw=1.3, zorder=z0 + 2)
        ax.plot([mA], [mF], "o", color=col, ms=5, mec="white", mew=0.8, zorder=z0 + 3)
        print(f"  {lab}: median A={mA:.3f} f={mF:.2f} A*f={mA*mF:.3f}")

    cloud(SIMC["da"][0], SIMC["da"][1], CB["purple"], 3, "sim DA      ")
    cloud(SIMC["cpg"][0], SIMC["cpg"][1], CB["green"], 5, "sim CPG     ")
    cloud(DA, DF, CB["orange"], 7, "deployed CPG")

    ax.scatter(P[isfit, 0], P[isfit, 1], s=34, color=CB["blue"], edgecolor="black",
               linewidth=0.8, zorder=4)
    ax.scatter(P[~isfit, 0], P[~isfit, 1], s=28, facecolor="white",
               edgecolor=CB["blue"], linewidth=0.9, zorder=4)

    ax.set_xlabel("stroke amplitude $A$  [rad]")
    ax.set_ylabel("stroke frequency $f$  [Hz]")
    ax.set_xlim(0.0, 0.72); ax.set_ylim(0.6, 8.4)
    ax.legend(handles=[
        Line2D([0],[0], marker="o", ls="", mfc=CB["blue"], mec="black", ms=5, label="identification, fitted"),
        Line2D([0],[0], marker="o", ls="", mfc="white", mec=CB["blue"], ms=5, label="identification, validation"),
        Line2D([0],[0], marker="o", ls="", mfc=CB["green"], mec="white", ms=5, label="trajectory generator, sim"),
        Line2D([0],[0], marker="o", ls="", mfc=CB["purple"], mec="white", ms=5, label="direct action, sim"),
        Line2D([0],[0], marker="o", ls="", mfc=CB["orange"], mec="white", ms=5, label="deployed, hardware")],
        loc="upper right")
    save(fig, "fig_domain_deploy", expect="col")
    print(f"  deploy windows plotted: {len(W)}")
    print(f"  identification hull: A {P[:,0].min():.3f}-{P[:,0].max():.3f}  f {P[:,1].min():.2f}-{P[:,1].max():.2f}")


if __name__ == "__main__":
    main()
