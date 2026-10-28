#!/usr/bin/env python3
"""The identification domain in (A, f) — evidence map, not a fence.

Each open-loop run is placed at its ACHIEVED motor shaft amplitude and measured
frequency (RUNS.local.md), and coloured by how badly the twin mispredicts the
distance covered in equal time. That makes the figure carry both halves of the
argument at once: where we have data, AND how good the model was there.

Iso-A*f hyperbolas are the stateless model's own level sets: inside the band its
speed is a function of A*f alone, so along these curves it cannot tell points
apart. The plant's saturation frontier sits at A*f ~ 1.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import use_paper_style, size, save, CB, LW

#  run       A_ach   f    steady-state speed error (signed)  fitted?  turn?
#  from probe_domain_metrics.py: both speeds measured over each run's own
#  50-80 % window, so runs are compared at the same phase of their transient.
RUNS = [
    ("R3",  0.637, 1.00, -0.142, True,  False), ("R4",  0.504, 2.00, -0.126, True,  False),
    ("R7",  0.331, 3.00, -0.140, True,  False), ("R10", 0.231, 4.00, -0.195, True,  False),
    ("R11", 0.520, 2.00, -0.136, True,  False), ("R9",  0.403, 2.00, +0.083, True,  True),
    ("R1",  0.353, 1.00, -0.187, False, False), ("R2",  0.349, 2.00, -0.053, False, False),
    ("R5",  0.500, 1.00, -0.107, False, False), ("R6",  0.460, 2.00, -0.075, False, False),
    ("R12", 0.511, 1.99, -0.079, False, False), ("R8",  0.405, 2.00, +0.067, False, True),
]


OFF = {"R4": (-6, 7), "R11": (6, 3), "R12": (6, -10), "R6": (-17, 4),
       "R9": (-6, 8), "R8": (-4, -12), "R2": (-17, 3), "R5": (5, -11),
       "R1": (-17, -3), "R3": (6, 2), "R7": (6, 3), "R10": (6, 3)}
# R8/R9 are separate runs (turn +3 mm / -3 mm) that landed within 0.002 rad of each
# other; dodged apart purely so both are visible.
DODGE = {"R9": (-0.011, 0.0), "R8": (+0.011, 0.0)}


def main():
    use_paper_style()
    fig, ax = plt.subplots(figsize=size(1.0, aspect=0.80), layout="constrained")

    # iso-A*f level sets of the stateless model
    A = np.linspace(0.12, 0.80, 200)
    for c in (0.4, 0.6, 0.8, 1.0):
        ax.plot(A, c / A, color="0.75", lw=0.5, ls=(0, (4, 3)), zorder=1)
        xi = 0.185 if c / 0.185 < 4.5 else c / 4.35
        yi = c / xi
        ax.text(xi, yi, r"$Af\!=\!%.1f$" % c, color="0.55", fontsize=6,
                ha="left", va="bottom", zorder=1)

    # convex hull of every run we have evidence for -- the sampled region, no more
    from scipy.spatial import ConvexHull
    P = np.array([[r[1], r[2]] for r in RUNS])
    h = ConvexHull(P); poly = P[np.append(h.vertices, h.vertices[0])]
    ax.fill(poly[:, 0], poly[:, 1], color=CB["blue"], alpha=0.10, zorder=2, lw=0)
    ax.plot(poly[:, 0], poly[:, 1], color=CB["blue"], lw=0.8, alpha=0.55, zorder=2)
    sc = None
    for name, a, f, e, fit, turn in RUNS:
        da, df = DODGE.get(name, (0.0, 0.0))
        a, f = a + da, f + df
        sc = ax.scatter([a], [f], c=[e], cmap="RdBu_r", vmin=-0.22, vmax=0.22,
                        s=52 if fit else 44, marker="o" if not turn else "D",
                        edgecolor="black" if fit else "0.45",
                        linewidth=0.9 if fit else 0.5, zorder=4)
        dx, dy = OFF.get(name, (5, 3))
        ax.annotate(name, (a, f), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6, color="0.3", zorder=5)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("steady-state speed error  $(u_\\mathrm{twin}-u_\\mathrm{real})/u_\\mathrm{real}$")
    cb.outline.set_linewidth(0.6)

    ax.set_xlabel("achieved stroke amplitude $A$  [rad]")
    ax.set_ylabel("stroke frequency $f$  [Hz]")
    ax.set_xlim(0.15, 0.72); ax.set_ylim(0.4, 4.6)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", mfc="0.85", mec="black", mew=0.9, ms=5,
               label="fitted"),
        Line2D([0], [0], marker="o", ls="", mfc="0.85", mec="0.45", mew=0.5, ms=5,
               label="validation"),
        Line2D([0], [0], marker="D", ls="", mfc="0.85", mec="0.45", mew=0.5, ms=4.5,
               label="turn")], loc="upper right", ncol=1)
    save(fig, "fig_domain", expect="col")


if __name__ == "__main__":
    main()
