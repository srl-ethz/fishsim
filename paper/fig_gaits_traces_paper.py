#!/usr/bin/env python3
"""Fig: what the direct-action flutter looks like -- panel (b) of the July gait figure, alone.

The July figure (`10_gaits_combined.pdf`) had three panels; (a) and (c) duplicate Table III
and were dropped on the advisor's note. This keeps only the three motor-angle traces:
commanded setpoint (dashed grey) and actual angle, one row per policy, over the first
3 s of the original 6 s window, same +-1 rad scale.

Data: data/gait_traces.npz, the plotted vertices recovered from the vector PDF by
tools/extract_gait_traces.py (the drawing script was never saved and the DA-LPF1 rollouts
are not in this repo). Times sit exactly on the 50 Hz grid; collinear samples that
matplotlib's simplifier dropped do not move the line. Rows are labelled as in Table III:
TG (trajectory generator, called "CPG" in the July figure), DA, DA-LPF1.

Single column, 7 pt type, >= 1 pt lines, built at final size.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
from _paths import DATA as _DATA
from paper_style import use_paper_style, size, save, CB, ROLE, LW, EMPH_LW
use_paper_style()

SRC = str(_DATA / "gait_traces.npz")
z = np.load(SRC)

ROWS = [                      # key, label as in Table III, colour of the actual angle
    ("tg",      "TG",      ROLE["cpg"]),          # blue, as everywhere the TG appears
    ("da",      "DA",      CB["purple"]),         # purple, as direct action in fig_velspectrum
    ("da_lpf1", "DA-FOF", CB["green"]),
]
T_MAX = 3.0                   # s; the July figure showed 0-6 s, cropped to the first 3 s

fig, axes = plt.subplots(len(ROWS), 1, sharex=True, sharey=True,
                         figsize=size(1.0, aspect=0.80), layout="constrained")
for ax, (key, label, col) in zip(axes, ROWS):
    ax.plot(z[f"{key}_cmd_t"], z[f"{key}_cmd_a"], color=ROLE["setpoint"], lw=LW,
            ls=(0, (5, 2)), zorder=2)
    ax.plot(z[f"{key}_act_t"], z[f"{key}_act_a"], color=col, lw=EMPH_LW, zorder=3,
            solid_capstyle="round")
    ax.set_title(label, loc="left", pad=2)          # above the axes, as in the July figure
    ax.set_yticks([-1, 0, 1])
    ax.set_ylim(-1.15, 1.15)
axes[-1].set_xlim(0, T_MAX)
axes[-1].set_xticks(range(0, int(T_MAX) + 1))
axes[-1].set_xlabel("Time (s)")
axes[1].set_ylabel("Motor angle (rad)")

handles = [Line2D([0], [0], color=ROLE["setpoint"], lw=LW, ls=(0, (5, 2)), label="Commanded"),
           Line2D([0], [0], color="0.25", lw=EMPH_LW, label="Actual")]
fig.legend(handles=handles, loc="outside lower center", ncol=2, handlelength=2.2, columnspacing=1.5)
fig.get_layout_engine().set(h_pad=0.02)
out = save(fig, "fig10_gait_traces", expect="col")
print("saved", out)
