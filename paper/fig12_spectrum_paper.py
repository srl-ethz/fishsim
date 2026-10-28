#!/usr/bin/env python3
"""Fig 12 (deploy command spectrum) built at the PAPER's column width.

The report asset is 325 pt wide shown in a 247 pt column (scale 0.75 -> 5.3 pt type).
This composes the same plot directly at 3.42 in with 7 pt type and >=1 pt lines, and
gives the legend its own space instead of letting it sit on the data.
Data helpers are reused from the report's deploy_plots; nothing there is modified.
"""
import os, sys
from _paths import PAPER as _PAPER, DEPLOY_RUNS as _DEPLOY_RUNS
SCRIPTS = str(_PAPER)
sys.path.insert(0, SCRIPTS); os.chdir(SCRIPTS)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
import deploy_plots as D

from paper_style import use_paper_style, size, save, CB, LW, EMPH_LW

BLUE, GREY, VERM = CB["blue"], CB["grey"], CB["orange"]
use_paper_style()

r = D.load_run("hope_didnt_work_2")
fs = 1.0 / np.median(np.diff(r["t"]))
f, P = welch(r["tail_cmd"] - r["tail_cmd"].mean(), fs=fs, nperseg=min(256, len(r["tail_cmd"])))
fpk = f[1:][np.argmax(P[1:])]
allP = D._all_run_psds(f)
lo, hi = np.percentile(allP, [10, 90], axis=0)

fig, ax = plt.subplots(figsize=size(1.0, aspect=0.62), layout="constrained")
ax.fill_between(f, np.maximum(lo, 1e-9), hi, color=BLUE, alpha=0.16, lw=0,
                label=r"all runs (10--90\%%, $n$=%d)" % len(allP))
ax.semilogy(f, np.median(allP, axis=0), color=BLUE, lw=LW, alpha=0.55, label="median")
ax.semilogy(f, P, color=BLUE, lw=EMPH_LW, label="first successful run")
ax.set_ylim(3e-5, 6e-1)
ax.axvline(4.0, color=GREY, ls="--", lw=LW)
ax.axvline(6.0, color=GREY, ls=":", lw=LW)
ax.annotate("4 Hz\ntrain", (4.0, 3.6e-1), xytext=(2, 0), textcoords="offset points",
            color=GREY, va="top", ha="left")
ax.annotate("6 Hz\ndeploy", (6.0, 3.6e-1), xytext=(2, 0), textcoords="offset points",
            color=GREY, va="top", ha="left")
ax.plot(fpk, P[1:].max(), "o", color=VERM, ms=3.5)
ax.annotate("peak %.2f Hz" % fpk, (fpk, P[1:].max()), xytext=(4, 5),
            textcoords="offset points", color=VERM)
ax.set_xlim(0, 8)
ax.set_xlabel("frequency [Hz]")
ax.set_ylabel(r"command PSD [rad$^2$/Hz]")
# legend below the axes so it never covers the band
# constrained layout reserves room for an "outside" legend, so the figure keeps its
# exact width -- unlike bbox_inches="tight", which would resize it after the fact
fig.legend(loc="outside lower center", ncol=1, frameon=False,
           handlelength=1.4, labelspacing=0.25)
print("peak %.2f Hz, n=%d" % (fpk, len(allP)))
save(fig, "fig12_spectrum_paper", expect="col")
