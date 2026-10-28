#!/usr/bin/env python3
"""Deployment collage (fig:deploy) at the PAPER's full page width.

Two rounded boxes. (A), left, a tight 2x2 of overhead pool composites of fixed-target runs:
target reaching, disturbance rejection, out-of-distribution target, backward-velocity start,
each titled and annotated on the photograph, bottom-right, white on a translucent dark tag,
with start -> final distance and duration (computed from the tracked paths exactly as the
report's nav collage does, plot_nav_collage.py). (B), right, the moving-target pursuit: its
overhead composite above the tracked fish and target paths over time (fig_figure8_paper.draw),
both untitled: the B letter and the caption name them.
The path plot's axes box is exactly a grid cell so it lines up with the composites; it has no
axis numbers or titles (the 1 m grid and the caption give the scale); its time colourbar lies
under it, in the row that carries the distance/time notes in box A.

Each composite is cropped to its own trajectory and target (from the tracked paths, x1.5 from
the 1280x720 cache to the 1920x1080 composite, plus a margin), then the window is widened to
one aspect ratio shared by all panels, so the cells stay uniform while every path fills its
cell. The world anchor tag on the left rail is not kept.

Geometry is laid out by hand in inches (no constrained layout) so the cells are as large as
the page allows and every edge lines up. Full text width (7.0 in), 7 pt type, nothing scaled
in LaTeX. Replaces fig11_strobes_paper.py.

The paper's figure is --variant divider (the default): no rounded outlines, a light solid line
between A and B, written as fig7_deploy_collage. --variant boxes writes fig7_deploy_collage_boxes
with a rounded outline around A and around B; --variant tinted writes fig7_deploy_collage_tinted,
the rounded boxes filled with the warm hardware tint of Fig. 1 and no border.
"""
import os
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

import sys as _sys
from _paths import ANALYSIS as _ANALYSIS, OUT as _OUT, FISHDEPLOY as _FISHDEPLOY
_sys.path.insert(0, str(_FISHDEPLOY / "analysis"))   # plot_nav_collage lives in fishdeploy/analysis (the code; its data is under data/)
from paper_style import use_paper_style, save, FULL, PANEL_PT
import fig_figure8_paper as F8
import plot_nav_collage as NAV                  # for the per-run distance/time annotation only
use_paper_style()
plt.rcParams.update({"axes.grid": False})       # image panels: no grid, no spines

SRC = str(_ANALYSIS / "report_strobes")

_ap = argparse.ArgumentParser(description="deployment collage")
_ap.add_argument("--variant", choices=["boxes", "divider", "tinted"], default="divider",
                 help="divider: no outlines, a light solid line between A and B (default, the paper's "
                      "fig7_deploy_collage); boxes: rounded outline around A and around B "
                      "(fig7_deploy_collage_boxes); tinted: rounded boxes filled with the warm hardware "
                      "tint of Fig. 1, no border (fig7_deploy_collage_tinted)")
VARIANT = _ap.parse_args().variant
STEM = {"divider": "fig7_deploy_collage", "boxes": "fig7_deploy_collage_boxes",
        "tinted": "fig7_deploy_collage_tinted"}[VARIANT]
TINT = "#F7F2EB"                                # Fig. 1's hardware (warm) side-panel tint

# (run, composite file, title, truncate-at-closest-approach)
LEFT = [
    ("hope_didnt_work_2", "hope_didnt_work_2_highres_traceover_strobe.png",      "Target reaching",           True),
    ("insane_run_8",      "insane_run_8_middle_bg_highres_traceover_strobe.png", "Disturbance rejection",     True),
    ("finetune_run_7",    "finetune_run_7_highres_traceover_strobe.png",         "Out-of-distribution target", True),
    ("insane_run_2",      "insane_run_2_highres_traceover_strobe.png",           "Backward-velocity start",   True),
]
RIGHT_STROBE = "figure_eight_6_clean_highres_traceover_strobe.png"

# per-run crop windows in composite pixels (1920x1080), all with one shared aspect ratio
CACHE = str(_ANALYSIS / "cache")
PX_SCALE = 1.5                                  # cache tracks are in 1280x720, composites are 1920x1080
PAD_PX = {"default": 70, "figure_eight_6": 35}  # margin around the path and the target tag (tighter on the
                                                #   figure-eight, whose paths already span most of the pool)
EXCLUDE_LEFT_PX = 150                           # never show the world anchor tag on the left rail
SHIFT_PX = {"figure_eight_6": 47}               # move a run's window down (+) after fitting: the figure-eight
                                                #   window otherwise starts at the pool's top edge and clips the
                                                #   fish at the bottom right
FRAME = (1920, 1080)


def content_box(run):
    """Robust pixel bbox of the fish path and the target in the composite."""
    import numpy as np
    d = np.load(os.path.join(CACHE, f"{run}.npz"), allow_pickle=True)
    fp = d["fish_px"].astype(float) * PX_SCALE; tp = d["tgt_px"].astype(float) * PX_SCALE
    fp = fp[~np.isnan(fp[:, 0])]; tp = tp[~np.isnan(tp[:, 0])]
    pad = PAD_PX.get(run, PAD_PX["default"])
    lo = np.minimum(np.percentile(fp, 0.5, axis=0), np.percentile(tp, 1, axis=0)) - pad
    hi = np.maximum(np.percentile(fp, 99.5, axis=0), np.percentile(tp, 99, axis=0)) + pad
    return [max(EXCLUDE_LEFT_PX, lo[0]), max(0.0, lo[1]), min(FRAME[0], hi[0]), min(FRAME[1], hi[1])]


def crop_windows(runs):
    """One window per run: its content box widened to the shared aspect (the median of the
    content aspects), kept inside the frame. Returns (windows, aspect)."""
    import numpy as np
    boxes = {r: content_box(r) for r in runs}
    aspect = float(np.median([(b[2] - b[0]) / (b[3] - b[1]) for b in boxes.values()]))
    out = {}
    for r, (x0, y0, x1, y1) in boxes.items():
        w, h = x1 - x0, y1 - y0
        if w / h < aspect:                       # too tall for the shared aspect: widen
            nw = min(h * aspect, FRAME[0]); cx = (x0 + x1) / 2; x0, x1 = cx - nw / 2, cx + nw / 2
            h = nw / aspect; cy = (y0 + y1) / 2; y0, y1 = cy - h / 2, cy + h / 2
        else:                                    # too wide: heighten
            nh = min(w / aspect, FRAME[1]); cy = (y0 + y1) / 2; y0, y1 = cy - nh / 2, cy + nh / 2
            w = nh * aspect; cx = (x0 + x1) / 2; x0, x1 = cx - w / 2, cx + w / 2
        if x0 < EXCLUDE_LEFT_PX: x1 += EXCLUDE_LEFT_PX - x0; x0 = EXCLUDE_LEFT_PX
        if x1 > FRAME[0]: x0 = max(EXCLUDE_LEFT_PX, x0 - (x1 - FRAME[0])); x1 = FRAME[0]
        if y0 < 0: y1 -= y0; y0 = 0
        if y1 > FRAME[1]: y0 -= y1 - FRAME[1]; y1 = FRAME[1]
        dy = min(SHIFT_PX.get(r, 0), FRAME[1] - y1)   # shift down, never past the frame
        y0 += dy; y1 += dy
        out[r] = tuple(int(round(v)) for v in (x0, y0, x1, y1))
        print(f"crop {r:18s} x {out[r][0]:4d}..{out[r][2]:4d}  y {out[r][1]:4d}..{out[r][3]:4d}")
    return out, aspect


RUNS = [r for r, *_ in LEFT] + ["figure_eight_6"]
CROPS, IMG_ASPECT = crop_windows(RUNS)

# ---- geometry, inches ------------------------------------------------------------------------
W = FULL                       # 7.0 in = \textwidth
GAP = 0.02                     # between cells (rows, and between the two left columns)
BOX_PAD = 0.045                # box edge to cell edge, above and below
BOX_PAD_X = 0.05               # box edge to cell edge, left and right (tight: the cells take the width)
BOX_GAP = 0.10                 # white between box A and box B
M_T = M_B = 0.03               # figure edge to box edge, above and below
M_L = 0.02
CBAR_H, CBAR_GAP = 0.06, 0.03                   # horizontal colourbar under the path plot, and its gap to it
M_R = 0.02
TITLE_H = 0.22                 # 9 pt title line above each cell, with air between title and photograph
                               #   (kept on the right too, untitled: the B label lives there)
TITLE_PAD = 3.5                # points between the title baseline and the top of the cell
NOTE_H = 0.25                  # under the bottom cells: the time colourbar + its 8 pt ticks (B); A's notes sit on the photographs
NOTE_H_TOP = 0.0               # under the top cells: nothing (the notes are on the photographs), so the rows sit close
CW = (W - M_L - M_R - 4 * BOX_PAD_X - BOX_GAP - GAP) / 3
RH = CW / IMG_ASPECT
H = M_T + M_B + 2 * BOX_PAD + 2 * (TITLE_H + RH) + NOTE_H + NOTE_H_TOP + GAP
X_A = M_L + BOX_PAD_X                                     # left edge of the cells in A
X_B = X_A + 2 * CW + GAP + BOX_PAD_X + BOX_GAP + BOX_PAD_X  # left edge of the cell in B
COL_X = [X_A, X_A + CW + GAP, X_B]
Y_BOT = M_B + BOX_PAD + NOTE_H                            # bottom of the bottom cells (note line below)
ROW_Y = [Y_BOT + RH + TITLE_H + NOTE_H_TOP + GAP, Y_BOT]  # top row, bottom row
Y_TOP = ROW_Y[0] + RH + TITLE_H                           # top of the title line of the top row


def cell(col, row):
    return fig.add_axes([COL_X[col] / W, ROW_Y[row] / H, CW / W, RH / H])


def note_for(run, truncate, scale):
    """The report collage's annotation string for this run (drawn on a throw-away axes)."""
    f, a = plt.subplots()
    NAV.panel(a, run, "", truncate, False, scale)
    s = a.texts[-1].get_text()
    plt.close(f)
    return s


def composite(ax, run, fn, title=None, note=None):
    x0, y0, x1, y1 = CROPS[run]
    im = plt.imread(os.path.join(SRC, fn))[y0:y1, x0:x1]
    ax.imshow(im, aspect="auto")
    if title:
        ax.set_title(title, pad=TITLE_PAD)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if note:                                    # on the photograph, bottom-right: white on a translucent dark tag
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom", color="white", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="0.2", ec="none", alpha=0.6))


def box(x0, y0, x1, y1, label):
    """The group's letter in its upper-left corner and, in the boxes variant, a rounded outline."""
    if VARIANT in ("boxes", "tinted"):
        fc, ec = ("none", "0.55") if VARIANT == "boxes" else (TINT, "none")
        fig.add_artist(FancyBboxPatch((x0 / W, y0 / H), (x1 - x0) / W, (y1 - y0) / H,
                                      boxstyle="round,pad=0,rounding_size=0.012", transform=fig.transFigure,
                                      fc=fc, ec=ec, lw=0.8, zorder=-1))
    fig.text((x0 + 0.05) / W, (y1 - 0.035) / H, r"\textbf{" + label + "}", ha="left", va="top",
             fontsize=PANEL_PT, zorder=12)


scale = NAV.get_scale()
fig = plt.figure(figsize=(W, H))

# ---- A: four fixed-target composites ------------------------------------------------------
for i, (run, fn, title, trunc) in enumerate(LEFT):
    composite(cell(i % 2, i // 2), run, fn, title, note_for(run, trunc, scale))
box(X_A - BOX_PAD_X, Y_BOT - NOTE_H - BOX_PAD, COL_X[1] + CW + BOX_PAD_X, Y_TOP + BOX_PAD, "A")

# ---- B: the pursuit composite over the fish and target paths ------------------------------
composite(cell(2, 0), "figure_eight_6", RIGHT_STROBE)    # no title, no note: the B letter and the caption say it
plt.rcParams.update({"axes.grid": True})
ax_f = cell(2, 1)
lc = F8.draw(fig, ax_f, legend=None, cbar=False, box_aspect=IMG_ASPECT)
ax_f.set_ylabel(""); ax_f.set_xlabel("")                  # no title, axis titles or numbers: the 1 m grid
ax_f.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)  # and the caption carry the scale
ax_f.legend(handles=F8.legend_handles(short=True), loc="upper right", ncol=2, frameon=True,
            framealpha=0.9, edgecolor="none", handlelength=2.0, columnspacing=1.2, handletextpad=0.5,
            borderpad=0.3, borderaxespad=0.3)
cax = fig.add_axes([X_B / W, (Y_BOT - CBAR_GAP - CBAR_H) / H, 0.78 * CW / W, CBAR_H / H])
cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
cb.outline.set_linewidth(0.6)
cax.tick_params(length=2, pad=1.5)
fig.text((X_B + 0.80 * CW) / W, (Y_BOT - NOTE_H / 2) / H, "Time (s)", ha="left", va="center")   # centred in the note row
box(X_B - BOX_PAD_X, Y_BOT - NOTE_H - BOX_PAD, W - M_R, Y_TOP + BOX_PAD, "B")

if VARIANT == "divider":                                  # one light solid line between A and B
    x_div = (COL_X[1] + CW + BOX_PAD_X + BOX_GAP / 2) / W
    fig.add_artist(Line2D([x_div, x_div], [(Y_BOT - NOTE_H - BOX_PAD) / H, (Y_TOP + BOX_PAD) / H],
                          transform=fig.transFigure, color="0.7", lw=0.8, ls="-", zorder=1))

out = save(fig, STEM, expect="full")
print("saved", out)
