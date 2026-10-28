#!/usr/bin/env python3
"""Fig: the figure-eight pursuit run (figure_eight_6): fish and target paths over time.

The point-target policy tracking a target moved by hand along a figure-eight for ~75 s.
Fish path coloured by time (plasma reversed: yellow at t = 0 to dark blue at the end, the same
direction as the pool composites; cut at 85% so the start is not pale yellow), moving-target
path dotted on the same time scale, a heading arrow on the fish and a hollow circle on the
target at t = 0 and every 10 s in the colour of that instant, so the two can be compared in
time; the fish start is the origin. Same geometry as panel (f) of the deployment nav collage
(plot_nav_collage.py): the overhead pixel track scaled to metres with the m/px factor of
insane_run_8 (this run has no metric AprilTag world track), image y flipped so up is up.

Standalone:  python fig_figure8_paper.py   -> out/fig_figure8_paper.pdf (single column)
As a panel:  from fig_figure8_paper import draw; draw(fig, ax, legend="axes")
             (used by fig_deploy_collage_paper.py for panel (f) of the deployment collage)

Data: analysis_20260520/cache/figure_eight_6.npz (+ insane_run_8.npz for the scale).
7 pt type, >= 1 pt lines, built at final size.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import sys as _sys
from _paths import ANALYSIS as _ANALYSIS
from paper_style import use_paper_style, size, save, LW, EMPH_LW

CACHE = str(_ANALYSIS / "cache")
RUN, SCALE_RUN = "figure_eight_6", "insane_run_8"
GAP_M = 0.30                       # break the drawn path across tracking gaps longer than this
TICK_S = 10.0                      # time stamps on both paths
LOOK_S = 0.6                       # s, look-ahead for the fish heading arrow
DOTS = (0, (0.6, 2.8))             # legend handle for the dotted target path: dot 0.6 lw, gap 2.8 lw
# plasma REVERSED (yellow start -> dark blue end) to match the pool composites, whose traces run
# yellow at the start to blue at the end; cut before the pale-yellow end so t = 0 still prints on white
CMAP = ListedColormap(plt.get_cmap("plasma")(np.linspace(0.85, 0.0, 256)), name="plasma_cut_r")


def _load(run):
    return dict(np.load(os.path.join(CACHE, f"{run}.npz"), allow_pickle=True))


def _metres_per_px():
    d = _load(SCALE_RUN)
    fw, fp = d["fish_world"], d["fish_px"]
    v = ~np.isnan(fw[:, 0]) & ~np.isnan(fp[:, 0])
    return (np.nanmax(fw[v, 0]) - np.nanmin(fw[v, 0])) / (np.nanmax(fp[v, 0]) - np.nanmin(fp[v, 0]))


def load_tracks():
    """Fish and target tracks in metres, fish start at the origin, time from the first fish frame."""
    d = _load(RUN)
    s = _metres_per_px()
    fm = np.column_stack([d["fish_px"][:, 0] * s, -d["fish_px"][:, 1] * s])   # flip image y -> world up
    tm = np.column_stack([d["tgt_px"][:, 0] * s, -d["tgt_px"][:, 1] * s])
    ft = d["frame_time"].astype(float)
    vf = ~np.isnan(fm).any(1); vt = ~np.isnan(tm).any(1)
    g0 = fm[np.where(vf)[0][0]].copy()
    fm -= g0; tm -= g0
    t0 = ft[np.where(vf)[0][0]]
    return dict(x=fm[vf, 0], y=fm[vf, 1], t=ft[vf] - t0,
                tx=tm[vt, 0], ty=tm[vt, 1], tt=ft[vt] - t0,
                t_end=ft[np.where(vf)[0][-1]] - t0)


def _at_time(tq, t_arr, x_arr, y_arr, tol=0.5):
    """Position at time tq by interpolation, or None if the track has no sample within tol seconds."""
    if np.min(np.abs(t_arr - tq)) > tol:
        return None
    return np.interp(tq, t_arr, x_arr), np.interp(tq, t_arr, y_arr)


def legend_handles(short=False):
    """short=True gives bare "Fish" / "Target" labels (the caption then explains the 10 s stamps)."""
    lf = "Fish" if short else f"Fish (heading every {TICK_S:.0f} s)"
    lt = "Target" if short else f"Target (every {TICK_S:.0f} s)"
    return [
        Line2D([0], [0], color="0.35", lw=EMPH_LW, marker=(3, 0, -90), ms=7, mfc="0.35", mec="none", label=lf),
        Line2D([0], [0], color="0.35", ls=DOTS, lw=LW, dash_capstyle="round", marker="o", ms=4.5,
               mfc="none", mec="0.35", mew=0.9, label=lt),
    ]


def draw(fig, ax, legend="fig", cbar=True, cbar_kw=None, box_aspect=None):
    """Draw the figure-eight panel into `ax`. legend: "fig" (outside lower centre of the figure),
    "axes" (below the axes), or None. box_aspect: width/height of the axes box; if given, the
    limits are widened to that ratio so the equal-scale plot fills the box exactly (used when the
    panel must line up with image cells in a collage). Returns the fish LineCollection."""
    D = load_tracks()
    x, y, tt, tx, ty, ttt, t_end = D["x"], D["y"], D["t"], D["tx"], D["ty"], D["tt"], D["t_end"]
    norm = plt.Normalize(0, t_end)

    # fish path as time-coloured segments, broken across tracking gaps
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seglen = np.hypot(*(segs[:, 1] - segs[:, 0]).T)
    ok = seglen < GAP_M
    lc = LineCollection(segs[ok], cmap=CMAP, norm=norm, linewidth=EMPH_LW, capstyle="round", zorder=3)
    lc.set_array(tt[:-1][ok])
    ax.add_collection(lc)

    # time stamps every TICK_S on both paths, in the path's colour at that instant: a heading
    # arrow on the fish, a hollow circle on the target; skipped where the track has a gap.
    for tq in np.arange(0.0, t_end, TICK_S):
        col = CMAP(norm(tq))
        pf, pf2 = _at_time(tq, tt, x, y), _at_time(tq + LOOK_S, tt, x, y)
        if pf is not None and pf2 is not None and np.hypot(pf2[0] - pf[0], pf2[1] - pf[1]) > 1e-3:
            ang = np.degrees(np.arctan2(pf2[1] - pf[1], pf2[0] - pf[0]))
            ax.plot(*pf, marker=(3, 0, ang - 90), ms=7, color=col, mec="none", ls="", zorder=6)
        pt = _at_time(tq, ttt, tx, ty)
        if pt is not None:
            ax.plot(*pt, "o", ms=4.5, mfc="none", mec=col, mew=0.9, ls="", zorder=6)

    ax.axhline(0, color="0.6", lw=0.6, ls=":", zorder=0)
    ax.axvline(0, color="0.6", lw=0.6, ls=":", zorder=0)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.xaxis.set_major_locator(MultipleLocator(1.0)); ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5)); ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, color="0.85", lw=0.4)
    for s_ in ax.spines.values():
        s_.set_visible(True)

    # equal aspect, limits tight around everything drawn (the run is wider than it is tall)
    ax_all = np.concatenate([x, tx, [0.0]]); ay_all = np.concatenate([y, ty, [0.0]])
    PAD = 0.15
    x0, x1 = ax_all.min() - PAD, ax_all.max() + PAD
    y0, y1 = ay_all.min() - PAD, ay_all.max() + PAD
    if box_aspect is None:
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
    else:                                            # pad the narrower dimension to the box ratio
        if (x1 - x0) / (y1 - y0) < box_aspect:
            extra = (y1 - y0) * box_aspect - (x1 - x0); x0 -= extra / 2; x1 += extra / 2
        else:
            extra = (x1 - x0) / box_aspect - (y1 - y0); y0 -= extra / 2; y1 += extra / 2
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("auto")                        # limits already match the box: equal scale, full fill

    if cbar:
        kw = dict(pad=0.02, fraction=0.05, aspect=28); kw.update(cbar_kw or {})
        cb = fig.colorbar(lc, ax=ax, **kw)
        cb.set_label("Time (s)")
        cb.outline.set_linewidth(0.6)

    if legend == "fig":
        fig.legend(handles=legend_handles(), loc="outside lower center", ncol=2, handlelength=2.0,
                   columnspacing=1.2, handletextpad=0.5)
    elif legend == "axes":                       # compact, under the axes, for use as a collage panel
        ax.legend(handles=legend_handles(short=True), loc="upper center", bbox_to_anchor=(0.5, -0.26),
                  ncol=2, handlelength=2.0, columnspacing=1.6, handletextpad=0.5, frameon=False)

    # target path: dotted, coloured by the same time scale. The track has thousands of tiny
    # segments, so a dash pattern would restart every frame and read solid; instead the dots are
    # placed at a fixed spacing along the arc length, in points on the page, like the legend handle.
    # Needs the final axes size, so the layout is settled first.
    fig.canvas.draw()
    tstep = np.hypot(np.diff(tx), np.diff(ty)); tstep[tstep > GAP_M] = 0.0
    tcum = np.concatenate([[0.0], np.cumsum(tstep)])
    pt_per_m = ax.get_window_extent().width / np.ptp(ax.get_xlim()) * 72 / fig.dpi
    spacing_m = (0.6 + 2.8) * LW / pt_per_m
    s_dots = np.arange(0.0, tcum[-1], spacing_m)
    ax.scatter(np.interp(s_dots, tcum, tx), np.interp(s_dots, tcum, ty), c=np.interp(s_dots, tcum, ttt),
               cmap=CMAP, norm=norm, s=(0.9 * LW) ** 2, marker="o", linewidths=0, zorder=2)

    seg_ok = np.where(seglen > GAP_M, 0.0, seglen)
    print(f"{RUN}: {len(x)} tracked fish frames over {t_end:.1f} s, path length {seg_ok.sum():.1f} m")
    return lc


if __name__ == "__main__":
    use_paper_style()
    fig, ax = plt.subplots(figsize=size(1.0, aspect=0.66), layout="constrained")
    draw(fig, ax, legend="fig")
    out = save(fig, "fig_figure8_paper", expect="col")
    print("saved", out)
