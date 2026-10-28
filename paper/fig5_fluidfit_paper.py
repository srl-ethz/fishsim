#!/usr/bin/env python3
"""Fig 5 (fluid-model fit) at the PAPER's column width.

The paper's figure is the matched pair (--pair): one fitted and one held-out run at the same
condition, written as fig5_fluidfit.pdf. Mike: "Figure 5 is too much, at most 3 trajectories.
Or we somehow overlay multiple trajectories into a single plot."  The candidate layouts that
led there are kept behind --all:

  fig5_fluidfit_overlay.pdf   all six fitting runs in one axes (nothing dropped)
  fig5_fluidfit_panels.pdf    three representative runs as a row of panels
  fig5_fluidfit_grid.pdf      2x2, fitted row over validation row (the default without --pair)

Each run is replayed exactly as the identification did it: the recorded motor angle
(Ghost_Alpha) is teleported into the E13a twin and the head path compared with the
measured one (Ghost_X/Ghost_Y), which share the same frame and origin.

Frame. The field pipeline zeroes the pose at the first sample, so the origin is the start
position and the x axis is the fish's INITIAL HEADING (Ghost_Yaw = 0 at t = 0), not the mean
swim direction: the lateral drift over a run is what the plots show. In that frame the fish
swims towards -x (the tag's x points aft). For the figure the frame is rotated by 180 degrees
(x -> -x, y -> -y) so the swim axis reads positive. A rotation, not a mirror: turns keep their
direction and every distance is unchanged.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import mujoco

from _paths import REPO as _REPO, FIELD as _FIELD, OUT as _OUT
ROOT = str(_REPO)
sys.path.insert(0, ROOT)
from geometry import sofi as G

FIELD = str(_FIELD)
DEST  = str(_OUT)
COLW  = 246.945 / 72.27
CB = {"real": "#0072B2", "twin": "#D55E00"}

# the six runs the E13a fluid was fitted on (five straight + one right turn)
# E13 was fitted on exactly these six runs (FLUID_COEFFICIENTS.md, "E13 --- 15 coefficients");
# run_9 is the fitted turn.
FITTED = [("run_03_1.0Hz_9.0mm",        "1.0 Hz, 0.64 rad"),
          ("run_04_2.0Hz_9.0mm",        "2.0 Hz, 0.51 rad"),
          ("run_07_3.0Hz_7.0mm",        "3.0 Hz, 0.33 rad"),
          ("run_10_4.0Hz_7.0mm",       "4.0 Hz, 0.24 rad"),
          ("run_11_2.0Hz_12.0mm",      "2.0 Hz, 0.52 rad"),
          ("run_09_2.0Hz_6.0mm_-3.0mm", "2.0 Hz, turn")]
# the other six field2sim-ready runs of the session; none appears in any fit.
# run_8 is the turn, in the direction opposite run_9.
VALID = [("run_01_1.0Hz_5.0mm",         "1.0 Hz, 0.35 rad"),
         ("run_02_2.0Hz_5.0mm",         "2.0 Hz, 0.35 rad"),
         ("run_05_1.0Hz_7.0mm",         "1.0 Hz, 0.50 rad"),
         ("run_06_2.0Hz_7.0mm",         "2.0 Hz, 0.46 rad"),
         ("run_12_2.0Hz_12.0mm_p-1600","2.0 Hz, 0.51 rad"),
         ("run_08_2.0Hz_6.0mm_3.0mm",   "2.0 Hz, turn")]
# run_2 was nudged by hand shortly after the start and slid off line, so it is sound
# for visual inspection but not for a quoted number.
QUARANTINE = {"run_02_2.0Hz_5.0mm"}
HELDOUT = VALID
FIT = FITTED + VALID
MARK_T = 8.0        # blue dot on both paths at this elapsed time
# 2x2: fitted row over validation row, straight column then turn column
GRID = [(FITTED[0][0], "1.0\\,Hz straight"), (FITTED[5][0], "2.0\\,Hz turn"),
        (VALID[2][0],  "1.0\\,Hz straight"), (VALID[5][0],  "2.0\\,Hz turn")]
# --pair: one fitted and one held-out run at the SAME condition. The figure makes one point,
# that held-out runs degrade gracefully, and a matched pair makes it without the other two.
PAIR = [(FITTED[0][0], "1.0\\,Hz"), (VALID[2][0], "1.0\\,Hz")]   # both straight runs; the caption says so
TWO = "--pair" in sys.argv
ALL = "--all" in sys.argv          # also write the overlay and panels variants
PANELS = [(FITTED[0][0], "1.0\\,Hz, fitted"),
          (VALID[3][0],  "2.0\\,Hz, validation"),
          (VALID[5][0],  "turn, validation")]

from paper_style import use_paper_style, size, save, CB as PCB, LW, EMPH_LW, ROLE
use_paper_style()


def build_model():
    p = G.SYSTEMPARAMETERS.copy(); p["motor_ctrlrange"] = 1.2
    xml = os.path.join(ROOT, "geometry", "_fig5_%d.xml" % os.getpid())
    p["filename"] = xml
    G.generate_xml(p, filename=xml)
    return mujoco.MjModel.from_xml_path(xml), xml


def replay(m, run):
    d = pd.read_csv(os.path.join(FIELD, run, "processed", "03_field2sim_ready.csv"))
    t = d["Time"].values - d["Time"].values[0]
    a, da = d["Ghost_Alpha"].values, d["Ghost_dAlpha"].values
    real = -np.column_stack([d["Ghost_X"].values, d["Ghost_Y"].values])   # 180 deg, see docstring
    dat = mujoco.MjData(m)
    J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    mq, mv = m.jnt_qposadr[J("motor_0")], m.jnt_dofadr[J("motor_0")]
    hx, hy = m.jnt_qposadr[J("headX_0")], m.jnt_qposadr[J("headY_0")]
    dt = m.opt.timestep; twin = []
    for i in range(int(t[-1] / dt)):
        tt = i * dt
        dat.qpos[mq] = np.interp(tt, t, a); dat.qvel[mv] = np.interp(tt, t, da)
        dat.ctrl[0] = dat.qpos[mq]
        mujoco.mj_step(m, dat)
        if i % 20 == 0:
            twin.append((dat.qpos[hx], dat.qpos[hy]))
    return real, -np.array(twin), t


if __name__ == "__main__":
    m, xml = build_model()
    data = {r: replay(m, r) for r, _ in FIT}
    os.remove(xml)

    # ---------------- variant A: everything overlaid ----------------
    # one hue per run so each measured/twin pair is visually linked
    # --- honest per-run error, so the text can quote numbers instead of "tracks well"
    # twin is sampled every 20 sim steps over [0, t[-1]]; real is on the recording's own
    # clock, so the two must be put on a common time base before any error is quoted.
    # The t=8 s gap is the number the green square in the plots visualises: has the twin
    # covered the same ground in the same time?
    def metrics(r):
        real, twin, t = data[r]
        tt = np.linspace(0.0, t[-1], len(twin))
        tw = np.column_stack([np.interp(t, tt, twin[:, 0]), np.interp(t, tt, twin[:, 1])])
        err = np.hypot(*(tw - real).T)
        path = float(np.sum(np.hypot(*np.diff(real, axis=0).T)))
        tm = min(MARK_T, t[-1])
        i8 = int(np.searchsorted(t, tm))
        i8 = min(i8, len(real) - 1)
        d_real = float(np.hypot(*(real[i8] - real[0])))
        d_twin = float(np.hypot(*(tw[i8] - real[0])))
        return dict(path=path, mean=float(err.mean()), gap8=float(err[i8]),
                    tm=tm, d_real=d_real, d_twin=d_twin)

    M = {r: metrics(r) for r, _ in FIT}
    print("\n  run                        role  path   mean err  t=8s gap  dist@8s real/twin")
    for grp, name in ((FITTED, "fit"), (VALID, "val")):
        for r, lab in grp:
            m = M[r]; q = " *hand-nudged" if r in QUARANTINE else ""
            print(f"  {r:26s} {name:4s} {m['path']:5.2f} m {m['mean']:7.3f} m {m['gap8']:8.3f} m "
                  f"{m['d_real']:6.2f}/{m['d_twin']:.2f} m{q}")
    def agg(rs, key):
        v = [M[r][key] for r in rs]
        return sum(v) / len(v)
    f_r = [r for r, _ in FITTED]
    v_r = [r for r, _ in VALID]
    v_q = [r for r in v_r if r not in QUARANTINE]
    print(f"\n  mean-of-means path error:  fitted (n={len(f_r)}) {agg(f_r,'mean'):.3f} m | "
          f"validation (n={len(v_r)}) {agg(v_r,'mean'):.3f} m | "
          f"validation minus run_2 (n={len(v_q)}) {agg(v_q,'mean'):.3f} m")
    print(f"  mean t=8 s gap:            fitted {agg(f_r,'gap8'):.3f} m | "
          f"validation {agg(v_r,'gap8'):.3f} m | minus run_2 {agg(v_q,'gap8'):.3f} m\n")

    if ALL:
        # colour carries the only thing the figure is arguing: fitted vs never fitted
        HELD = {r for r, _ in HELDOUT}
        C_FIT, C_HELD = PCB["blue"], PCB["orange"]
        fig, ax = plt.subplots(figsize=size(1.0, aspect=0.58), layout="constrained")
        STRAIGHT = [(r, l) for r, l in FIT if 'turn' not in l]
        for r, lab in STRAIGHT:
            real, twin, _t = data[r]
            c = C_HELD if r in HELD else C_FIT
            z = 5 if r in HELD else 3
            ax.plot(real[:, 0], real[:, 1], color=c, lw=LW, alpha=0.9, zorder=z)
            ax.plot(twin[:, 0], twin[:, 1], color=c, lw=LW, ls=(0, (3, 2)), alpha=0.9, zorder=z + 1)
            ax.plot(real[-1, 0], real[-1, 1], "o", color=c, ms=2.6, zorder=z + 2)
        ax.plot(0, 0, "k.", ms=4, zorder=9)
        ax.set_xlabel("Swim axis (m)"); ax.set_ylabel("Lateral (m)")
        ax.margins(0.04)
        h = [
            Line2D([0], [0], color="0.25", lw=LW, label="Measured"),
            Line2D([0], [0], color="0.25", lw=LW, ls=(0, (3, 2)), label="Simulated twin"),
            Line2D([0], [0], color=C_FIT, lw=LW, label="Fitted"),
            Line2D([0], [0], color=C_HELD, lw=LW, label="Held out")]
        fig.legend(handles=h, loc="outside lower center", ncol=4)
        save(fig, "fig5_fluidfit_overlay", expect="col")

        # ---------------- variant B: three panels ----------------
        fig, axs = plt.subplots(1, 3, figsize=size(1.0, aspect=0.46), layout="constrained", sharey=True)
        for ax, (r, lab) in zip(axs, PANELS):
            real, twin, _t = data[r]
            ax.plot(real[:, 0], real[:, 1], color=ROLE["measured"], lw=LW, zorder=3)
            ax.plot(twin[:, 0], twin[:, 1], color=ROLE["twin"], lw=LW, ls="--", zorder=4)
            ax.plot(0, 0, "k.", ms=3)
            ax.set_title(lab, pad=2)
            ax.xaxis.set_major_locator(MaxNLocator(3))
        axs[0].set_ylabel("Lateral (m)")
        fig.supxlabel("Swim axis (m)")
        axs[0].plot([], [], color=ROLE["measured"], lw=LW, label="Measured")
        axs[0].plot([], [], color=ROLE["twin"], lw=LW, ls="--", label="Twin")
        axs[0].legend(loc="lower left", handlelength=1.2)
        save(fig, "fig5_fluidfit_panels", expect="col")

    # ---------------- variant C: 2x2, fitted row over validation row ----------------
    # one straight and one turn in each row, so "how we fitted" and "how validation came
    # out" are read off the same geometry. Blue dot = position at t=8 s on BOTH
    # paths, which is the distance-in-equal-time check (convention from eval_sysid.py).
    # --pair: the fitted run stacked over the held-out run at the same condition, both panels on
    # identical x/y limits at equal aspect, so the two are read at the same scale
    _items = PAIR if TWO else GRID
    fig, axs = plt.subplots(*( (2, 1) if TWO else (2, 2) ),
                            figsize=size(1.0, aspect=0.50 if TWO else 0.78),
                            layout="constrained", sharex=TWO, sharey=TWO)
    axs = np.atleast_1d(axs)
    for ax, (r, cond) in zip(np.atleast_1d(axs).flat, _items):
        real, twin, t = data[r]
        tt = np.linspace(0.0, t[-1], len(twin))
        tw = np.column_stack([np.interp(t, tt, twin[:, 0]), np.interp(t, tt, twin[:, 1])])
        role = "Fitted" if r in {x for x, _ in FITTED} else "Validation"
        ax.plot(real[:, 0], real[:, 1], color=ROLE["measured"], lw=LW, zorder=3)
        ax.plot(tw[:, 0], tw[:, 1], color=ROLE["twin"], lw=LW, ls="--", zorder=4)
        i8 = min(int(np.searchsorted(t, min(MARK_T, t[-1]))), len(real) - 1)
        ax.plot(real[i8, 0], real[i8, 1], "o", color=PCB["blue"], ms=3.2, zorder=6)
        ax.plot(tw[i8, 0], tw[i8, 1], "o", color=PCB["blue"], ms=3.2, zorder=6)
        ax.plot(real[0, 0], real[0, 1], "k.", ms=3, zorder=7)
        ax.set_title(r"%s Trial (1\,Hz)" % role if TWO else r"%s, %s" % (role, cond), pad=2)
        ax.margins(0.06)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
    if TWO:
        # both panels on identical limits, so the two runs are compared on the same scale,
        # with a margin so neither trace runs into the frame
        xs = [v for r, _ in _items for v in (data[r][0][:, 0].min(), data[r][0][:, 0].max())]
        ys = [v for r, _ in _items for v in (data[r][0][:, 1].min(), data[r][0][:, 1].max())]
        for r, _ in _items:
            tw = data[r][1]
            xs += [tw[:, 0].min(), tw[:, 0].max()]
            ys += [tw[:, 1].min(), tw[:, 1].max()]
        xpad = 0.03 * (max(xs) - min(xs))
        for ax in axs:
            ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
            ax.set_ylim(min(ys) - 0.10, max(ys) + 0.08)     # room for the traces and two tick labels
            ax.set_yticks([-0.4, -0.2, 0.0])
            ax.margins(0)
            ax.set_aspect("equal", adjustable="box")     # 0.3 m is the same length on both axes
            ax.set_ylabel("$y$ (m)")
        axs[-1].set_xlabel("$x$ (m)")
    else:
        for ax in axs[:, 0]: ax.set_ylabel("Lateral (m)")
        for ax in axs[1, :]: ax.set_xlabel("Swim axis (m)")
    fig.legend(handles=[
        Line2D([0], [0], color=ROLE["measured"], lw=LW, label="Measured"),
        Line2D([0], [0], color=ROLE["twin"], lw=LW, ls="--", label="Simulated twin"),
        Line2D([0], [0], color=PCB["blue"], marker="o", ls="", ms=3.2,
               label=r"Position at $t=8$\,s")], loc="outside lower center", ncol=3)
    if TWO:
        # Equal-aspect panels are height-limited: too tall a figure leaves slack between the
        # panels, too short a one narrows them. Bisect the figure height to the smallest value
        # at which the axes still span the same width as at a generous height, so the plots are
        # as wide as the other column figures (Fig. 6) and the vertical scale follows from that.
        w = fig.get_size_inches()[0]
        def axw(h):
            fig.set_size_inches(w, h); fig.canvas.draw()
            return axs[0].get_position().width
        full = axw(w * 0.9); lo, hi = w * 0.3, w * 0.9
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if axw(mid) < full - 1e-3 else (lo, mid)
        fig.set_size_inches(w, hi)
        print(f"  fig5: figure height {hi:.2f} in (aspect {hi / w:.3f}), axes width {full:.3f} of figure")
    save(fig, "fig5_fluidfit" if TWO else "fig5_fluidfit_grid", expect="col")
