#!/usr/bin/env python3
"""Where each gait puts its tail-velocity power, against the band the fluid was identified on.

Four conditions, all from the ACTUAL tail motion (encoder on hardware, joint angle in
simulation) rather than the commanded setpoint. The command is the wrong signal: it carries
the 50 Hz stair-stepping of the controller, which the motor removes and the fluid never sees.

Velocity rather than angle, because the quasi-steady fluid computes force from a body's
instantaneous velocity: V(f) = (2*pi*f)^2 * P_theta(f). Each spectrum is normalised to unit
power over 0.3-25 Hz, so the comparison is about where the energy sits, not how much there is.

Hardware content above about 10 Hz is mostly acquisition artefact rather than tail motion, and
its origin is timing, not quantisation: the measured floor sits ~190x above the ideal encoder
quantisation floor, which is why filtering on a quantisation argument was rejected. See
collect_deployment for what is corrected, what is not, and how both were verified. The hardware
fraction remains a lower bound.

Inputs
    fishdeploy/data/01_field_openloop_20260401/   open-loop identification runs
    upstream/sofi_control/rl_deploy/runs/                        deployment runs (encoder)
    data/sim_rollouts.npz                                        simulated TG and direct action
    data/sim_rollouts_lpf.npz                                    direct action behind a 4.5 Hz LPF

The filtered condition is what lets this figure carry the argument Table III used to make.
Note what it does and does not show: the filter's MEDIAN run is as clean as the trajectory
generator (96% in band against 98%), so the honest claim is not that filtering fails. It is
that filtering is unreliable -- its 10th-percentile run puts only 52% of velocity power in
band, where the generator's worst run still manages 93%. The lower edge of the shaded region
is the argument, not the line.

The filtered policy was trained against the refitB fluid and the identified motor, while the
deployed TG policy was trained against E13a and the near-ideal servo. Measurement is
like-for-like; what the policies were trained on is not. VI-D says so.
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from _paths import FIELD as _FIELD, DEPLOY_RUNS as _DEPLOY_RUNS, DATA as _DATA, paper_runs as _paper_runs
from paper_style import use_paper_style, size, save, CB, LW, EMPH_LW

FIELD = str(_FIELD)
DEPLOY = str(_DEPLOY_RUNS)
SIMROLL = str(_DATA / "sim_rollouts.npz")
SIMROLL_LPF = str(_DATA / "sim_rollouts_lpf.npz")

BAND = (1.0, 4.0)          # frequency range the fluid coefficients were identified over
FMIN, FMAX = 0.3, 25.0
LIN = np.linspace(FMIN, FMAX, 4000)   # linear grid: equal bins, so the integral is unbiased
WINDOW_S = 4.0             # Welch window in SECONDS, not samples: the sources run at
                           # 50 Hz (deployment, sim) and 1000 Hz (field), so a fixed sample
                           # count would give 0.2 Hz resolution on one and 3.9 Hz on the other
Q_STEP = 2 * np.pi / 4096  # Dynamixel encoder, one count [rad]


def velocity_psd(theta, dt):
    """Tail-velocity power spectral density, normalised to unit power over FMIN..FMAX."""
    x = np.asarray(theta, float)
    x = x - x.mean()
    if len(x) < 64:
        return None
    nperseg = min(int(round(WINDOW_S / dt)), len(x))
    f, P = welch(x, fs=1.0 / dt, nperseg=nperseg)
    V = (2 * np.pi * f) ** 2 * P
    m = (f >= FMIN) & (f <= FMAX)
    if V[m].sum() <= 0:
        return None
    return f[m], V[m] / np.trapezoid(V[m], f[m])


def pooled_cdf(runs):
    """Cumulative velocity power for a condition as a whole.

    Each run is normalised to unit power first, then the spectra are summed, so every run
    counts once regardless of length. This is not the median of the per-run curves: the
    identification set spans four drive frequencies with seven of its twelve runs at 2 Hz, and
    a median would be pinned by those seven and hide the 1, 3 and 4 Hz runs entirely.
    """
    tot = np.zeros_like(LIN)
    for f, V in runs:
        tot += np.interp(LIN, f, V)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (tot[1:] + tot[:-1]) * np.diff(LIN))])
    return LIN, c / c[-1]


def below_band_top(f, C):
    """Fraction of velocity power below the top of the identified band."""
    return np.interp(BAND[1], f, C) * 100


def collect_identification():
    out = []
    for d in sorted(glob.glob(f"{FIELD}/run_*")):
        p = os.path.join(d, "processed", "03_field2sim_ready.csv")
        if not os.path.exists(p):
            continue
        c = pd.read_csv(p)
        col = next((k for k in ("Ghost_Alpha", "motor_pos", "alpha") if k in c.columns), None)
        tcol = next((k for k in ("Time", "time", "t", "pi_time") if k in c.columns), None)
        if col is None or tcol is None:
            continue
        t = c[tcol].to_numpy()
        r = velocity_psd(c[col].to_numpy(), float(np.median(np.diff(t))))
        if r:
            out.append(r)
    return out


def collect_deployment():
    """Deployment encoder angle, resampled onto a uniform grid using the logged timestamps.

    The deployment loop does not run on a steady clock: the period is 20.1 ms with a standard
    deviation of 5.4 ms, 27% of the period. Treating those samples as evenly spaced files each
    reading at the wrong time, and a timing error eps on a moving tail is an angle error of
    about theta_dot * eps -- at 2 Hz and a half-radian stroke, ~32 mrad, or twenty encoder
    counts. That error is broadband, and the (2*pi*f)^2 velocity weighting then amplifies it
    exactly where it depresses the in-band fraction.

    The identification runs need no such correction here because the field pipeline already
    applies it: 03_prep_field2sim.py fits a cubic spline against the recorded timestamps and
    resamples to 1 kHz. Its raw log is the same ~54 Hz motor read with the same jitter. So this
    is not a correction applied to one condition and withheld from the others; it puts the
    deployment path on the footing the identification path already had. Simulation is exempt
    because its timestep is exact by construction.

    Verified on signals whose answer is known: pushing clean simulated rollouts through a model
    of the real acquisition chain (free-running encoder thread, two-deep FIFO, jittered
    consumer -- see rl_deploy/pi/motor_runner.py) biases the in-band fraction down by 9.9
    points, of which this resamples back 9.5.

    One residual is not removable from the log. The encoder is read by a background thread and
    delivered through a queue, so pi_time is the time the control step began, not the time the
    angle was sampled; the sample is of order one thread period (~6 ms) old. That lag is nearly
    constant, which is why it costs little, but it is why the hardware number stays a lower
    bound.
    """
    return [velocity_psd(a, dt) for a, dt in deployment_angles()
            if velocity_psd(a, dt) is not None]


def deployment_angles():
    """Cleaned, uniformly-resampled tail angle for each deployment run: (angle, dt)."""
    for p in [str(d / "raw" / "run.csv") for d in _paper_runs()]:   # the 43 runs of the paper
        c = pd.read_csv(p)
        if len(c) < 300 or not {"obs_00", "obs_01", "pi_time"} <= set(c.columns):
            continue
        t = c["pi_time"].to_numpy()
        enc = np.unwrap(np.arctan2(c["obs_01"].to_numpy(), c["obs_00"].to_numpy()))
        # get_state() returns a hardcoded pos_rad=0.0 when the queue is empty. Those are not
        # measurements and they are step discontinuities, so drop them -- but only when
        # isolated, since the tail genuinely passes through zero twice a cycle.
        tick = np.round(enc / Q_STEP).astype(int)
        drop = np.zeros(len(tick), bool)
        inner = np.arange(1, len(tick) - 1)
        drop[inner] = (tick[inner] == 0) & (np.maximum(np.abs(tick[inner - 1]),
                                                       np.abs(tick[inner + 1])) > 8)
        t, enc = t[~drop], enc[~drop]
        dt = float(np.median(np.diff(t)))
        yield np.interp(np.arange(t[0], t[-1], dt), t, enc), dt


def deployment_beat():
    """Stroke fundamental of each deployment run [Hz]: the peak of the ANGLE spectrum.

    Reported in the paper as "about 1.3 Hz". Quoted to one decimal deliberately. The Welch
    bin spacing is 1/WINDOW_S, so any single run's peak is pinned to a coarse grid and the
    median moves with the window: 1.25 Hz at 4 s, 1.17 at 6 s, 1.37 at 8 s, 1.27 at 16 s.
    The claim is stable to that choice, the second decimal is not.

    This supersedes an earlier "peaks at 1.75 Hz", which was the COMMANDED setpoint of a
    single run (hope_didnt_work_2) from report_scripts/deploy_plots.py, plotted in the since
    removed 20d_spectrum_firstever.pdf. It was true of that run and did not generalise.
    """
    out = []
    for a, dt in deployment_angles():
        x = a - a.mean()
        f, P = welch(x, fs=1.0 / dt, nperseg=min(int(round(WINDOW_S / dt)), len(x)))
        m = (f >= 0.8) & (f <= 6.0)          # the stroke band, above turn/drift, below noise
        if P[m].size:
            out.append(f[m][np.argmax(P[m])])
    return np.array(out)


def collect_sim(key):
    z = np.load(SIMROLL, allow_pickle=True)
    dt = float(z[f"{key}_dt"])
    return [r for r in (velocity_psd(tr, dt) for tr in z[f"{key}_meas"]) if r]


def collect_sim_lpf(n=25):
    """Direct action behind a first-order 4.5 Hz setpoint filter.

    Capped at n episodes so every simulated condition contributes the same count; the
    generator produced 40, the other two have 25.
    """
    z = np.load(SIMROLL_LPF, allow_pickle=True)
    dt = float(z["lpf_dt"])
    return [r for r in (velocity_psd(tr, dt) for tr in z["lpf_meas"][:n]) if r]


# The controlled comparison is simulation against simulation: identification, the trajectory generator, and
# the unconstrained policy, all measured the same way. The deployed hardware run answers a
# different question with a different instrument, so it is off by default; --with-hardware
# adds it.
WITH_HW = "--with-hardware" in sys.argv

CONDITIONS = [
    ("Identification", CB["black"], collect_identification),
    ("Trajectory generator", CB["blue"], lambda: collect_sim("cpg")),
    ("Direct action + filter", CB["green"], collect_sim_lpf),
    ("Direct action", CB["purple"], lambda: collect_sim("da")),
]
if WITH_HW:
    CONDITIONS.insert(2, ("Deployed hardware", CB["orange"], collect_deployment))

use_paper_style()
fig, ax = plt.subplots(figsize=size(1.0, aspect=0.66), layout="constrained")
ax.axvspan(*BAND, color=CB["blue"], alpha=0.10, lw=0, zorder=0)

for label, colour, getter in CONDITIONS:
    runs = getter()
    # 10-90% spread across runs, so the pooled line is not read as a single trace
    per = np.vstack([np.interp(LIN, f,
                     np.concatenate([[0.0], np.cumsum(0.5 * (V[1:] + V[:-1]) * np.diff(f))]) /
                     np.trapezoid(V, f)) for f, V in runs])
    lo, hi = np.percentile(per, [10, 90], axis=0)
    ax.fill_between(LIN, lo, hi, color=colour, alpha=0.11, lw=0, zorder=2)
    g, C = pooled_cdf(runs)
    ax.plot(g, C, color=colour, lw=LW, zorder=3,
            label=f"{label} ({below_band_top(g, C):.0f}\\%)")

ax.set_xscale("log")
ax.set_xlim(FMIN, FMAX)
ax.set_ylim(0, 1.02)
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Cumulative tail velocity" "\n" "power fraction")
for f_edge in BAND:                    # both edges of the identified band, dotted
    ax.axvline(f_edge, color=CB["blue"], lw=0.8, ls=":", zorder=1)
# band label: white type on a solid tag in the band's colour, at the top of the band
ax.text(np.sqrt(BAND[0] * BAND[1]), 0.975, "Identified band", color="white",
        ha="center", va="top", zorder=8,
        bbox=dict(boxstyle="round,pad=0.25", fc=CB["blue"], ec="none"))
fig.legend(loc="outside lower center", ncol=2, frameon=False, fontsize=6.5,
           handlelength=1.2, columnspacing=0.8, handletextpad=0.4)

save(fig, "fig9_velspectrum_lpf" if WITH_HW else "fig9_velspectrum_lpf_sim", expect="col")

for label, _, getter in CONDITIONS:
    runs = getter()
    g, C = pooled_cdf(runs)
    per = []
    for f, V in runs:
        c = np.concatenate([[0.0], np.cumsum(0.5 * (V[1:] + V[:-1]) * np.diff(f))])
        per.append(np.interp(BAND[1], f, c / c[-1]) * 100)
    print(f"  {label:24s} n={len(runs):3d}   pooled {below_band_top(g, C):5.1f}%   "
          f"p10 {np.percentile(per, 10):5.1f}%   median {np.median(per):5.1f}%   "
          f"p90 {np.percentile(per, 90):5.1f}%")

b = deployment_beat()
print(f"\n  deployed tail beat (angle fundamental)  n={len(b):3d}   median {np.median(b):.2f} Hz   "
      f"IQR {np.percentile(b, 25):.2f}-{np.percentile(b, 75):.2f}   "
      f"{np.mean(b < 4) * 100:.0f}% of runs below 4 Hz")
