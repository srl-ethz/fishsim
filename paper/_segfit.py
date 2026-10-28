#!/usr/bin/env python3
"""Windowed sinusoid fit -> (amplitude, frequency, bias) per 2 s segment.

One extractor, applied identically to the open-loop identification runs, the
deployed policy, and (later) the simulated policies, so the clouds are comparable
by construction.

Per window we fit          theta(t) ~ c + A sin(2 pi f t + phi)
For a FIXED f this is linear in (c, A cos phi, A sin phi), so we grid-search f and
solve a 3-parameter least squares at each -- no initialisation problem, no local
minima. Out of it we get, per window:

    A     stroke amplitude          [rad]
    f     stroke frequency          [Hz]
    c     turn bias / DC offset     [rad]   <- separated, not filtered away
    R2    how sinusoid-like the window actually was

R2 is the quality metric that makes the messy deploy runs honest: a parked tail or
a multi-scale wiggle simply fails it, and the fraction of windows that pass is
reported rather than hidden. This follows the reasoning of Cole & Voytek (2019),
who argue that oscillation features are meaningless where no oscillation is
evident -- see SOURCES.local.md for the full-read record and the scope limits.
"""
import numpy as np
from scipy.signal import butter, filtfilt

SEG = 2.0          # window length [s]; >= 2 cycles at the 1 Hz lower edge
HOP = 1.0          # 50 % overlap
FMIN, FMAX = 1.0, 8.0    # >= 2 cycles per 2 s window; also the identified lower edge
F_SPLIT = 1.0            # below this the motion is manoeuvre, not stroke
NF = 160           # frequency grid points
R2_MIN = 0.60      # a window must be this sinusoid-like to count


def fit_window(th, tt, fgrid):
    """Best (A, f, c, R2) for one window."""
    th = np.asarray(th, float); tt = np.asarray(tt, float) - tt[0]
    if len(th) < 8:
        return None
    ss_tot = float(np.sum((th - th.mean()) ** 2))
    if ss_tot <= 1e-12:
        return None
    best = None
    for f in fgrid:
        w = 2 * np.pi * f * tt
        M = np.column_stack([np.ones_like(tt), np.cos(w), np.sin(w)])
        coef, *_ = np.linalg.lstsq(M, th, rcond=None)
        res = th - M @ coef
        r2 = 1.0 - float(np.sum(res ** 2)) / ss_tot
        if best is None or r2 > best[0]:
            best = (r2, float(np.hypot(coef[1], coef[2])), float(f), float(coef[0]))
    r2, A, f, c = best
    return dict(A=A, f=f, bias=c, R2=r2)


def split_slow_fast(theta, t, f_split=F_SPLIT):
    """Separate slow manoeuvre from stroke.

    A single sinusoid fitted to the raw deployed signal locks onto the slow
    turn manoeuvre, not the stroke: lengthening the window drives the fitted
    frequency DOWN (72 %/87 %/94 % of windows below 1 Hz at 2/4/6 s), so the
    slow content dominates the variance at every scale. Both existing analyses
    band-pass for the same reason. We therefore fit the stroke on the fast
    residual and report the slow part separately rather than discarding it.
    """
    fs = 1.0 / np.median(np.diff(t))
    if f_split >= fs / 2:
        return np.zeros_like(theta), theta
    b, a = butter(2, f_split / (fs / 2), btype="low")
    slow = filtfilt(b, a, theta)
    return slow, theta - slow


def segment_fits(theta, t, seg=SEG, hop=HOP, fmin=FMIN, fmax=FMAX, nf=NF,
                 split=True):
    """All windows of one run. Returns a list of dicts (unfiltered)."""
    theta = np.asarray(theta, float); t = np.asarray(t, float)
    ok = np.isfinite(theta) & np.isfinite(t)
    theta, t = theta[ok], t[ok]
    if len(t) < 16 or t[-1] - t[0] < seg:
        return []
    if split:
        slow, theta = split_slow_fast(theta, t)
    else:
        slow = np.zeros_like(theta)
    fgrid = np.linspace(fmin, fmax, nf)
    out = []
    t0 = t[0]
    while t0 + seg <= t[-1]:
        m = (t >= t0) & (t < t0 + seg)
        if m.sum() >= 8:
            r = fit_window(theta[m], t[m], fgrid)
            if r is not None:
                r["t"] = float(t0 + seg / 2)
                r["slow"] = float(np.percentile(slow[m], 97.5)
                                  - np.percentile(slow[m], 2.5)) / 2
                out.append(r)
        t0 += hop
    return out


def summarise(fits, r2_min=R2_MIN):
    """Per-run summary over the windows that pass the quality gate."""
    good = [w for w in fits if w["R2"] >= r2_min]
    n = len(fits)
    if not good:
        return dict(n=n, kept=0, frac=0.0)
    A = np.array([w["A"] for w in good]); f = np.array([w["f"] for w in good])
    b = np.array([w.get("slow", w["bias"]) for w in good])
    C = np.cov(np.vstack([A, f])) if len(good) > 2 else np.zeros((2, 2))
    return dict(n=n, kept=len(good), frac=len(good) / n,
                A=float(np.median(A)), f=float(np.median(f)),
                A_sd=float(A.std()), f_sd=float(f.std()),
                Af=float(np.median(A * f)),
                bias=float(np.median(b)), bias_abs=float(np.median(np.abs(b))),
                R2=float(np.median([w["R2"] for w in good])), cov=C,
                A_all=A, f_all=f)
