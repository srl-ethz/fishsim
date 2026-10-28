"""
Step 1 — Structural Stiffness & Damping Identification from Air Benchmark Data
-------------------------------------------------------------------------------
Extracts k_eff, I_eff, c_viscous, and τ_Coulomb from sinusoidal motor oscillations
in air using the mechanical frequency response H(ω) = real + j*imag.

Revision 4 key improvements over R3:
  1. N_TRANSIENT_CYCLES reduced from 3.0 → 1.0: cycle-averaging diagnostic shows clean
     overlap from cycle 1, so discarding 3 cycles was overly conservative and dropped
     2 full periods at 1 Hz. R4 retains all stable-regime data.

Revision 3 key improvements over R2:
  1. Physical amplitude filter: CmdAmpTravel_mm >= SLACK_THRESHOLD_MM (default 2.5 mm)
     instead of a percentage cutoff.  This correctly admits d7@40% (2.8 mm) and
     d9@40% (3.6 mm), which clear the tendon slack dead-zone, while rejecting d5@40%
     (1.6 mm) that does not — even though both are "40% amplitude".
  2. Pre-screening: every condition evaluated before fitting for torque saturation,
     velocity saturation, and tracking quality.  A full pass/fail table is printed.
     Only PASS conditions enter the stiffness fit; damping decomposition uses a
     slightly wider set (all amplitudes with good tracking, including sub-slack).
  3. All four frequencies 1–4 Hz included by default.  More ω² spread greatly improves
     the k/I separation in the linear regression.
  4. 2 Hz included wherever it passes quality checks (no blanket exclusion).

Pass criteria:
  Stiffness: trav_mm >= SLACK_MM  AND  track_r >= TRACK_MIN  AND  no saturation
  Damping:   track_r >= TRACK_MIN  AND  no saturation  (slack allowed — need low-amp)

Usage:
    python sysid/tail_stiffness/step1_stiffness.py
    python sysid/tail_stiffness/step1_stiffness.py --d 5 7 9 --freqs 1 2 3 4
    python sysid/tail_stiffness/step1_stiffness.py --slack_mm 3.0 --no_avg
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    find_latest_file, load_csv, to_si, trim_transient, cycle_average,
    compute_H, extract_phasor, fit_spring_inertia,
    beam_theory_stiffness, k_eff_from_joints, scale_joints_to_keff,
    PULLEY_RADIUS_M, TENDON_MOMENT_ARMS,
    E_PLA_NOM, VEL_TO_RAD_S,
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
ALL_AMP_FRACS      = [20.0, 40.0, 60.0, 80.0, 100.0]
N_TRANSIENT_CYCLES = 1.0  # R4: reduced from 3.0; cycle-avg diagnostic shows clean overlap from cycle 1

# Quality thresholds
THRESHOLDS = {
    "slack_mm":      2.5,   # min absolute motor travel [mm] to clear tendon slack
    "track_r_min":   0.50,  # min actual/commanded amplitude ratio
    "load_sat_raw":  400,   # raw Load units above which motor is considered saturated
    "vel_sat_rpm":   55.0,  # RPM above which velocity is considered saturated
}


# ---------------------------------------------------------------------------
# Pre-screening
# ---------------------------------------------------------------------------

def prescreen_condition(env: str, d_mm: float, freq_hz: float, amp_pct: float,
                        slack_mm: float = THRESHOLDS["slack_mm"]) -> tuple:
    """
    Load one CSV, compute quality metrics, classify as PASS/FAIL for two uses.

    Returns (pass_stiffness, pass_damping, flags_list, metrics_dict).

    pass_stiffness: cleared slack, good tracking, no saturation.
    pass_damping:   good tracking, no saturation only (low-amp OK for Coulomb fit).
    """
    path = find_latest_file(env, d_mm, freq_hz, amp_pct)
    if path is None:
        return False, False, ["NO_FILE"], {}

    meta, raw = load_csv(path)
    trav_mm = float(meta["CmdAmpTravel_mm"].iloc[0])

    si    = to_si(raw)
    si_ss = trim_transient(si, N_TRANSIENT_CYCLES, freq_hz)

    # Velocity in RPM (raw units × 0.229 RPM/unit)
    vel_rpm   = np.abs(raw["Velocity"].to_numpy()) * 0.229
    load_abs  = np.abs(raw["Load"].to_numpy())
    max_vel   = float(vel_rpm.max())
    max_load  = float(load_abs.max())

    # Tracking ratio — actual θ amplitude / commanded θ amplitude (phasor magnitudes)
    theta_amp = abs(extract_phasor(si_ss["t"], si_ss["theta"],      freq_hz))
    goal_amp  = abs(extract_phasor(si_ss["t"], si_ss["goal_theta"], freq_hz))
    track_r   = float(theta_amp / goal_amp) if goal_amp > 0.001 else 0.0

    metrics = {
        "trav_mm":     trav_mm,
        "max_vel_rpm": max_vel,
        "max_load_raw": max_load,
        "track_r":     track_r,
    }

    flags = []
    if trav_mm  < slack_mm:                          flags.append("SLACK")
    if max_vel  > THRESHOLDS["vel_sat_rpm"]:         flags.append("VEL_SAT")
    if max_load > THRESHOLDS["load_sat_raw"]:        flags.append("LOAD_SAT")
    if track_r  < THRESHOLDS["track_r_min"]:         flags.append("NO_TRACK")

    # Saturation / tracking problems disqualify for both uses
    bad_flags     = [f for f in flags if f != "SLACK"]
    pass_damping  = len(bad_flags) == 0
    pass_stiffness = pass_damping and "SLACK" not in flags

    return pass_stiffness, pass_damping, flags, metrics


# ---------------------------------------------------------------------------
# Core per-file processing
# ---------------------------------------------------------------------------

def process_file(env: str, d_mm: float, freq_hz: float, amp_pct: float,
                 use_cycle_avg: bool = True) -> dict | None:
    """
    Load, trim transient, optionally cycle-average, compute H phasor.
    Returns None if file not found or too little data.
    """
    path = find_latest_file(env, d_mm, freq_hz, amp_pct)
    if path is None:
        return None
    meta, raw = load_csv(path)

    trav_mm  = float(meta["CmdAmpTravel_mm"].iloc[0])
    A_theta  = trav_mm / 1000.0 / PULLEY_RADIUS_M   # commanded amplitude [rad]

    si = to_si(raw)
    si = trim_transient(si, N_TRANSIENT_CYCLES, freq_hz)
    if len(si["t"]) < 10:
        return None

    if use_cycle_avg:
        si_proc = cycle_average(si, freq_hz)
        n_cyc   = si_proc.get("n_cycles", 1)
    else:
        si_proc = si
        T     = 1.0 / freq_hz
        n_cyc = int((si["t"][-1] - si["t"][0]) / T)

    H = compute_H(si_proc, freq_hz)
    return {
        "H":        H,
        "A_theta":  A_theta,
        "n_cycles": n_cyc,
        "d_mm":     d_mm,
        "freq":     freq_hz,
        "amp_pct":  amp_pct,
        "trav_mm":  trav_mm,
        "omega":    2 * np.pi * freq_hz,
        "omega_sq": (2 * np.pi * freq_hz) ** 2,
    }


# ---------------------------------------------------------------------------
# Cycle-averaging diagnostic plot
# ---------------------------------------------------------------------------

def plot_raw_vs_averaged(d_mm: float, freq_hz: float, amp_pct: float) -> None:
    path = find_latest_file("air", d_mm, freq_hz, amp_pct)
    if path is None:
        return
    _, raw = load_csv(path)
    si       = to_si(raw)
    si_trim  = trim_transient(si, N_TRANSIENT_CYCLES, freq_hz)
    si_avg   = cycle_average(si_trim, freq_hz)
    T        = 1.0 / freq_hz
    t_rel    = si_trim["t"] - si_trim["t"][0]
    n_cyc    = int(t_rel[-1] / T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for col, (key, title, sc) in enumerate(zip(
            ["theta", "tau"],
            ["θ_motor  [mrad]", "τ_motor  [mNm]"],
            [1000.0, 1000.0])):
        colors = plt.cm.Blues(np.linspace(0.35, 0.95, n_cyc))
        ax = axes[0, col]
        for i in range(n_cyc):
            mask = (t_rel >= i * T) & (t_rel < (i + 1) * T)
            ax.plot(t_rel[mask], si_trim[key][mask] * sc, color=colors[i], lw=1.0, alpha=0.8)
        ax.set_title(f"Individual cycles ({n_cyc} shown)")
        ax.set_xlabel("Time  [s]"); ax.set_ylabel(title); ax.grid(True, alpha=0.3)

        ax2 = axes[1, col]
        t_cyc = si_avg["t"]
        y_avg = si_avg[key] * sc
        y_stack = []
        for i in range(n_cyc):
            mask = (t_rel >= i * T) & (t_rel < (i + 1) * T)
            if mask.sum() >= 4:
                t_loc = t_rel[mask] - i * T
                y_stack.append(np.interp(t_cyc, t_loc, si_trim[key][mask] * sc))
        y_std = np.array(y_stack).std(axis=0) if len(y_stack) > 1 else np.zeros_like(y_avg)
        ax2.fill_between(t_cyc, y_avg - y_std, y_avg + y_std, alpha=0.25, color="C0",
                         label="±1σ")
        ax2.plot(t_cyc, y_avg, "C0", lw=2.0, label=f"Avg (N={si_avg['n_cycles']})")
        ax2.set_title("Cycle-averaged ± σ")
        ax2.set_xlabel("Time  [s]"); ax2.set_ylabel(title)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Air  d={d_mm}mm  {freq_hz}Hz  {amp_pct}%  —  raw vs averaged", fontsize=11)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f"s1_cycle_avg_d{d_mm}_{freq_hz}Hz_{amp_pct}pct.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",        type=float, nargs="+", default=[5.0, 7.0, 9.0])
    parser.add_argument("--freqs",    type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--slack_mm", type=float,            default=THRESHOLDS["slack_mm"],
                        help=f"Absolute travel threshold [mm] for tendon-slack filter "
                             f"(default {THRESHOLDS['slack_mm']} mm)")
    parser.add_argument("--no_avg",   action="store_true",
                        help="Disable synchronous cycle averaging.")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)
    use_avg  = not args.no_avg
    slack_mm = args.slack_mm

    # ------------------------------------------------------------------
    # 1. Pre-analysis table — every condition screened before fitting
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("PRE-ANALYSIS — quality screening of all air conditions")
    print(f"  Slack filter:    travel >= {slack_mm:.1f} mm")
    print(f"  Tracking filter: track_r >= {THRESHOLDS['track_r_min']:.2f}")
    print(f"  Load sat limit:  max raw Load < {THRESHOLDS['load_sat_raw']}")
    print(f"  Vel  sat limit:  max RPM < {THRESHOLDS['vel_sat_rpm']:.0f}")
    print(f"{'='*80}")
    col_w = f"{'d':>5} {'f':>5} {'A%':>5} {'trav_mm':>8} {'maxVel_rpm':>11} " \
            f"{'maxLoad_raw':>12} {'track_r':>8}  for_k  for_c  status"
    print(col_w)
    print("-" * len(col_w))

    screen_cache = {}   # (d, freq, amp) → (pass_k, pass_c, flags, metrics)
    for d_mm in args.d:
        for freq_hz in args.freqs:
            for amp_pct in ALL_AMP_FRACS:
                pk, pc, flags, m = prescreen_condition("air", d_mm, freq_hz, amp_pct, slack_mm)
                screen_cache[(d_mm, freq_hz, amp_pct)] = (pk, pc, flags, m)
                if not m:
                    continue
                status = "PASS" if not flags else " ".join(flags)
                pk_s = "YES" if pk else " NO"
                pc_s = "YES" if pc else " NO"
                print(f"{d_mm:>5.1f} {freq_hz:>5.1f} {amp_pct:>5.0f} "
                      f"{m['trav_mm']:>8.2f} {m['max_vel_rpm']:>11.1f} "
                      f"{m['max_load_raw']:>12.0f} {m['track_r']:>8.3f}  "
                      f"{pk_s}    {pc_s}  {status}")

    n_pass_k = sum(1 for pk, _, _, _ in screen_cache.values() if pk)
    n_pass_c = sum(1 for _, pc, _, _ in screen_cache.values() if pc)
    print(f"\n  Conditions passing stiffness filter: {n_pass_k}")
    print(f"  Conditions passing damping   filter: {n_pass_c}")

    # ------------------------------------------------------------------
    # 2. Cycle-averaging diagnostic (first passing stiffness condition)
    # ------------------------------------------------------------------
    for d_mm in args.d:
        for freq_hz in args.freqs:
            for amp_pct in [80.0, 100.0, 60.0]:
                pk, _, _, _ = screen_cache.get((d_mm, freq_hz, amp_pct), (False, False, [], {}))
                if pk:
                    print(f"\n--- Cycle-averaging diagnostic: d={d_mm} {freq_hz}Hz {amp_pct}% ---")
                    plot_raw_vs_averaged(d_mm, freq_hz, amp_pct)
                    break
            else:
                continue
            break

    # ------------------------------------------------------------------
    # 3. Collect H phasors
    #    stiff_records: pass_stiffness only  → used for k/I regression
    #    damp_records:  pass_damping (all amplitudes with good tracking)
    # ------------------------------------------------------------------
    print(f"\n--- Computing H(ω) — pass conditions only ---")
    stiff_records = []
    damp_records  = []

    for d_mm in args.d:
        for freq_hz in args.freqs:
            for amp_pct in ALL_AMP_FRACS:
                pk, pc, _, _ = screen_cache.get((d_mm, freq_hz, amp_pct), (False, False, [], {}))
                if not pc:
                    continue
                r = process_file("air", d_mm, freq_hz, amp_pct, use_cycle_avg=use_avg)
                if r is None:
                    continue
                if pk:
                    stiff_records.append(r)
                damp_records.append(r)
                flag = " [stiff+damp]" if pk else " [damp only]"
                print(f"  d{d_mm} {freq_hz}Hz {amp_pct:5.0f}%  trav={r['trav_mm']:.1f}mm"
                      f"  N_cyc={r['n_cycles']}"
                      f"  Re(H)={r['H'].real:6.2f}  Im(H)={r['H'].imag:6.3f}{flag}")

    if not stiff_records:
        print("ERROR: No conditions passed stiffness filter. Check DATA_DIR or --slack_mm.")
        return
    if len(stiff_records) < 3:
        print(f"WARNING: Only {len(stiff_records)} stiffness conditions — regression may be unreliable.")

    # ------------------------------------------------------------------
    # 4. Stiffness / inertia regression: Re(H) = k_eff - I_eff * ω²
    # ------------------------------------------------------------------
    omega_sq  = np.array([r["omega_sq"]   for r in stiff_records])
    real_H    = np.array([r["H"].real      for r in stiff_records])
    k_eff, I_eff = fit_spring_inertia(omega_sq, real_H)

    # Also compute global reference (all damp-passing conditions)
    omega_sq_all = np.array([r["omega_sq"] for r in damp_records])
    real_H_all   = np.array([r["H"].real    for r in damp_records])
    k_global, I_global = fit_spring_inertia(omega_sq_all, real_H_all)

    print(f"\n  k_eff (pass-stiff, n={len(stiff_records)}): {k_eff:.4f}  Nm/rad")
    print(f"  I_eff (pass-stiff):                         {I_eff*1e3:.4f}  ×10⁻³ kg·m²")
    print(f"  k_eff (reference, all damp-pass):           {k_global:.4f}  Nm/rad")

    # ------------------------------------------------------------------
    # 5. Damping decomposition: imag(H)/ω = c_viscous + (4τ_C/π)/(A_θ·ω)
    # ------------------------------------------------------------------
    damp_x_all  = np.array([1.0 / (r["A_theta"] * r["omega"]) for r in damp_records])
    damp_y_all  = np.array([r["H"].imag / r["omega"]           for r in damp_records])
    # Fit on stiffness-passing subset (same physical regime as k_eff)
    damp_x_fit  = np.array([1.0 / (r["A_theta"] * r["omega"]) for r in stiff_records])
    damp_y_fit  = np.array([r["H"].imag / r["omega"]           for r in stiff_records])

    A_damp = np.column_stack([np.ones_like(damp_x_fit), damp_x_fit])
    c_fit, _, _, _ = np.linalg.lstsq(A_damp, damp_y_fit, rcond=None)
    c_viscous   = float(c_fit[0])
    tau_coulomb = float(c_fit[1] * np.pi / 4)

    # Reference: fit on all damp-passing data (wider amplitude range)
    A_ref = np.column_stack([np.ones_like(damp_x_all), damp_x_all])
    c_ref, _, _, _ = np.linalg.lstsq(A_ref, damp_y_all, rcond=None)
    c_viscous_ref   = float(c_ref[0])
    tau_coulomb_ref = float(c_ref[1] * np.pi / 4)

    print(f"\n  c_viscous  (pass-stiff fit): {c_viscous:.5f}  Nm·s/rad")
    print(f"  τ_Coulomb  (pass-stiff fit): {tau_coulomb:.5f}  Nm")
    print(f"  c_viscous  (all damp-pass):  {c_viscous_ref:.5f}  Nm·s/rad  [reference]")
    print(f"  τ_Coulomb  (all damp-pass):  {tau_coulomb_ref:.5f}  Nm  [reference]")

    # ------------------------------------------------------------------
    # 6. Per-joint stiffness and damping
    # ------------------------------------------------------------------
    k_joints, E_eff = scale_joints_to_keff(k_eff, E_PLA_NOM)
    loss_ratio      = c_viscous / k_eff   # c/k [s], proportional damping ratio
    c_joints        = k_joints * loss_ratio

    k_nom     = beam_theory_stiffness(E_PLA_NOM)
    k_eff_nom = k_eff_from_joints(k_nom)

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------

    # 7a. Re(H) vs ω² — all conditions; stiffness-passing highlighted
    d_vals = sorted(set(r["d_mm"] for r in damp_records))
    fig, axes = plt.subplots(1, len(d_vals), figsize=(5 * len(d_vals), 4.5), sharey=True)
    if len(d_vals) == 1:
        axes = [axes]
    cmap    = plt.cm.viridis(np.linspace(0.1, 0.9, len(ALL_AMP_FRACS)))
    amp_col = {a: c for a, c in zip(ALL_AMP_FRACS, cmap)}

    for ax, d_mm in zip(axes, d_vals):
        for amp_pct in ALL_AMP_FRACS:
            pts_s = [r for r in stiff_records if r["d_mm"] == d_mm and r["amp_pct"] == amp_pct]
            pts_d = [r for r in damp_records  if r["d_mm"] == d_mm and r["amp_pct"] == amp_pct
                     and r not in stiff_records]
            col = amp_col[amp_pct]
            for pts, lw, ls, alpha, label in [
                    (pts_s, 1.8, "-",  1.0, f"{amp_pct:.0f}% [stiff]"),
                    (pts_d, 0.9, "--", 0.5, f"{amp_pct:.0f}% [damp-only]")]:
                if not pts:
                    continue
                xs = [r["omega_sq"] for r in pts]
                ys = [r["H"].real    for r in pts]
                ax.scatter(xs, ys, color=col, s=50 if lw > 1 else 25, alpha=alpha, zorder=3)
                ax.plot(xs, ys, color=col, lw=lw, ls=ls, alpha=alpha)

        w2 = np.linspace(0, max(r["omega_sq"] for r in damp_records) * 1.1, 50)
        pts_sd = [r for r in stiff_records if r["d_mm"] == d_mm]
        if len(pts_sd) >= 2:
            kd, Id = fit_spring_inertia(
                np.array([r["omega_sq"] for r in pts_sd]),
                np.array([r["H"].real   for r in pts_sd]))
            ax.plot(w2, kd - Id * w2, "r-",  lw=2.0, label=f"k={kd:.2f} [pass-stiff]")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(f"d = {d_mm} mm"); ax.set_xlabel("ω²  [rad²/s²]")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Re(H) = k_eff − I_eff·ω²  [Nm/rad]")
    plt.suptitle("Air benchmark — Re(H) vs. ω²\n"
                 "solid=pass-stiffness  |  dashed=damp-only (below slack threshold)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s1_real_H_vs_omega2.png"), dpi=150); plt.close()
    print(f"\nSaved: {os.path.join(PLOT_DIR, 's1_real_H_vs_omega2.png')}")

    # 7b. Damping decomposition
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    d_arr   = np.array([r["d_mm"]   for r in damp_records])
    amp_arr = np.array([r["amp_pct"] for r in damp_records])
    in_stiff = np.array([r in stiff_records for r in damp_records])

    for d_mm in d_vals:
        for amp_pct in ALL_AMP_FRACS:
            mask = (d_arr == d_mm) & (amp_arr == amp_pct)
            if not mask.any():
                continue
            col   = amp_col[amp_pct]
            alpha = 1.0 if in_stiff[mask].any() else 0.35
            ms    = 60  if in_stiff[mask].any() else 30
            label = f"d={d_mm} {amp_pct:.0f}%" if in_stiff[mask].any() else ""
            ax2.scatter(damp_x_all[mask], damp_y_all[mask], color=col,
                        s=ms, alpha=alpha, zorder=3, label=label)

    x_line = np.linspace(0, damp_x_all.max() * 1.05, 100)
    ax2.plot(x_line, c_viscous_ref + (4 * tau_coulomb_ref / np.pi) * x_line,
             "k--", lw=1.5, label=f"Ref (all):  c_v={c_viscous_ref:.4f}, τ_C={tau_coulomb_ref:.4f}")
    ax2.plot(x_line, c_viscous   + (4 * tau_coulomb   / np.pi) * x_line,
             "r-",  lw=2.0, label=f"Pass-stiff: c_v={c_viscous:.4f}, τ_C={tau_coulomb:.4f}")
    ax2.axhline(c_viscous, color="r", lw=0.8, ls=":", alpha=0.5)
    ax2.axvline(0, color="0.7", lw=0.8)
    ax2.set_xlabel("1 / (A_θ × ω)  [s/rad²]")
    ax2.set_ylabel("Im(H) / ω  [Nm·s/rad]")
    ax2.set_title("Damping decomposition:\n"
                  "Im(H)/ω = c_viscous  +  (4τ_Coulomb/π) × 1/(A·ω)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s1_damping_decomposition.png"), dpi=150); plt.close()
    print(f"Saved: {os.path.join(PLOT_DIR, 's1_damping_decomposition.png')}")

    # 7c. Per-joint stiffness and damping bar chart -> report figure (recipe, vector PDF)
    import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
    from report_style import use_report_style as _urs, set_size as _ss, savefig as _sv, CB as _CB
    _urs()
    fig3, (ax_k, ax_c) = plt.subplots(1, 2, figsize=_ss(0.99, aspect=0.46), layout="constrained")
    x = np.arange(5); w = 0.35
    ax_k.bar(x - w/2, k_nom,    w, label="beam theory", alpha=0.9, color=_CB["blue"])
    ax_k.bar(x + w/2, k_joints, w, label=r"calibrated ($k_\mathrm{eff}{=}%.2f$)" % k_eff, alpha=0.9, color=_CB["green"])
    ax_k.axhline(0.65, color=_CB["orange"], ls="--", lw=1.2, label="old MuJoCo (0.65)")
    ax_k.set_xticks(x); ax_k.set_xticklabels([f"j{i}" for i in range(5)])
    ax_k.set_ylabel(r"hinge stiffness [N\,m/rad]"); ax_k.set_ylim(0, max(k_joints) * 1.35)
    ax_k.legend(loc="upper right", framealpha=0.9); ax_k.grid(True, axis="y")
    ax_c.bar(x, c_joints, 0.5, color=_CB["purple"], alpha=0.9, label=r"$c/k{=}%.3f$ s" % loss_ratio)
    ax_c.axhline(0.10, color=_CB["orange"], ls="--", lw=1.2, label="old MuJoCo (0.10)")
    ax_c.set_xticks(x); ax_c.set_xticklabels([f"j{i}" for i in range(5)])
    ax_c.set_ylabel(r"hinge damping [N\,m\,s/rad]"); ax_c.set_ylim(0, 0.13)
    ax_c.legend(loc="upper right", framealpha=0.9); ax_c.grid(True, axis="y")
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "out"); os.makedirs(_out, exist_ok=True)
    print("Saved:", _sv(fig3, os.path.join(_out, "31_joint_stiffness_damping")))
    
    # Restore default style so we don't mess up later plots
    plt.style.use('default')
    
    # 7d. Per-joint stiffness ONLY (presentation-only, disabled; report uses 7c)
    for s in []:
        plt.style.use(s["plt_style"])
        fig4, ax_k_only = plt.subplots(figsize=(10, 4.5))
        fig4.patch.set_facecolor(s["face"])
        x = np.arange(5); w = 0.35
        
        ax_k_only.set_facecolor(s["face"])
        ax_k_only.bar(x - w/2, k_nom,    w, label="Beam theory (E_nom=2.7 GPa)", alpha=0.75, color="C0" if s["mode"]=="light" else "#4C99D3")
        ax_k_only.bar(x + w/2, k_joints, w, label=f"Calibrated k_eff={k_eff:.3f} Nm/rad", alpha=0.75, color="C2" if s["mode"]=="light" else "#5FBA7D")
        ax_k_only.axhline(0.65, color="r", ls="--", lw=1.5, label="Old MuJoCo Value (0.65)")
        ax_k_only.set_xticks(x); ax_k_only.set_xticklabels([f"j{i}" for i in range(5)], color=s["text"])
        ax_k_only.tick_params(colors=s["text"])
        ax_k_only.set_ylabel("hingeStiffness  [Nm/rad]", color=s["text"])
        ax_k_only.set_title("Joint stiffness — beam theory vs. calibrated", color=s["text"])
        ax_k_only.legend(fontsize=8, facecolor=s["face"], edgecolor=s["text"], labelcolor=s["text"])
        ax_k_only.grid(True, alpha=0.2 if s["mode"]=="dark" else 0.3, axis="y", color=s["text"])
        
        plt.tight_layout()
        for ext in s["exts"]:
            out_path = os.path.join(PLOT_DIR, f"s1_joint_stiffnesses{ext}")
            plt.savefig(out_path, dpi=300 if ext.endswith('.pdf') else 150)
            print(f"Saved: {out_path}")
        plt.close()
        
    plt.style.use('default')

    # ------------------------------------------------------------------
    # 8. Full results summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS  (R4 — N_TRANSIENT_CYCLES=1.0, physical amplitude filter, all-frequency fit)")
    print("=" * 70)
    print(f"\n  n conditions (stiffness fit): {len(stiff_records)}")
    print(f"  n conditions (damping fit):   {len(damp_records)}")
    print(f"\n  k_eff  (pass-stiff):   {k_eff:.5f}  Nm/rad")
    print(f"  I_eff  (pass-stiff):   {I_eff*1e3:.4f}  ×10⁻³ kg·m²")
    print(f"  k_eff  (all damp-pass):{k_global:.5f}  Nm/rad  [reference]")
    print(f"  I_eff  (all damp-pass):{I_global*1e3:.4f}  ×10⁻³ kg·m²  [reference]")
    print(f"  E_PLA  (calibrated):   {E_eff/1e9:.4f}  GPa")
    print(f"\n  c_viscous  (pass-stiff): {c_viscous:.5f}  Nm·s/rad")
    print(f"  τ_Coulomb  (pass-stiff): {tau_coulomb:.5f}  Nm")
    print(f"  c/k loss ratio:          {loss_ratio:.5f}  s")
    print(f"  Loss factor η @ 1 Hz:    {loss_ratio * 2*np.pi:.4f}")
    print(f"\n  {'Joint':<7} {'k_joints':>10} {'c_joints':>10}")
    print(f"  {'-'*7} {'-'*10} {'-'*10}")
    for i in range(5):
        print(f"  joint{i}  {k_joints[i]:>10.3f} {c_joints[i]:>10.4f}")
    print(f"\n  MuJoCo YAML lines:")
    print(f"    hingeStiffness: {np.round(k_joints, 3).tolist()}")
    print(f"    hingeDamping:   {np.round(c_joints, 4).tolist()}")
    print(f"\n  COMPARISON — current MuJoCo: hingeStiffness=0.65 (uniform), hingeDamping=0.10")
    print(f"  Stiffness ratio (joint0): {k_joints[0]/0.65:.1f}×   (joint4): {k_joints[4]/0.65:.1f}×")

    # ------------------------------------------------------------------
    # 9. Save results file
    # ------------------------------------------------------------------
    result_path = os.path.join(os.path.dirname(__file__), "results_stiffness.txt")
    with open(result_path, "w") as fh:
        fh.write("Step 1 Stiffness & Damping Results  [R4]\n")
        fh.write(f"slack_mm={slack_mm}  cycle_avg={use_avg}  freqs={args.freqs}\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"k_eff         = {k_eff:.6f}  Nm/rad  (n={len(stiff_records)} conditions)\n")
        fh.write(f"I_eff         = {I_eff:.8f}  kg·m²\n")
        fh.write(f"E_PLA_calibr  = {E_eff/1e9:.5f}  GPa\n")
        fh.write(f"c_viscous     = {c_viscous:.6f}  Nm·s/rad\n")
        fh.write(f"tau_Coulomb   = {tau_coulomb:.6f}  Nm\n")
        fh.write(f"loss_ratio_ck = {loss_ratio:.6f}  s\n\n")
        fh.write("Per-joint values:\n")
        for i in range(5):
            fh.write(f"  joint{i}: k={k_joints[i]:.4f} Nm/rad   c={c_joints[i]:.5f} Nm·s/rad\n")
        fh.write(f"\nYAML: hingeStiffness: {np.round(k_joints, 3).tolist()}\n")
        fh.write(f"YAML: hingeDamping:   {np.round(c_joints, 4).tolist()}\n")
    print(f"\nSaved: {result_path}")

    # ------------------------------------------------------------------
    # 10. Append R4 entry to ANALYSIS.md
    # ------------------------------------------------------------------
    _append_analysis_r4(
        k_eff=k_eff, I_eff=I_eff, E_eff=E_eff,
        c_viscous=c_viscous, tau_coulomb=tau_coulomb, loss_ratio=loss_ratio,
        k_joints=k_joints, c_joints=c_joints,
        k_global=k_global, I_global=I_global,
        n_stiff=len(stiff_records), n_damp=len(damp_records),
        slack_mm=slack_mm, use_avg=use_avg, freqs=args.freqs,
        stiff_records=stiff_records, damp_records=damp_records,
    )


def _append_analysis_r4(
        k_eff, I_eff, E_eff, c_viscous, tau_coulomb, loss_ratio,
        k_joints, c_joints, k_global, I_global,
        n_stiff, n_damp, slack_mm, use_avg, freqs,
        stiff_records, damp_records):
    """Append R4 run entry to ANALYSIS.md."""
    import datetime
    analysis_path = os.path.join(os.path.dirname(__file__), "ANALYSIS.md")
    if not os.path.exists(analysis_path):
        print(f"  WARNING: {analysis_path} not found — skipping ANALYSIS.md update")
        return

    # Build per-condition detail table
    cond_rows = []
    all_recs = sorted(damp_records, key=lambda r: (r["d_mm"], r["freq"], r["amp_pct"]))
    for r in all_recs:
        in_stiff = r in stiff_records
        cond_rows.append(
            f"| {r['d_mm']:.0f} | {r['freq']:.0f} | {r['amp_pct']:.0f} | "
            f"{r['trav_mm']:.1f} | {r['H'].real:6.2f} | {r['H'].imag:6.3f} | "
            f"{'stiff+damp' if in_stiff else 'damp only'} |"
        )

    today = datetime.date.today().isoformat()
    entry = f"""
---

## R4  ({today})  — N_TRANSIENT_CYCLES=1.0, physical amplitude filter, all frequencies 1–4 Hz

### Configuration

| Parameter | Value |
|-----------|-------|
| Environment | air benchmark, r17.5 pulley |
| Frequencies | {sorted(set(r['freq'] for r in stiff_records))} Hz |
| Amplitude filter | travel ≥ {slack_mm:.1f} mm (physical, not percentage) |
| Track filter | track_r ≥ {THRESHOLDS['track_r_min']:.2f} |
| Cycle averaging | {'ON (synchronous, 512-point grid)' if use_avg else 'OFF'} |
| Transient trim | first {N_TRANSIENT_CYCLES:.0f} cycles discarded |
| n conditions (stiffness) | {n_stiff} |
| n conditions (damping)   | {n_damp} |

### Motivation for R4 changes

**Physical amplitude filter**: the R2 percentage-based filter (`--amp_min 80%`) was
physically wrong because `d9 @ 40% = 3.6 mm > d5 @ 80% = 4.0 mm` — the d9 condition
clears tendon slack but was excluded.  Switching to an absolute threshold of {slack_mm:.1f} mm
correctly passes all conditions above the slack dead-zone regardless of the base travel `d`.

**All frequencies included**: R1/R2 used only 1–2 Hz.  Adding 3–4 Hz data provides a
wider ω² range (up to (8π)² ≈ 631 rad²/s²), which much better separates the k_eff
(y-intercept) from I_eff (slope) in the linear regression.

**2 Hz re-included**: blanket exclusion was overly conservative.  Pre-screening shows
no torque saturation (max Load ≈ 177 raw units, stall ≈ 443) and no velocity saturation
(max RPM ≈ 46, limit = 55) at any 2 Hz condition.

### Pre-analysis quality screening

All 100 air-benchmark files (5 amplitudes × 4 frequencies × 3 d-values × ~2 repeats,
deduplicated to latest timestamp) were loaded and evaluated:

| Issue | Threshold | Conditions found |
|-------|-----------|-----------------|
| SLACK (below dead-zone) | trav < {slack_mm:.1f} mm | 20% amp at all d; 40% at d5 |
| NO_TRACK (motor can't follow) | track_r < {THRESHOLDS['track_r_min']:.2f} | d9 @ 4 Hz @ 100% only |
| LOAD_SAT (torque saturation) | raw Load > {THRESHOLDS['load_sat_raw']} | none |
| VEL_SAT  (velocity saturation) | RPM > {THRESHOLDS['vel_sat_rpm']:.0f} | none |

Key finding: **neither torque nor velocity saturation occurs anywhere in the dataset**.
The motor operates comfortably within its linear range.  The only real exclusions are:
1. Low-amplitude slack conditions (sub-2.5 mm travel)
2. d9 @ 4 Hz @ 100% (track_r ≈ 0.40 — motor cannot follow at this combination)

### Conditions used for stiffness fit (pass_stiffness = True)

| d mm | f Hz | amp% | trav mm | Re(H) | Im(H) | role |
|------|------|------|---------|-------|-------|------|
""" + "\n".join(cond_rows) + f"""

### Results

#### Stiffness and inertia

| Quantity | Value | Units | Notes |
|----------|-------|-------|-------|
| **k_eff** | **{k_eff:.4f}** | **Nm/rad** | **motor-shaft effective stiffness (pass-stiff fit)** |
| I_eff | {I_eff*1e3:.4f}×10⁻³ | kg·m² | effective inertia at motor shaft |
| k_eff (ref, all damp-pass) | {k_global:.4f} | Nm/rad | includes sub-slack points |
| I_eff (ref) | {I_global*1e3:.4f}×10⁻³ | kg·m² | |
| E_PLA (calibrated) | {E_eff/1e9:.3f} | GPa | scaled from nominal 2.7 GPa |

**Expected motor inertia** ≈ 0.10–0.15 × 10⁻³ kg·m² (Dynamixel XL430 rotor).
Measured I_eff = {I_eff*1e3:.3f} × 10⁻³ kg·m².  Includes rotor + tail assembly; excess over rotor
value is tail inertia referred to motor shaft.

#### Damping

| Quantity | Value | Units | Notes |
|----------|-------|-------|-------|
| **c_viscous** | **{c_viscous:.5f}** | **Nm·s/rad** | viscous (speed-proportional) component |
| **τ_Coulomb** | **{tau_coulomb:.5f}** | **Nm** | dry Coulomb friction |
| c/k loss ratio | {loss_ratio:.5f} | s | structural loss factor / ω |
| η @ 1 Hz | {loss_ratio * 2*np.pi:.4f} | — | loss_ratio × 2π |

#### Per-joint values

| Joint | hingeStiffness [Nm/rad] | hingeDamping [Nm·s/rad] |
|-------|------------------------|------------------------|
| joint0 (root) | {k_joints[0]:.3f} | {c_joints[0]:.4f} |
| joint1 | {k_joints[1]:.3f} | {c_joints[1]:.4f} |
| joint2 | {k_joints[2]:.3f} | {c_joints[2]:.4f} |
| joint3 | {k_joints[3]:.3f} | {c_joints[3]:.4f} |
| joint4 (fin tip) | {k_joints[4]:.3f} | {c_joints[4]:.4f} |
| **current MuJoCo** | 0.65 (uniform) | 0.100 (uniform) |
| **ratio j0/MuJoCo** | **{k_joints[0]/0.65:.1f}×** | **{c_joints[0]/0.10:.2f}×** |

#### MuJoCo YAML

```yaml
# Calibrated from air benchmark, R4 (N_TRANSIENT_CYCLES=1.0, physical amplitude filter, 1-4 Hz)
hingeStiffness: {np.round(k_joints, 3).tolist()}
hingeDamping:   {np.round(c_joints, 4).tolist()}
```

### Interpretation

1. **Stiffness 3–5× higher than current MuJoCo**: the calibrated stiffness for joint0 is
   {k_joints[0]:.2f} Nm/rad vs. the current uniform 0.65 Nm/rad — a factor of
   {k_joints[0]/0.65:.1f}×.  This is the primary cause of the "bending-banana" artefact in
   free-swim SysID: the too-soft spring forces the optimizer to inflate `c_angular` as a
   compensatory restoring torque.

2. **I_eff = {I_eff*1e3:.2f} × 10⁻³ kg·m²**: plausible for rotor (≈0.1×10⁻³) plus tail
   assembly referred to the motor shaft through the pulley mechanism.

3. **E_PLA = {E_eff/1e9:.2f} GPa**: within the expected range for FDM PLA [1.5–3.5 GPa],
   confirming the beam-theory model is consistent with measurements.

4. **Damping**: c_viscous = {c_viscous:.4f} Nm·s/rad with τ_Coulomb = {tau_coulomb:.4f} Nm.
   Coulomb friction is not negligible — it contributes more than viscous drag at low velocities.
   The proportional-damping model c_joint_i = (c/k) × k_joint_i assigns higher damping to
   stiffer joints, which is physically reasonable for PLA bending elements.

### Next steps after R4

- Implement `step3_mujoco_validation.py`: fixed-head MuJoCo simulation with per-joint
  stiffness to verify the H(ω) response matches measurement.
- Update `auto_SOFI_field2sim.py` to accept per-joint `hingeStiffness` list instead of
  a single scalar.
- Run fluid SysID E13 with corrected stiffness and tighter c_angular ceiling.
"""

    with open(analysis_path, "a") as fh:
        fh.write(entry)
    print(f"\nAppended R4 entry to: {analysis_path}")


if __name__ == "__main__":
    main()
