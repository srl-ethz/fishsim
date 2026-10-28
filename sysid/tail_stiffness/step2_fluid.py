"""
Step 2 — Fluid Force Identification from Air-Water Benchmark Difference
-----------------------------------------------------------------------
Extracts the added mass m_added [kg·m² at motor shaft] and drag coefficient
c_drag [Nm·s/rad at motor shaft] from the difference in frequency response
between water and air benchmarks.

Method (see ANALYSIS.md §4):
  ΔH(ω) = H_water(ω) - H_air(ω)
         = -m_added * ω²  +  j * c_drag * ω

  real(ΔH) = -m_added * ω²   (in-phase with position; water adds inertia)
  imag(ΔH) =  c_drag * ω     (in-phase with velocity; water adds drag)

Dividing by actual position amplitude (via phasor) cancels the effect of
different motor tracking in air vs. water — the two signals need not have
equal amplitude.

Available overlap (r17.5):
  d05: air {1,2,3,4} Hz, water {0.5,1,1.5,2} Hz → overlap: 1 Hz, 2 Hz
  d07: air {1-4} Hz,     water {0.5,1} Hz        → overlap: 1 Hz
  d09: air {1-4} Hz,     water {0.5,1} Hz        → overlap: 1 Hz

Usage:
    python sysid/tail_stiffness/step2_fluid.py
    python sysid/tail_stiffness/step2_fluid.py --d 5 --freqs 1 2
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
    find_latest_file, load_csv, to_si, trim_transient, compute_H,
    PULLEY_RADIUS_M
)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
AMP_FRACS = [20.0, 40.0, 60.0, 80.0, 100.0]
N_TRANSIENT_CYCLES = 3.0

# d-values for which BOTH air and water data exist at r17.5
OVERLAP_CONDITIONS = {
    5.0: [1.0, 2.0],
    7.0: [1.0],
    9.0: [1.0],
}


def get_H(env, d_mm, freq_hz, amp_pct):
    """Return H phasor or None if data unavailable."""
    path = find_latest_file(env, d_mm, freq_hz, amp_pct)
    if path is None:
        return None
    _, raw = load_csv(path)
    si = to_si(raw)
    si = trim_transient(si, N_TRANSIENT_CYCLES, freq_hz)
    if len(si["t"]) < 20:
        return None
    return compute_H(si, freq_hz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d",     type=float, nargs="+", default=list(OVERLAP_CONDITIONS.keys()))
    parser.add_argument("--freqs", type=float, nargs="+", default=None,
                        help="Frequencies (Hz). Default: all available overlaps.")
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    records = []

    print("\n--- Computing ΔH = H_water - H_air ---")
    print(f"  {'d_mm':>5} {'freq':>5} {'amp':>5}  "
          f"{'Re(H_air)':>10} {'Re(H_w)':>10} {'Re(ΔH)':>10} {'Im(ΔH)':>10}")

    for d_mm in args.d:
        freqs_for_d = args.freqs if args.freqs else OVERLAP_CONDITIONS.get(d_mm, [])
        for freq_hz in freqs_for_d:
            omega = 2 * np.pi * freq_hz
            for amp_pct in AMP_FRACS:
                H_air   = get_H("air",   d_mm, freq_hz, amp_pct)
                H_water = get_H("water", d_mm, freq_hz, amp_pct)
                if H_air is None or H_water is None:
                    continue
                dH = H_water - H_air
                records.append({
                    "d_mm":    d_mm,
                    "freq":    freq_hz,
                    "amp_pct": amp_pct,
                    "omega":   omega,
                    "Re_dH":   dH.real,
                    "Im_dH":   dH.imag,
                    "Re_H_air":   H_air.real,
                    "Re_H_water": H_water.real,
                    # Derived: m_added and c_drag per this single point
                    "m_added_pt": -dH.real / omega**2 if omega > 0 else np.nan,
                    "c_drag_pt":   dH.imag / omega    if omega > 0 else np.nan,
                })
                print(f"  {d_mm:>5.1f} {freq_hz:>5.1f} {amp_pct:>5.0f}  "
                      f"{H_air.real:>10.1f} {H_water.real:>10.1f} "
                      f"{dH.real:>10.2f} {dH.imag:>10.2f}")

    if not records:
        print("No matched air-water pairs found.")
        return

    import pandas as pd
    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 1. ΔH decomposition plot — one panel per d value
    # ------------------------------------------------------------------
    d_vals = sorted(df["d_mm"].unique())
    fig, axes = plt.subplots(2, len(d_vals), figsize=(5 * len(d_vals), 7), sharex="col")
    if len(d_vals) == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(AMP_FRACS)))
    amp_map = {a: c for a, c in zip(AMP_FRACS, colors)}

    for j, d_mm in enumerate(d_vals):
        sub = df[df["d_mm"] == d_mm]
        for amp_pct in AMP_FRACS:
            pts = sub[sub["amp_pct"] == amp_pct].sort_values("omega")
            if pts.empty:
                continue
            col = amp_map[amp_pct]
            axes[0, j].scatter(pts["omega"], pts["Re_dH"], color=col, s=50, zorder=3)
            axes[0, j].plot(pts["omega"],    pts["Re_dH"], color=col, lw=1.2, label=f"{amp_pct:.0f}%")
            axes[1, j].scatter(pts["omega"], pts["Im_dH"], color=col, s=50, zorder=3)
            axes[1, j].plot(pts["omega"],    pts["Im_dH"], color=col, lw=1.2)

        # Guide lines: -m*ω² (real) and c*ω (imag) with median params
        sub_d = sub.dropna(subset=["m_added_pt", "c_drag_pt"])
        m_med = sub_d["m_added_pt"].median()
        c_med = sub_d["c_drag_pt"].median()
        if len(sub_d) > 0:
            om_line = np.linspace(0, sub["omega"].max() * 1.1, 50)
            axes[0, j].plot(om_line, -m_med * om_line**2, "k--", lw=1.2,
                            label=f"−m·ω² (m={m_med*1e3:.2f}×10⁻³)")
            axes[1, j].plot(om_line,  c_med * om_line,    "k--", lw=1.2,
                            label=f"c·ω   (c={c_med:.4f})")

        axes[0, j].set_title(f"d = {d_mm} mm")
        axes[0, j].set_ylabel("Re(ΔH)  [Nm/rad]") if j == 0 else None
        axes[1, j].set_ylabel("Im(ΔH)  [Nm·s/rad]") if j == 0 else None
        axes[1, j].set_xlabel("ω  [rad/s]")
        for ax in axes[:, j]:
            ax.axhline(0, color="0.7", lw=0.8)
            ax.grid(True, alpha=0.3)
        axes[0, j].legend(fontsize=7, ncol=2)
        axes[1, j].legend(fontsize=7)

    plt.suptitle("Fluid force: ΔH = H_water − H_air\n"
                 "real(ΔH) = −m_added·ω²  |  imag(ΔH) = c_drag·ω", fontsize=11)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "s2_delta_H.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nSaved: {out}")

    # ------------------------------------------------------------------
    # 2. m_added and c_drag vs. amplitude (nonlinearity check)
    # ------------------------------------------------------------------
    fig2, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(10, 4))
    markers = ["o", "s", "^", "D"]
    freq_vals = sorted(df["freq"].unique())

    for d_mm, mk in zip(d_vals, markers[:len(d_vals)]):
        for freq_hz in sorted(df[df["d_mm"] == d_mm]["freq"].unique()):
            sub = df[(df["d_mm"] == d_mm) & (df["freq"] == freq_hz)].sort_values("amp_pct")
            label = f"d={d_mm}mm  {freq_hz}Hz"
            ax_m.plot(sub["amp_pct"], sub["m_added_pt"] * 1e3, marker=mk, label=label, lw=1.5)
            ax_c.plot(sub["amp_pct"], sub["c_drag_pt"],         marker=mk, label=label, lw=1.5)

    for ax, title, ylabel in [
        (ax_m, "Added mass vs. amplitude", "m_added  [×10⁻³ kg·m²]"),
        (ax_c, "Drag coefficient vs. amplitude", "c_drag  [Nm·s/rad]"),
    ]:
        ax.set_xlabel("Amplitude  [%]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(PLOT_DIR, "s2_fluid_vs_amplitude.png")
    plt.savefig(out2, dpi=150); plt.close()
    print(f"Saved: {out2}")

    # ------------------------------------------------------------------
    # 3. Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS — fluid force at motor shaft")
    print("=" * 60)
    print(f"\n  {'d_mm':>5} {'freq':>5} {'m_added [g·m²]':>15} {'c_drag [Nm·s/rad]':>18}")
    print(f"  {'-'*5} {'-'*5} {'-'*15} {'-'*18}")
    for _, row in df.sort_values(["d_mm", "freq", "amp_pct"]).iterrows():
        print(f"  {row['d_mm']:>5.1f} {row['freq']:>5.1f} "
              f"{row['m_added_pt']*1e3:>15.3f} {row['c_drag_pt']:>18.4f}")

    m_med_all = df["m_added_pt"].median()
    c_med_all = df["c_drag_pt"].median()
    print(f"\n  Median across all conditions:")
    print(f"    m_added = {m_med_all*1e3:.3f} ×10⁻³ kg·m²")
    print(f"    c_drag  = {c_med_all:.4f} Nm·s/rad")

    # How much of the motor's torque budget is fluid at typical swimming conditions?
    # 2Hz, d05 = 60%: theta_peak ≈ 0.0175 * 2π / 17.5mm * 3.0mm = 0.171 rad
    omega_swim = 2 * np.pi * 2.0
    theta_peak = 0.171  # rad, approximate for 3mm travel
    tau_fluid_peak = (m_med_all * omega_swim**2 + c_med_all * omega_swim) * theta_peak
    print(f"\n  Estimated fluid torque at 2Hz, 3mm travel (θ≈0.171 rad):")
    print(f"    τ_fluid_peak ≈ {tau_fluid_peak*1000:.1f} mNm  "
          f"({tau_fluid_peak/1.5*100:.0f}% of motor stall torque)")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    result_path = os.path.join(os.path.dirname(__file__), "results_fluid.txt")
    with open(result_path, "w") as fh:
        fh.write("Step 2 Fluid Results\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"m_added_median = {m_med_all:.6f}  kg·m² (motor shaft)\n")
        fh.write(f"c_drag_median  = {c_med_all:.6f}  Nm·s/rad (motor shaft)\n\n")
        fh.write("Per-condition breakdown:\n")
        fh.write(df.to_string(index=False))
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
