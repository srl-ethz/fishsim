"""Module 01 — Motor Torque Analysis.

Uses field test Load data to estimate operating torque range and recommend
motor_forcerange for sim domain randomization.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import load_field_test_runs, parse_run_name_field
from sysid.motor._torque_utils import voltage_scaled_speed_limit, is_saturated, STALL_TORQUE_NM

OUT_JSON  = Path(__file__).parent / "results" / "01_motor_torque.json"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def main():
    runs = load_field_test_runs()

    all_torques, all_speeds, all_volts = [], [], []
    peak_torques, mean_volts = [], []
    sat_total, total_rows = 0, 0

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("tab10")
    freq_set = sorted({parse_run_name_field(n)[0] for n in runs})

    for run_name, df in sorted(runs.items()):
        freq_hz, amp_mm = parse_run_name_field(run_name)
        df = df.dropna(subset=["torque_Nm"])

        speed_limits = df["speed_limit_rad_s"].values
        sat_mask = np.array([
            is_saturated(r, v, s)
            for r, v, s in zip(
                (df["torque_Nm"] / (0.00269 * 1.769)).values,
                df["vel_rad_s"].values,
                speed_limits,
            )
        ])
        sat_total  += sat_mask.sum()
        total_rows += len(df)

        color = cmap(freq_set.index(freq_hz) / max(len(freq_set) - 1, 1))
        ax1.plot(df["Time_s"], df["torque_Nm"], color=color, alpha=0.5, lw=0.8,
                 label=f"{freq_hz}Hz" if run_name == list(runs.keys())[0] else "")

        all_torques.append(df["torque_Nm"].values)
        all_speeds.append(df["vel_rad_s"].values)
        all_volts.append(df["Volt"].values)

        peak_torques.append(float(df["torque_Nm"].abs().max()))
        mean_volts.append(float(df["Volt"].mean()))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Torque (Nm)")
    ax1.set_title("Motor torque time series (all field test runs)")
    ax1.legend(title="Frequency", ncol=2)
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "01a_torque_timeseries.pdf")
    fig1.savefig(PLOTS_DIR / "01a_torque_timeseries.png", dpi=150)
    plt.close(fig1)

    all_torques = np.concatenate(all_torques)
    all_speeds  = np.concatenate(all_speeds)
    all_volts   = np.concatenate(all_volts)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(all_speeds, all_torques, s=2, alpha=0.3)
    ax2.axhline(1.0,  color="red",    lw=1.5, label="sim forcerange = 1.0 Nm")
    ax2.axhline(-1.0, color="red",    lw=1.5)
    ax2.axhline(STALL_TORQUE_NM,  color="orange", lw=1.5, ls="--", label=f"stall ceiling {STALL_TORQUE_NM} Nm")
    ax2.axhline(-STALL_TORQUE_NM, color="orange", lw=1.5, ls="--")
    ax2.set_xlabel("Motor speed (rad/s)")
    ax2.set_ylabel("Torque (Nm)")
    ax2.set_title("Speed-torque operating envelope")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "01b_speed_torque_scatter.pdf")
    fig2.savefig(PLOTS_DIR / "01b_speed_torque_scatter.png", dpi=150)
    plt.close(fig2)

    slope, intercept, r, p, _ = stats.linregress(mean_volts, peak_torques)

    fig3, ax3 = plt.subplots()
    ax3.scatter(mean_volts, peak_torques)
    xfit = np.linspace(min(mean_volts), max(mean_volts), 50)
    ax3.plot(xfit, slope * xfit + intercept, "r--",
             label=f"slope={slope:.3f} Nm/V, R²={r**2:.2f}")
    ax3.set_xlabel("Mean Volt per run (V)")
    ax3.set_ylabel("Peak |torque| (Nm)")
    ax3.set_title("Voltage sensitivity of peak torque")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "01c_voltage_sensitivity.pdf")
    fig3.savefig(PLOTS_DIR / "01c_voltage_sensitivity.png", dpi=150)
    plt.close(fig3)

    peak_obs   = float(np.max(np.abs(all_torques)))
    typical    = float(np.percentile(np.abs(all_torques[all_torques != 0]), 75))
    sat_frac   = float(sat_total / max(total_rows, 1))
    volt_range = [float(all_volts.min()), float(all_volts.max())]

    # Sanity check: peak below stall ceiling?
    sane = peak_obs < STALL_TORQUE_NM
    print(f"Peak torque observed: {peak_obs:.3f} Nm  (stall = {STALL_TORQUE_NM} Nm — {'OK' if sane else 'WARNING: EXCEEDS STALL'})")
    print(f"75th-pct operating torque: {typical:.3f} Nm")
    print(f"Saturation fraction: {sat_frac:.3%}")
    print(f"Voltage sensitivity: {slope:.4f} Nm/V")

    # Recommend forcerange as 1.2× peak observed, clamped to stall ceiling
    recommended = min(round(peak_obs * 1.2, 2), STALL_TORQUE_NM)

    result = {
        "assumption": "XW430-T200-R, 2.69mA/unit, torque_const=1.769Nm/A",
        "peak_torque_observed_Nm":       round(peak_obs,   4),
        "75pct_operating_torque_Nm":     round(typical,    4),
        "saturation_fraction":           round(sat_frac,   4),
        "voltage_range_observed_V":      volt_range,
        "voltage_sensitivity_Nm_per_V":  round(slope,      4),
        "stall_ceiling_Nm":              STALL_TORQUE_NM,
        "stall_sanity_check_passed":     sane,
        "recommended_motor_forcerange_Nm": recommended,
        "future_online_use_notes": (
            "PRESENT_LOAD at addr 126 = PRESENT_CURRENT in 2.69mA/unit. "
            "One control-cycle read latency ~10ms. Characterise noise before "
            "adding as policy observation — use module 05 methodology."
        ),
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
