"""Module 02 — Motor Range Analysis.

Evidence for whether increasing motor_ctrlrange beyond ±0.8 rad is justified.
Three evidence streams: stiffness calibration, field test tracking, static hold.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import (
    load_field_test_runs, load_stiffness_calibration,
    load_rl_deploy_runs, parse_run_name_field, RAD_PER_TICK,
)

OUT_JSON  = Path(__file__).parent / "results" / "02_motor_range.json"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SIM_CTRLRANGE   = 0.8   # current sim clip
DEPLOY_LIMIT    = 1.0   # motor_runner.py MOTOR_LIMIT_RAD


def analyse_stiffness_calibration():
    """Convert Commanded_Current_mA vs Measured_Delta_Ticks to radians."""
    cals = load_stiffness_calibration()
    results = []
    for cal in cals:
        df = cal["data"]
        df["delta_rad"] = df["Measured_Delta_Ticks"] * RAD_PER_TICK
        max_rad = float(df["delta_rad"].max())
        results.append({
            "pulley_mm": cal["pulley_mm"],
            "max_rad": max_rad,
            "data": df,
        })
    return results


def analyse_field_test_tracking():
    """Peak achieved position and tracking error per field test run."""
    runs = load_field_test_runs()
    summary = {}
    for run_name, df in sorted(runs.items()):
        freq_hz, amp_mm = parse_run_name_field(run_name)
        df = df.dropna(subset=["goal_rad", "actual_rad"])
        peak_goal   = float(df["goal_rad"].abs().max())
        peak_actual = float(df["actual_rad"].abs().max())
        error = df["goal_rad"] - df["actual_rad"]
        summary[run_name] = {
            "freq_Hz": freq_hz,
            "amp_mm":  amp_mm,
            "peak_goal_rad":   round(peak_goal,   4),
            "peak_actual_rad": round(peak_actual, 4),
            "mean_abs_error":  round(float(error.abs().mean()), 4),
        }
    return summary


def analyse_static_hold():
    """Steady-state error from straight_then_left static hold runs (May 8)."""
    runs = load_rl_deploy_runs("20260508_*straight_then_left*")
    results = {}
    for run_name, df in runs.items():
        if "trajectory_cmd" not in df.columns:
            continue
        motor_pos = df["motor_pos_rad"].values
        setpoint  = df["trajectory_cmd"].values

        # Find rows where setpoint is near ±0.8 rad (static hold phase)
        hold_mask = np.abs(np.abs(setpoint) - 0.8) < 0.05
        if hold_mask.sum() < 10:
            continue
        error_hold = (setpoint - motor_pos)[hold_mask]
        results[run_name] = {
            "hold_rows":         int(hold_mask.sum()),
            "mean_hold_error_rad": round(float(error_hold.mean()), 4),
            "std_hold_error_rad":  round(float(error_hold.std()),  4),
        }
    return results


def main():
    stiff   = analyse_stiffness_calibration()
    ft      = analyse_field_test_tracking()
    hold    = analyse_static_hold()

    # --- Plot 1: stiffness calibration ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for cal in stiff:
        df = cal["data"]
        ax1.plot(df["Commanded_Current_mA"], df["delta_rad"],
                 label=f"r={cal['pulley_mm']}mm")
    ax1.axhline(SIM_CTRLRANGE,  color="blue",  lw=1.5, ls="--", label=f"sim ±{SIM_CTRLRANGE} rad")
    ax1.axhline(DEPLOY_LIMIT,   color="green", lw=1.5, ls="--", label=f"deploy ±{DEPLOY_LIMIT} rad")
    ax1.set_xlabel("Commanded Current (mA)")
    ax1.set_ylabel("Displacement (rad)")
    ax1.set_title("Stiffness calibration: physical range from current-displacement sweep")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "02a_stiffness_calibration.pdf")
    fig1.savefig(PLOTS_DIR / "02a_stiffness_calibration.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: field test peak positions ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for run_name, info in sorted(ft.items()):
        ax2.scatter(info["amp_mm"], info["peak_actual_rad"],
                    label=f"{info['freq_Hz']}Hz", s=60)
    ax2.axhline(SIM_CTRLRANGE, color="blue",  lw=1.5, ls="--", label=f"sim ±{SIM_CTRLRANGE} rad")
    ax2.axhline(DEPLOY_LIMIT,  color="green", lw=1.5, ls="--", label=f"deploy ±{DEPLOY_LIMIT} rad")
    ax2.set_xlabel("Commanded amplitude (mm)")
    ax2.set_ylabel("Peak achieved position (rad)")
    ax2.set_title("Field test: peak motor position per run")
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "02b_field_test_peak_positions.pdf")
    fig2.savefig(PLOTS_DIR / "02b_field_test_peak_positions.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: tracking error vs commanded position ---
    runs = load_field_test_runs()
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for run_name, df in sorted(runs.items()):
        df = df.dropna(subset=["goal_rad", "actual_rad"])
        error = (df["goal_rad"] - df["actual_rad"]).abs()
        ax3.scatter(df["goal_rad"].abs(), error, s=1, alpha=0.2)
    ax3.axvline(SIM_CTRLRANGE, color="blue",  lw=1.5, ls="--", label=f"±{SIM_CTRLRANGE} rad (sim)")
    ax3.axvline(DEPLOY_LIMIT,  color="green", lw=1.5, ls="--", label=f"±{DEPLOY_LIMIT} rad (deploy)")
    ax3.set_xlabel("|Commanded position| (rad)")
    ax3.set_ylabel("|Tracking error| (rad)")
    ax3.set_title("Tracking error vs commanded position (all field test runs)")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "02c_tracking_error_vs_position.pdf")
    fig3.savefig(PLOTS_DIR / "02c_tracking_error_vs_position.png", dpi=150)
    plt.close(fig3)

    # Mechanical limit estimate: max across stiffness cals
    mech_lim = max((c["max_rad"] for c in stiff), default=None)
    max_achieved = max((v["peak_actual_rad"] for v in ft.values()), default=None)
    hold_errors  = [v["mean_hold_error_rad"] for v in hold.values()]
    hold_err_mean = float(np.mean(hold_errors)) if hold_errors else None

    print(f"Mechanical limit (stiffness cal): {mech_lim:.3f} rad" if mech_lim else "No stiffness cal data")
    print(f"Max achieved in field: {max_achieved:.3f} rad" if max_achieved else "No field data")
    print(f"Static hold error @ 0.8 rad: {hold_err_mean:.4f} rad" if hold_err_mean else "No hold data")

    result = {
        "sim_ctrlrange_rad":             SIM_CTRLRANGE,
        "deploy_motor_limit_rad":        DEPLOY_LIMIT,
        "mechanical_limit_estimated_rad": round(mech_lim, 4) if mech_lim else None,
        "max_achieved_in_field_rad":      round(max_achieved, 4) if max_achieved else None,
        "static_hold_error_at_0p8rad_rad": round(hold_err_mean, 4) if hold_err_mean else None,
        "field_test_tracking_summary":    ft,
        "static_hold_summary":            hold,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    print("\n*** Review results/02_motor_range.json to set recommended_ctrlrange_rad manually ***")


if __name__ == "__main__":
    main()
