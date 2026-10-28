"""Module 03 — PD Gains Analysis.

Empirically estimates real Kp from field test torque/error data.
Builds Bode-style frequency response. Overlays MuJoCo sim trajectory.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import load_field_test_runs, parse_run_name_field
from sysid.motor._torque_utils import TORQUE_CONST, CURRENT_SCALE

OUT_JSON  = Path(__file__).parent / "results" / "03_pd_gains.json"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SIM_KP = 800.0   # Nm/rad (MuJoCo position actuator gainprm[0])
SIM_KV = 20.0    # Nm·s/rad (biasprm[2])


def saturation_mask(df):
    """True where the row is unsaturated (safe for Kp estimation)."""
    speed_limit = df["speed_limit_rad_s"].values
    vel         = df["vel_rad_s"].values
    load_raw    = df["torque_Nm"].values / (CURRENT_SCALE * TORQUE_CONST)
    return (np.abs(load_raw) < 800) & (np.abs(vel) < 0.8 * speed_limit)


def estimate_kp(runs):
    """Linear regression of torque_Nm vs (goal_rad - actual_rad) in unsat rows."""
    all_error, all_torque = [], []
    sat_fracs = {}
    for run_name, df in runs.items():
        df = df.dropna(subset=["torque_Nm", "goal_rad", "actual_rad"])
        mask = saturation_mask(df)
        sat_fracs[run_name] = float(mask.mean())
        error  = (df["goal_rad"] - df["actual_rad"])[mask].values
        torque = df["torque_Nm"][mask].values
        all_error.append(error)
        all_torque.append(torque)

    all_error  = np.concatenate(all_error)
    all_torque = np.concatenate(all_torque)
    slope, intercept, r, _, _ = stats.linregress(all_error, all_torque)
    return float(slope), float(intercept), float(r**2), sat_fracs


def bode_analysis(runs):
    """Return (freq_list, magnitude_real, magnitude_kp_model, phase_lag_ms) per frequency."""
    freq_groups = {}
    for run_name, df in runs.items():
        freq_hz, _ = parse_run_name_field(run_name)
        if freq_hz is None:
            continue
        if freq_hz not in freq_groups:
            freq_groups[freq_hz] = []
        freq_groups[freq_hz].append(df)

    bode = {}
    for freq_hz, dfs in sorted(freq_groups.items()):
        mags_actual, phase_lags = [], []
        for df in dfs:
            df = df.dropna(subset=["goal_rad", "actual_rad"])
            goal   = df["goal_rad"].values
            actual = df["actual_rad"].values
            if len(goal) < 10:
                continue
            mag_goal   = float(np.std(goal))
            mag_actual = float(np.std(actual))
            if mag_goal > 1e-4:
                mags_actual.append(mag_actual / mag_goal)

            # Phase lag via cross-correlation
            corr  = correlate(actual - actual.mean(), goal - goal.mean(), mode="full")
            lags  = np.arange(-(len(goal)-1), len(goal))
            dt    = float(df["Time_s"].diff().median())
            best  = int(lags[np.argmax(corr)])
            phase_lags.append(float(best * dt * 1000))  # ms

        bode[freq_hz] = {
            "mag_mean": float(np.mean(mags_actual)) if mags_actual else None,
            "phase_lag_ms_mean": float(np.mean(phase_lags)) if phase_lags else None,
        }
    return bode


def mujoco_sim_overlay(runs, kp=SIM_KP, kv=SIM_KV):
    """Short MuJoCo rollout for 1 Hz run — compare sim vs real motor trajectory.

    Returns (time_real, actual_real, time_sim, actual_sim) for plotting.
    """
    try:
        import mujoco
        from pathlib import Path as P
        xml_candidates = sorted(
            P(__file__).resolve().parents[2].glob(
                "geometry/SOFI_cpg.xml"
            )
        )
        if not xml_candidates:
            print("  [sim overlay] No SOFI XML found — skipping MuJoCo overlay")
            return None

        xml_path = str(xml_candidates[0])
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Find 1 Hz run
        ref_df = None
        for run_name, df in runs.items():
            f, _ = parse_run_name_field(run_name)
            if f == 1.0:
                ref_df = df.dropna(subset=["goal_rad", "actual_rad"])
                break
        if ref_df is None:
            print("  [sim overlay] No 1 Hz field test run found — skipping")
            return None

        dt_sim   = float(model.opt.timestep)
        dt_real  = float(ref_df["Time_s"].diff().median())
        n_steps  = min(len(ref_df), 500)

        # Motor actuator index
        motor_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_0")
        if motor_idx < 0:
            print("  [sim overlay] motor_0 actuator not found — skipping")
            return None

        # Override Kp/Kv
        model.actuator_gainprm[motor_idx, 0]  = kp
        model.actuator_biasprm[motor_idx, 1]  = -kp
        model.actuator_biasprm[motor_idx, 2]  = -kv

        goal_rad_seq = ref_df["goal_rad"].values[:n_steps]
        actual_sim   = []

        for i in range(n_steps):
            data.ctrl[motor_idx] = goal_rad_seq[i]
            steps_per_frame = max(1, int(round(dt_real / dt_sim)))
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            motor_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "motor_0")
            actual_sim.append(float(data.qpos[model.jnt_qposadr[motor_jid]]))

        time_real   = ref_df["Time_s"].values[:n_steps]
        actual_real = ref_df["actual_rad"].values[:n_steps]
        time_sim    = time_real.copy()

        return time_real, actual_real, time_sim, np.array(actual_sim)

    except Exception as e:
        print(f"  [sim overlay] Error: {e} — skipping")
        return None


def main():
    runs = load_field_test_runs()

    empirical_kp, intercept, r2, sat_fracs = estimate_kp(runs)
    bode = bode_analysis(runs)

    print(f"Empirical Kp: {empirical_kp:.1f} Nm/rad  (R²={r2:.3f})")
    print(f"Sim Kp: {SIM_KP:.1f} Nm/rad")
    kp_mismatch = float((empirical_kp - SIM_KP) / SIM_KP)
    print(f"Kp mismatch: {kp_mismatch:+.1%}")

    # --- Plot 1: torque vs error scatter ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for run_name, df in list(runs.items())[:6]:  # subset for readability
        df = df.dropna(subset=["torque_Nm", "goal_rad", "actual_rad"])
        mask  = saturation_mask(df)
        error = (df["goal_rad"] - df["actual_rad"])[mask]
        ax1.scatter(error, df["torque_Nm"][mask], s=2, alpha=0.3)
    xfit = np.linspace(-0.5, 0.5, 100)
    ax1.plot(xfit, empirical_kp * xfit + intercept, "r-", lw=2,
             label=f"Empirical Kp={empirical_kp:.0f} Nm/rad")
    ax1.plot(xfit, SIM_KP * xfit, "b--", lw=2, label=f"Sim Kp={SIM_KP:.0f} Nm/rad")
    ax1.set_xlabel("Tracking error (rad)")
    ax1.set_ylabel("Torque (Nm)")
    ax1.set_title("Empirical Kp estimation (unsaturated rows only)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "03a_kp_scatter.pdf")
    fig1.savefig(PLOTS_DIR / "03a_kp_scatter.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Bode magnitude ---
    freqs = sorted(bode.keys())
    mags  = [bode[f]["mag_mean"] for f in freqs]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(freqs, mags, "o-", label="Real (field test)")
    ax2.axhline(1.0, color="gray", ls="--", label="Ideal tracking")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude ratio (actual/goal std)")
    ax2.set_title("Bode magnitude: real motor tracking")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "03b_bode_magnitude.pdf")
    fig2.savefig(PLOTS_DIR / "03b_bode_magnitude.png", dpi=150)
    plt.close(fig2)

    # --- Plot 3: MuJoCo sim overlay ---
    overlay = mujoco_sim_overlay(runs)
    if overlay is not None:
        t_real, a_real, t_sim, a_sim = overlay
        fig3, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t_real, a_real, label="Real", lw=1.5)
        axes[0].plot(t_sim,  a_sim,  label=f"Sim (Kp={SIM_KP}, Kv={SIM_KV})", ls="--", lw=1.5)
        axes[0].set_ylabel("Motor position (rad)")
        axes[0].legend()
        axes[0].set_title("Sim vs real motor trajectory — 1 Hz field test run")
        axes[1].plot(t_real, a_real - a_sim, color="red", lw=1.0)
        axes[1].set_ylabel("Residual (real − sim) rad")
        axes[1].set_xlabel("Time (s)")
        fig3.tight_layout()
        fig3.savefig(PLOTS_DIR / "03c_sim_overlay.pdf")
        fig3.savefig(PLOTS_DIR / "03c_sim_overlay.png", dpi=150)
        plt.close(fig3)

    result = {
        "empirical_kp_Nm_per_rad":  round(empirical_kp, 1),
        "kp_fit_r_squared":         round(r2, 4),
        "kp_fit_intercept":         round(intercept, 4),
        "sim_kp_Nm_per_rad":        SIM_KP,
        "kp_mismatch_fraction":     round(kp_mismatch, 4),
        "sim_kv_Nm_s_per_rad":      SIM_KV,
        "bode_by_freq_Hz":          bode,
        "unsaturated_fraction_per_run": {k: round(v, 3) for k, v in sat_fracs.items()},
        "recommended_kp_dr_range":  [round(SIM_KP * 0.8, 0), round(SIM_KP * 1.2, 0)],
        "recommended_kv_dr_range":  [0.0, 20.0],
        "confidence":               "medium — Kp depends on current_limit assumption from module 01",
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
