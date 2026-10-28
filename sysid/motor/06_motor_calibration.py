"""Module 06 — Motor Kp/Kv Calibration via 2-D Open-Loop Period Analysis.

2D grid sweep over (Kp, Kv) to find the MuJoCo position-actuator parameters that
simultaneously match hardware amplitude AND phase lag across all standard field test
runs (0.5–4 Hz, 5–12 mm).

WHY the hardware lags more than sim at Kv=0
-------------------------------------------
With Kv=0 the sim motor has no velocity damping — it tracks almost in-phase with
the command.  The real motor has ~47 ms phase lag at 1 Hz (compared to ~10 ms sim).
Adding a small Kv (0.05–0.30 Nm·s/rad) introduces the missing lag without
the large overdamping seen with the original Kv=20.

WHY Kp is NOT 800 Nm/rad
-------------------------
The XML kp=800 over-tracks: sim amplitude ~0.71 rad at 1 Hz vs hardware 0.64 rad.
The real motor is bandwidth-limited by its torque-speed curve (velocity saturation
at ~6 rad/s); at 4 Hz the hardware reaches only 44% of the commanded amplitude.
A lower sim Kp replicates this amplitude roll-off through insufficient control
gain — different physics, similar per-run RMSE.

WHY module-03's empirical Kp=1.3 Nm/rad is wrong
-------------------------------------------------
03_pd_gains regresses torque_Nm on position error. torque_Nm includes fluid drag
and inertia in addition to PD effort (R²=0.36 — almost noise). The slope 1.3 is
the apparent drivetrain spring constant, not the control bandwidth.

RL validation
-------------
Uses only the newest (May 20 2026) runs which have motor_setpoint and motor_pos_rad
columns.  motor_setpoint is the commanded motor angle (rad); motor_pos_rad is the
measured encoder angle.  MuJoCo is stepped at 20 ms / step (matching deploy dt).

Run from the fishsim root:
    python sysid/motor/06_motor_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import parse_run_name_field, load_rl_deploy_runs

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
FIELD_ROOT = REPO_ROOT / "field_test/data/20260401_first_swim"
XML_PATH   = str(Path(__file__).resolve().parent.parent.parent / "geometry/SOFI_cpg.xml")
OUT_JSON   = Path(__file__).parent / "results" / "06_motor_calibration.json"
PLOTS_DIR  = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
(Path(__file__).parent / "results").mkdir(exist_ok=True)

# ─── Sweep parameters ─────────────────────────────────────────────────────────
KP_SWEEP = np.array([0.3, 0.5, 0.8, 1.0, 1.3, 2.0, 3.0, 5.0, 8.0, 10.0])
KV_SWEEP = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])

N_GRID       = 300
SKIP_PERIODS = 2
MAX_PERIODS  = 8

SKIP_RUNS = {
    "run_8_2.0Hz_6.0mm_3.0mm",
    "run_9_2.0Hz_6.0mm_-3.0mm",
    "run_12_2.0Hz_12.0mm_p-1600",
}

# RL-relevant frequency band for picking the RL-best (Kp, Kv)
RL_FREQ_RANGE = (0.8, 2.5)

# ─── Module-level model (loaded once, reused with param overwrites) ───────────
_MODEL:     mujoco.MjModel | None = None
_DATA:      mujoco.MjData  | None = None
_MOTOR_IDX: int = -1
_SIM_CACHE: dict = {}


def _ensure_model():
    global _MODEL, _DATA, _MOTOR_IDX
    if _MODEL is None:
        _MODEL = mujoco.MjModel.from_xml_path(XML_PATH)
        _DATA  = mujoco.MjData(_MODEL)
        _MOTOR_IDX = mujoco.mj_name2id(_MODEL, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_0")
        assert _MOTOR_IDX >= 0, "motor_0 actuator not found in SOFI.xml"
    return _MODEL, _DATA, _MOTOR_IDX


# ─── Hardware data loading ────────────────────────────────────────────────────

def load_run_03(run_dir: Path):
    """Return (t, cmd_rad, meas_rad) from 03_field2sim_ready.csv.

    meas_rad is NaN for non-encoder-update rows and for placeholder zeros.
    """
    csv = run_dir / "processed" / "03_field2sim_ready.csv"
    df  = pd.read_csv(csv)
    t   = df["Time"].values
    cmd = df["Commanded_Alpha"].values
    raw = df["Meas_Alpha"].values.astype(float)
    meas = np.full_like(raw, np.nan)
    valid = df["Meas_Alpha"].notna().values & (raw != 0.0)
    meas[valid] = raw[valid]
    return t, cmd, meas


def period_average(t, cmd, meas, freq):
    """Average sinusoidal command + sparse measured over multiple cycles.

    Returns (t_grid, cmd_avg, meas_avg, n_cmd_periods, n_meas_periods).
    """
    period = 1.0 / freq
    t_grid = np.linspace(0, period, N_GRID, endpoint=False)
    t_start = SKIP_PERIODS * period

    cmd_acc = np.zeros(N_GRID)
    meas_acc = np.zeros(N_GRID)
    cnt = np.zeros(N_GRID)
    n_cmd = 0

    for i in range(MAX_PERIODS):
        t0, t1 = t_start + i * period, t_start + (i + 1) * period
        sel_c = (t >= t0) & (t < t1)
        if sel_c.sum() < 4:
            break
        cmd_acc += np.interp(t_grid, t[sel_c] - t0, cmd[sel_c])
        n_cmd += 1
        sel_m = sel_c & ~np.isnan(meas)
        if sel_m.sum() < 4:
            continue
        meas_acc += np.interp(t_grid, t[sel_m] - t0, meas[sel_m])
        cnt += 1

    cmd_avg  = cmd_acc / max(n_cmd, 1)
    meas_avg = np.where(cnt > 0, meas_acc / cnt, np.nan)
    return t_grid, cmd_avg, meas_avg, n_cmd, int(cnt.max())


# ─── MuJoCo simulation ────────────────────────────────────────────────────────

def sim_openloop(kp, kv, freq, amp_rad, dt=0.001):
    """Period-averaged open-loop response; cached by (kp, kv, freq, amp_rad)."""
    key = (round(kp, 5), round(kv, 5), float(freq), round(float(amp_rad), 5))
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]

    model, data, motor_idx = _ensure_model()
    model.actuator_gainprm[motor_idx, 0] =  kp
    model.actuator_biasprm[motor_idx, 1] = -kp
    model.actuator_biasprm[motor_idx, 2] = -kv
    mujoco.mj_resetData(model, data)

    period  = 1.0 / freq
    n_total = int((SKIP_PERIODS + MAX_PERIODS) * period / dt)
    t_skip  = SKIP_PERIODS * period

    t_list, cmd_list, q_list = [], [], []
    for step in range(n_total):
        t_sim = step * dt
        ctrl  = float(amp_rad) * np.sin(2.0 * np.pi * freq * t_sim)
        data.ctrl[motor_idx] = ctrl
        mujoco.mj_step(model, data)
        if t_sim >= t_skip:
            t_list.append(t_sim - t_skip)
            cmd_list.append(ctrl)
            q_list.append(float(data.joint("motor_0").qpos[0]))

    t_arr = np.array(t_list)
    cmd_arr, q_arr = np.array(cmd_list), np.array(q_list)
    t_grid = np.linspace(0, period, N_GRID, endpoint=False)
    cmd_acc = np.zeros(N_GRID)
    q_acc   = np.zeros(N_GRID)
    n_p     = 0
    for i in range(MAX_PERIODS):
        t0  = i * period
        sel = (t_arr >= t0) & (t_arr < t0 + period)
        if sel.sum() < 4:
            break
        cmd_acc += np.interp(t_grid, t_arr[sel] - t0, cmd_arr[sel])
        q_acc   += np.interp(t_grid, t_arr[sel] - t0, q_arr[sel])
        n_p += 1

    result = (t_grid, cmd_acc / max(n_p, 1), q_acc / max(n_p, 1))
    _SIM_CACHE[key] = result
    return result


def _peak_and_lag(t_grid, sig, ref):
    amp = float(np.nanmax(np.abs(sig)))
    ic  = int(np.argmax(ref))
    im  = int(np.argmax(np.nan_to_num(sig, nan=0.0)))
    dt  = t_grid[1] - t_grid[0]
    return amp, float((im - ic) * dt * 1000)


# ─── RL validation ────────────────────────────────────────────────────────────

def _sim_rl_run(kp, kv, motor_setpoints):
    """Run MuJoCo with the given motor_setpoint sequence (20 ms per step)."""
    model, data, motor_idx = _ensure_model()
    model.actuator_gainprm[motor_idx, 0] =  kp
    model.actuator_biasprm[motor_idx, 1] = -kp
    model.actuator_biasprm[motor_idx, 2] = -kv
    mujoco.mj_resetData(model, data)

    dt_sim    = float(model.opt.timestep)   # 0.001 s
    n_sub     = max(1, int(round(0.020 / dt_sim)))   # 20 sub-steps per command

    out = []
    for ctrl in motor_setpoints:
        data.ctrl[motor_idx] = float(ctrl)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        out.append(float(data.joint("motor_0").qpos[0]))
    return np.array(out)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load hardware data ─────────────────────────────────────────────────
    runs_hw = {}
    for run_dir in sorted(FIELD_ROOT.glob("run_*")):
        rn = run_dir.name
        if rn in SKIP_RUNS:
            continue
        freq, amp_mm = parse_run_name_field(rn)
        if freq is None:
            continue
        csv_path = run_dir / "processed" / "03_field2sim_ready.csv"
        if not csv_path.exists():
            continue
        t, cmd, meas = load_run_03(run_dir)
        t_grid, cmd_avg, meas_avg, n_cmd, n_meas = period_average(t, cmd, meas, freq)
        if np.isnan(meas_avg).all() or n_meas == 0:
            print(f"  skip {rn}: no valid encoder data")
            continue
        amp_rad = float(np.abs(cmd_avg).max())
        hw_amp, hw_lag = _peak_and_lag(t_grid, meas_avg, cmd_avg)
        runs_hw[rn] = dict(
            freq=freq, amp_mm=amp_mm, amp_rad=amp_rad,
            t_grid=t_grid, cmd_avg=cmd_avg, meas_avg=meas_avg,
            n_cmd_periods=n_cmd, n_meas_periods=n_meas,
            hw_amp=hw_amp, hw_lag_ms=hw_lag,
        )
        print(f"  {rn}: {freq}Hz amp_cmd={amp_rad:.3f} hw_amp={hw_amp:.3f} hw_lag={hw_lag:.0f}ms")

    print(f"\nLoaded {len(runs_hw)} runs. Starting 2-D (Kp, Kv) sweep …")

    # ── 2. 2-D RMSE sweep ─────────────────────────────────────────────────────
    # rmse_grid[i,j,rn] = RMSE at KP_SWEEP[i], KV_SWEEP[j] for run rn
    n_kp, n_kv = len(KP_SWEEP), len(KV_SWEEP)
    run_names   = list(runs_hw.keys())
    n_runs      = len(run_names)
    rmse_arr    = np.zeros((n_kp, n_kv, n_runs))

    for ri, rn in enumerate(run_names):
        d = runs_hw[rn]
        valid = ~np.isnan(d["meas_avg"])
        for ki, kp in enumerate(KP_SWEEP):
            for vi, kv in enumerate(KV_SWEEP):
                _, _, qpos = sim_openloop(kp, kv, d["freq"], d["amp_rad"])
                rmse = float(np.sqrt(np.mean((d["meas_avg"][valid] - qpos[valid]) ** 2)))
                rmse_arr[ki, vi, ri] = rmse
        print(f"  {rn}: done ({ri+1}/{n_runs})")

    total_rmse_grid = rmse_arr.sum(axis=2)   # shape (n_kp, n_kv)

    best_flat  = np.unravel_index(total_rmse_grid.argmin(), total_rmse_grid.shape)
    best_kp    = float(KP_SWEEP[best_flat[0]])
    best_kv    = float(KV_SWEEP[best_flat[1]])
    print(f"\nBest global (Kp, Kv) = ({best_kp:.2f}, {best_kv:.2f})"
          f"  total_rmse={total_rmse_grid[best_flat]:.4f}")

    # Per-frequency best
    freq_groups: dict[float, list[int]] = {}
    for ri, rn in enumerate(run_names):
        freq_groups.setdefault(runs_hw[rn]["freq"], []).append(ri)

    freq_best: dict[float, tuple] = {}
    for freq, ridxs in sorted(freq_groups.items()):
        rmse_f  = rmse_arr[:, :, ridxs].sum(axis=2)
        fi, fj  = np.unravel_index(rmse_f.argmin(), rmse_f.shape)
        freq_best[freq] = (float(KP_SWEEP[fi]), float(KV_SWEEP[fj]))
        print(f"  {freq:.1f} Hz → best (Kp={KP_SWEEP[fi]:.2f}, Kv={KV_SWEEP[fj]:.2f})"
              f"  rmse={rmse_f[fi,fj]:.4f}")

    # RL-relevant best (1–2 Hz runs)
    rl_ridxs = [ri for ri, rn in enumerate(run_names)
                if RL_FREQ_RANGE[0] <= runs_hw[rn]["freq"] <= RL_FREQ_RANGE[1]]
    if rl_ridxs:
        rmse_rl = rmse_arr[:, :, rl_ridxs].sum(axis=2)
        ri_best, vi_best = np.unravel_index(rmse_rl.argmin(), rmse_rl.shape)
        best_kp_rl = float(KP_SWEEP[ri_best])
        best_kv_rl = float(KV_SWEEP[vi_best])
        print(f"RL-best (1–2 Hz): Kp={best_kp_rl:.2f}, Kv={best_kv_rl:.2f}")
    else:
        best_kp_rl, best_kv_rl = best_kp, best_kv

    # ── 3. Plots ──────────────────────────────────────────────────────────────
    freqs_sorted = sorted(freq_groups)
    BLUE = "#1f77b4"
    BLACK = "#000000"

    # Comparison combos: (label, kp, kv, color, ls)
    combos_plot = [
        ("HW measured",         None,    None,    BLACK, "-"),
        (f"Sim best ({best_kp:.1f},{best_kv:.2f})", best_kp, best_kv, "red",    "-"),
        ("Sim Kp=800 Kv=20",    800.0,   20.0,    "green",  "--"),
        ("Commanded",           None,    None,    BLUE,  "-"),
    ]

    # --- 06a: 2-D RMSE heatmap ------------------------------------------------
    fig1, axes1 = plt.subplots(
        2, (len(freqs_sorted) + 2) // 2 + 1,
        figsize=(14, 8),
        squeeze=False,
    )
    # Total heatmap (first axis)
    ax_tot = axes1[0, 0]
    im = ax_tot.pcolormesh(
        KV_SWEEP, np.log10(KP_SWEEP),
        total_rmse_grid,
        cmap="viridis_r", shading="auto",
    )
    plt.colorbar(im, ax=ax_tot, label="Total RMSE (rad)")
    ax_tot.scatter([best_kv], [np.log10(best_kp)], color="red", s=80, zorder=5, label=f"best ({best_kp:.1f},{best_kv:.2f})")
    ax_tot.set_xlabel("Kv (Nm·s/rad)")
    ax_tot.set_ylabel("log₁₀(Kp)")
    ax_tot.set_title("All runs — total RMSE")
    ax_tot.set_yticks(np.log10(KP_SWEEP))
    ax_tot.set_yticklabels([f"{k:.2g}" for k in KP_SWEEP])
    ax_tot.legend(fontsize=7)

    # Per-frequency heatmaps
    all_freq_axes = [a for row in axes1 for a in row][1:]
    for ax, freq in zip(all_freq_axes, freqs_sorted):
        ridxs = freq_groups[freq]
        rmse_f = rmse_arr[:, :, ridxs].sum(axis=2)
        bkp, bkv = freq_best[freq]
        im_f = ax.pcolormesh(
            KV_SWEEP, np.log10(KP_SWEEP), rmse_f,
            cmap="viridis_r", shading="auto",
        )
        plt.colorbar(im_f, ax=ax, label="RMSE (rad)")
        ax.scatter([bkv], [np.log10(bkp)], color="red", s=80, zorder=5,
                   label=f"best ({bkp:.1f},{bkv:.2f})")
        ax.set_xlabel("Kv (Nm·s/rad)")
        ax.set_ylabel("log₁₀(Kp)")
        ax.set_title(f"{freq:.1f} Hz")
        ax.set_yticks(np.log10(KP_SWEEP))
        ax.set_yticklabels([f"{k:.2g}" for k in KP_SWEEP])
        ax.legend(fontsize=7)

    # Hide unused axes
    for ax in all_freq_axes[len(freqs_sorted):]:
        ax.set_visible(False)

    fig1.suptitle("2-D (Kp, Kv) sweep — RMSE heatmaps", fontsize=13)
    fig1.tight_layout()
    fig1.savefig(PLOTS_DIR / "06a_kp_kv_heatmap.pdf")
    fig1.savefig(PLOTS_DIR / "06a_kp_kv_heatmap.png", dpi=150)
    plt.close(fig1)
    print("  → 06a_kp_kv_heatmap.png")

    # --- 06b: period panels per frequency -------------------------------------
    rep = {freq: max(rns_idx, key=lambda ri: runs_hw[run_names[ri]]["n_meas_periods"])
           for freq, rns_idx in freq_groups.items()}

    fig2, axes2 = plt.subplots(len(freqs_sorted), 1,
                                figsize=(10, 4 * len(freqs_sorted)), sharex=False)
    if len(freqs_sorted) == 1:
        axes2 = [axes2]

    for ax, freq in zip(axes2, freqs_sorted):
        ri  = rep[freq]
        rn  = run_names[ri]
        d   = runs_hw[rn]
        tg  = d["t_grid"] * 1000

        ax.plot(tg, d["cmd_avg"],  color=BLUE,  lw=2.0, label="Commanded", zorder=5)
        ax.plot(tg, d["meas_avg"], color=BLACK, lw=2.5, label="HW measured", zorder=5)

        bkp_f, bkv_f = freq_best[freq]
        for (label, kp, kv, color, ls) in [
            (f"Sim best-global ({best_kp:.1f},{best_kv:.2f})", best_kp, best_kv, "red",    "-"),
            (f"Sim best-freq   ({bkp_f:.1f},{bkv_f:.2f})",    bkp_f, bkv_f,     "purple", "--"),
            ("Sim (800, 20)",                                   800.0, 20.0,      "green",  ":"),
        ]:
            _, _, q_sim = sim_openloop(kp, kv, freq, d["amp_rad"])
            s_amp, s_lag = _peak_and_lag(d["t_grid"], q_sim, d["cmd_avg"])
            ax.plot(tg, q_sim, color=color, lw=1.5, ls=ls,
                    label=f"{label}  amp={s_amp:.3f} lag={s_lag:.0f}ms")

        hw_amp_v = d["hw_amp"]
        ax.set_title(
            f"{freq:.1f} Hz | {d['amp_mm']} mm | "
            f"hw amp={hw_amp_v:.3f} rad  lag={d['hw_lag_ms']:.0f} ms"
        )
        ax.set_ylabel("Motor position (rad)")
        ax.set_xlabel("Time within period (ms)")
        ax.set_ylim(-0.90, 0.90)
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.25)

    fig2.suptitle(f"Period-averaged motor tracking  —  best global (Kp={best_kp:.1f}, Kv={best_kv:.2f})",
                  fontsize=13)
    fig2.tight_layout()
    fig2.savefig(PLOTS_DIR / "06b_best_fit_period.pdf")
    fig2.savefig(PLOTS_DIR / "06b_best_fit_period.png", dpi=150)
    plt.close(fig2)
    print("  → 06b_best_fit_period.png")

    # --- 06c: Bode comparison -------------------------------------------------
    hw_mag_d:  dict[float, list] = {}
    hw_lag_d:  dict[float, list] = {}
    sim_data_bode: dict[tuple, dict[float, list]] = {
        (best_kp, best_kv):  {},
        (best_kp_rl, best_kv_rl): {},
        (800.0, 20.0):        {},
    }

    for rn, d in runs_hw.items():
        freq    = d["freq"]
        cmd_amp = float(np.nanmax(np.abs(d["cmd_avg"])))
        hw_mag_d.setdefault(freq, []).append(d["hw_amp"] / cmd_amp if cmd_amp > 0 else np.nan)
        hw_lag_d.setdefault(freq, []).append(d["hw_lag_ms"])
        for key in sim_data_bode:
            kp, kv = key
            _, _, q_sim = sim_openloop(kp, kv, freq, d["amp_rad"])
            s_amp, s_lag = _peak_and_lag(d["t_grid"], q_sim, d["cmd_avg"])
            sim_data_bode[key].setdefault(freq, []).append(
                (s_amp / cmd_amp if cmd_amp > 0 else np.nan, s_lag)
            )

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fx = sorted(hw_mag_d)
    ax3a.plot(fx, [np.nanmean(hw_mag_d[f]) for f in fx], "ko-", ms=7, lw=2, label="Hardware")
    ax3b.plot(fx, [np.nanmean(hw_lag_d[f]) for f in fx], "ko-", ms=7, lw=2, label="Hardware")

    bode_styles = {
        (best_kp, best_kv):        ("red",    "-",  f"Sim ({best_kp:.1f}, {best_kv:.2f}) global-best"),
        (best_kp_rl, best_kv_rl):  ("purple", "--", f"Sim ({best_kp_rl:.1f}, {best_kv_rl:.2f}) RL-best"),
        (800.0, 20.0):              ("green",  ":",  "Sim (800, 20)"),
    }
    for key, (c, ls, lbl) in bode_styles.items():
        mags = [np.nanmean([v[0] for v in sim_data_bode[key].get(f, [(np.nan, 0)])]) for f in fx]
        lags = [np.nanmean([v[1] for v in sim_data_bode[key].get(f, [(0, np.nan)])]) for f in fx]
        ax3a.plot(fx, mags, "o", ms=5, color=c, lw=1.5, linestyle=ls, label=lbl)
        ax3b.plot(fx, lags, "o", ms=5, color=c, lw=1.5, linestyle=ls, label=lbl)

    ax3a.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax3a.set_ylabel("Magnitude ratio (actual / commanded)")
    ax3a.set_title("Bode — amplitude ratio")
    ax3a.legend(fontsize=9)
    ax3a.grid(True, alpha=0.3)
    ax3b.set_xlabel("Frequency (Hz)")
    ax3b.set_ylabel("Phase lag (ms)")
    ax3b.set_title("Bode — phase lag")
    ax3b.legend(fontsize=9)
    ax3b.grid(True, alpha=0.3)

    fig3.suptitle("Bode analysis: hardware vs simulation at calibrated (Kp, Kv)", fontsize=13)
    fig3.tight_layout()
    fig3.savefig(PLOTS_DIR / "06c_bode_compare.pdf")
    fig3.savefig(PLOTS_DIR / "06c_bode_compare.png", dpi=150)
    plt.close(fig3)
    print("  → 06c_bode_compare.png")

    # --- 06d: RL validation (May 20 runs) -------------------------------------
    rl_runs_may20 = load_rl_deploy_runs("20260520_*")
    # Pick up to 3 longest runs that have motor_setpoint column
    valid_rl = {rn: df for rn, df in rl_runs_may20.items()
                if "motor_setpoint" in df.columns and len(df) >= 500}
    rl_pick  = sorted(valid_rl, key=lambda rn: len(valid_rl[rn]), reverse=True)[:3]

    if rl_pick:
        fig4, axes4 = plt.subplots(len(rl_pick), 2,
                                    figsize=(14, 4 * len(rl_pick)), squeeze=False)

        rl_rmse_summary = {}
        for row, rl_rn in enumerate(rl_pick):
            df      = valid_rl[rl_rn]
            t_rel   = df["pi_time"].values - df["pi_time"].iloc[0]
            setpts  = df["motor_setpoint"].values
            mpos_hw = df["motor_pos_rad"].values

            ax_pos  = axes4[row, 0]
            ax_res  = axes4[row, 1]

            ax_pos.plot(t_rel, setpts,  color=BLUE,  lw=1.2, label="Commanded", alpha=0.8)
            ax_pos.plot(t_rel, mpos_hw, color=BLACK, lw=1.5, label="HW measured")

            rmse_row = {}
            for (kp, kv, color, ls, lbl) in [
                (best_kp,    best_kv,    "red",   "-",  f"Sim ({best_kp:.1f},{best_kv:.2f}) global-best"),
                (best_kp_rl, best_kv_rl, "purple","--", f"Sim ({best_kp_rl:.1f},{best_kv_rl:.2f}) RL-best"),
                (800.0,      20.0,       "green", ":",  "Sim (800, 20)"),
            ]:
                mpos_sim = _sim_rl_run(kp, kv, setpts)
                rmse = float(np.sqrt(np.nanmean((mpos_hw - mpos_sim) ** 2)))
                rmse_row[f"Kp{kp:.1f}_Kv{kv:.2f}"] = round(rmse, 4)
                ax_pos.plot(t_rel, mpos_sim, color=color, lw=1.2, ls=ls,
                            label=f"{lbl}  RMSE={rmse:.3f}")
                ax_res.plot(t_rel, mpos_hw - mpos_sim, color=color, lw=0.8, ls=ls,
                            label=f"{lbl}  RMSE={rmse:.3f}")

            rl_rmse_summary[rl_rn] = rmse_row
            ax_pos.set_ylabel("Motor position (rad)")
            ax_pos.legend(fontsize=7, ncol=2)
            ax_pos.grid(True, alpha=0.25)
            ax_pos.set_title(rl_rn)
            ax_res.axhline(0, color="gray", lw=0.5)
            ax_res.set_xlabel("Time (s)")
            ax_res.set_ylabel("HW − Sim (rad)")
            ax_res.legend(fontsize=7, ncol=2)
            ax_res.grid(True, alpha=0.25)

        fig4.suptitle("RL validation — May 20 2026 runs  (newest deployment day)", fontsize=12)
        fig4.tight_layout()
        fig4.savefig(PLOTS_DIR / "06d_rl_validation.pdf")
        fig4.savefig(PLOTS_DIR / "06d_rl_validation.png", dpi=150)
        plt.close(fig4)
        print("  → 06d_rl_validation.png")
    else:
        print("  (no suitable May 20 RL runs found — skipping 06d)")
        rl_rmse_summary = {}

    # ── 4. Write JSON ─────────────────────────────────────────────────────────
    kp_spread = sorted(set(v[0] for v in freq_best.values()))
    kv_spread = sorted(set(v[1] for v in freq_best.values()))
    dr_kp_lo  = round(float(min(kp_spread)) * 0.7, 3)
    dr_kp_hi  = round(float(max(kp_spread)) * 1.3, 2)
    dr_kv_hi  = round(float(max(kv_spread)) * 2.0, 2) if kv_spread else 0.5

    result = {
        "calibrated_kp_global_Nm_per_rad": best_kp,
        "calibrated_kv_global_Nm_s_per_rad": best_kv,
        "calibrated_kp_rl_Nm_per_rad":  best_kp_rl,
        "calibrated_kv_rl_Nm_s_per_rad": best_kv_rl,
        "kp_sweep_values": [float(k) for k in KP_SWEEP],
        "kv_sweep_values": [float(k) for k in KV_SWEEP],
        "best_kp_kv_by_freq_Hz": {
            str(f): {"kp": v[0], "kv": v[1]}
            for f, v in freq_best.items()
        },
        "rl_validation_rmse": rl_rmse_summary,
        "note_on_frequency_dependence": (
            f"Best (Kp, Kv) is frequency-dependent: "
            + "; ".join(f"{f:.0f}Hz→({v[0]:.1f},{v[1]:.2f})"
                        for f, v in sorted(freq_best.items()))
            + ". Kp controls amplitude roll-off (hardware velocity saturation), "
            "Kv introduces the missing phase lag."
        ),
        "note_on_module03_mismatch": (
            "03_pd_gains estimated Kp=1.3 Nm/rad via torque/error regression "
            "(R²=0.36). That measures drivetrain apparent stiffness under mixed "
            "fluid+inertia loading, not the control bandwidth. The 2-D period-sweep "
            f"gives (Kp={best_kp:.1f}, Kv={best_kv:.2f}) as the global best match."
        ),
        "recommended_kp_dr_range_updated": [dr_kp_lo, dr_kp_hi],
        "recommended_kv_dr_range_updated": [0.0, dr_kv_hi],
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"\nWrote {OUT_JSON}")

    print("\n=== Motor Calibration Summary ===")
    print(f"Global best:  Kp={best_kp:.2f} Nm/rad  Kv={best_kv:.3f} Nm·s/rad")
    print(f"RL best:      Kp={best_kp_rl:.2f} Nm/rad  Kv={best_kv_rl:.3f} Nm·s/rad")
    for freq in sorted(freq_best):
        kp_f, kv_f = freq_best[freq]
        print(f"  {freq:.1f} Hz → Kp={kp_f:.2f}  Kv={kv_f:.3f}")
    print(f"DR Kp range: [{dr_kp_lo}, {dr_kp_hi}] Nm/rad")
    print(f"DR Kv range: [0.0, {dr_kv_hi}] Nm·s/rad")


if __name__ == "__main__":
    main()
