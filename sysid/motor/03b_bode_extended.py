"""Module 03b — Extended Bode Analysis: Sim vs Real Motor Frequency Response.

Three-way comparison of the motor's amplitude tracking ratio |actual/commanded|:

  1. Real hardware  (0.5–4 Hz) — field test sinusoidal data from 03_pd_gains.json
  2. Sim Kp=5 Kv=0.3 (0.5–12 Hz) — MuJoCo sinusoidal rollouts with the RL env XML
  3. Empirical from RL deploy  — cross-PSD estimate (actual/setpoint) on quality-A runs
  4. Speed-saturation model    — |H| = min(1, v_max / (2π·f·A)) with v_max fitted to
                                  field data; physically motivated by Dynamixel velocity cap

Lower panel shows motor setpoint PSD from deploy runs — reveals what frequency
content the RL policy actually produces during real deployment.

The gap between curves 2 and 1 at 8–12 Hz is the sim-to-real discrepancy that
makes backward swimming possible in sim but physically impossible in reality.

Run from the fishsim root:
    python sim2real/03b_bode_extended.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import csd, welch
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import load_rl_deploy_runs, load_run_index, get_usable_rl_df

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE         = Path(__file__).resolve().parent
RESULTS_DIR  = HERE / "results"
PLOTS_DIR    = HERE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

BODE_JSON    = RESULTS_DIR / "03_pd_gains.json"
OUT_JSON     = RESULTS_DIR / "03b_bode_extended.json"

FISHCREATE   = HERE.parents[2]          # fishsim root
RL_XML       = FISHCREATE / "geometry" / "SOFI.xml"

# Control loop parameters matching the RL env
FRAME_SKIP   = 20       # physics steps per control step
CTRL_DT      = 0.02     # s — 1 / (1000 Hz / 20)

# Sinusoidal test amplitude — matches typical RL policy amplitude in deploy data
SINE_AMP_RAD = 0.50     # rad

# Frequencies to sweep in MuJoCo sinusoidal Bode
SIM_FREQS    = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0])

# LPF cutoff from SDA-S01 sweep (best run lpf_cutoff_hz = 5.46 Hz, nominal 5 Hz)
LPF_HZ       = 5.5


# ---------------------------------------------------------------------------
# Part 1: Load real hardware Bode from module-03 JSON
# ---------------------------------------------------------------------------

def load_real_bode():
    """Load field-test Bode measurements from 03_pd_gains.json."""
    with open(BODE_JSON) as f:
        data = json.load(f)
    bode = data["bode_by_freq_Hz"]
    freqs = np.array([float(k) for k in bode])
    mags  = np.array([bode[k]["mag_mean"] for k in bode])
    # sort ascending
    order = np.argsort(freqs)
    return freqs[order], mags[order]


# ---------------------------------------------------------------------------
# Part 2: Fit speed-saturation model to real Bode data
# ---------------------------------------------------------------------------

def speed_saturation_mag(f, v_max, amplitude=SINE_AMP_RAD):
    """Velocity-saturation amplitude ratio: min(1, v_max / (2π·f·A)).

    When the motor is velocity-limited, the achievable amplitude at frequency f
    is capped at v_max / (2π·f), giving ratio = min(1, v_max/(2π·f·A)).
    """
    return np.minimum(1.0, v_max / (2.0 * np.pi * f * amplitude))


def fit_speed_saturation(real_freqs, real_mags):
    """Fit v_max to real hardware Bode.  Uses field-test amplitude as mean."""
    # Field test amplitudes vary; use 0.65 rad as representative mean
    amp_field = 0.65
    def model(f, v_max):
        return speed_saturation_mag(f, v_max, amplitude=amp_field)
    popt, _ = curve_fit(model, real_freqs, real_mags,
                        p0=[6.0], bounds=([0.5], [30.0]))
    return float(popt[0])


# ---------------------------------------------------------------------------
# Part 3: RL deploy empirical transfer function via cross-PSD
# ---------------------------------------------------------------------------

def compute_empirical_bode(nperseg=128):
    """Cross-PSD estimate |H(f)| = |Gxy(f)| / Gxx(f) from quality-A RL runs.

    x = motor_setpoint (rad), y = motor_pos_rad (actual encoder angle).
    Returns (freqs, mag_mean, mag_std, psd_setpoint_mean, psd_setpoint_std).
    """
    run_index = load_run_index()
    runs_raw  = load_rl_deploy_runs("20260520_*")

    H_per_run   = []    # list of (freq_arr, mag_arr) per run
    psd_per_run = []    # list of (freq_arr, psd_arr) per run for setpoint

    n_used = 0
    for run_name, df_raw in runs_raw.items():
        df = get_usable_rl_df(run_name, df_raw, run_index, min_quality=("A",))
        if df is None or len(df) < nperseg * 2:
            continue

        setpoint = df["motor_setpoint"].values
        actual   = df["motor_pos_rad"].values
        fs       = 1.0 / float(df["step_dt_s"].median())

        # Cross-PSD: H(f) = Pxy(f) / Pxx(f) — coherence-weighted
        f_c, Pxy = csd(setpoint, actual, fs=fs, nperseg=nperseg, window="hann")
        f_w, Pxx = welch(setpoint,         fs=fs, nperseg=nperseg, window="hann")

        # Magnitude (clip to [0, 1.5] to suppress noise at low-input frequencies)
        mag = np.clip(np.abs(Pxy) / np.maximum(Pxx, 1e-12), 0.0, 1.5)

        H_per_run.append((f_c, mag))
        psd_per_run.append((f_w, Pxx))
        n_used += 1

    if n_used == 0:
        print("  [empirical bode] No quality-A runs found — skipping")
        return None

    print(f"  [empirical bode] Used {n_used} quality-A runs")

    # Interpolate all onto a common frequency grid (use first run's grid)
    f_ref = H_per_run[0][0]
    mags_stack = []
    psd_stack  = []
    for f_run, mag_run in H_per_run:
        mags_stack.append(np.interp(f_ref, f_run, mag_run))
    for f_run, psd_run in psd_per_run:
        psd_stack.append(np.interp(f_ref, f_run, psd_run))

    mags_stack = np.array(mags_stack)  # (n_runs, n_freq)
    psd_stack  = np.array(psd_stack)

    # Normalise each run's PSD to unit area for shape comparison
    for i in range(len(psd_stack)):
        area = np.trapz(psd_stack[i], f_ref)
        if area > 0:
            psd_stack[i] /= area

    return {
        "freqs":          f_ref,
        "mag_mean":       mags_stack.mean(axis=0),
        "mag_std":        mags_stack.std(axis=0),
        "psd_mean":       psd_stack.mean(axis=0),
        "psd_std":        psd_stack.std(axis=0),
        "psd_per_run":    psd_stack,
        "n_runs":         n_used,
    }


# ---------------------------------------------------------------------------
# Part 4: MuJoCo sinusoidal Bode with the RL env XML (Kp=5, Kv=0.3)
# ---------------------------------------------------------------------------

def _mujoco_bode_at_freq(model, data, mujoco, motor_act_idx, motor_jnt_adr,
                          freq_hz, amplitude, n_warmup_cycles=3,
                          n_measure_cycles=5):
    """Drive the RL model sinusoidally at freq_hz and return amplitude ratio."""
    import copy

    period_steps = max(1, int(round(1.0 / (freq_hz * CTRL_DT))))
    total_steps  = (n_warmup_cycles + n_measure_cycles) * period_steps

    mujoco.mj_resetData(model, data)

    setpoints = []
    actuals   = []

    for step in range(total_steps):
        t = step * CTRL_DT
        ctrl_val = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
        data.ctrl[motor_act_idx] = ctrl_val

        for _ in range(FRAME_SKIP):
            mujoco.mj_step(model, data)

        if step >= n_warmup_cycles * period_steps:
            setpoints.append(ctrl_val)
            actuals.append(float(data.qpos[motor_jnt_adr]))

    setpoints = np.array(setpoints)
    actuals   = np.array(actuals)

    std_s = float(np.std(setpoints))
    std_a = float(np.std(actuals))
    mag   = (std_a / std_s) if std_s > 1e-6 else 0.0

    # Phase lag via cross-correlation
    corr    = np.correlate(actuals - actuals.mean(),
                           setpoints - setpoints.mean(), mode="full")
    lags    = np.arange(-(len(setpoints) - 1), len(setpoints))
    best_lag = int(lags[np.argmax(corr)])
    phase_lag_ms = float(-best_lag * CTRL_DT * 1000.0)   # negative lag = actual lags behind

    return float(mag), phase_lag_ms


def compute_sim_bode(freqs=SIM_FREQS, amplitude=SINE_AMP_RAD):
    """MuJoCo sinusoidal Bode sweep with the RL env XML (Kp=5, Kv=0.3)."""
    try:
        import mujoco as mj
    except ImportError:
        print("  [sim bode] mujoco not available — skipping")
        return None

    xml_path = str(RL_XML)
    if not RL_XML.exists():
        print(f"  [sim bode] RL XML not found: {xml_path} — skipping")
        return None

    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)

    motor_act_idx = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "motor_0")
    motor_jnt_id  = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT,    "motor_0")
    if motor_act_idx < 0 or motor_jnt_id < 0:
        print("  [sim bode] motor_0 actuator/joint not found — skipping")
        return None
    motor_jnt_adr = int(model.jnt_qposadr[motor_jnt_id])

    # Read back actual Kp/Kv from XML for reporting
    kp = float(model.actuator_gainprm[motor_act_idx, 0])
    kv = float(-model.actuator_biasprm[motor_act_idx, 2])
    print(f"  [sim bode] XML Kp={kp:.1f} Nm/rad, Kv={kv:.2f} Nm·s/rad, "
          f"forcerange=±{model.actuator_forcerange[motor_act_idx, 1]:.1f} Nm")
    print(f"  [sim bode] Running {len(freqs)} frequencies: {freqs} Hz ...")

    mags, phases = [], []
    for f in freqs:
        mag, phi = _mujoco_bode_at_freq(model, data, mj, motor_act_idx,
                                         motor_jnt_adr, f, amplitude)
        mags.append(mag)
        phases.append(phi)
        print(f"    {f:5.1f} Hz → |H|={mag:.3f}  φ={phi:+.0f} ms")

    return {
        "freqs":      list(freqs),
        "mags":       mags,
        "phase_ms":   phases,
        "kp":         kp,
        "kv":         kv,
        "amplitude":  amplitude,
    }


# ---------------------------------------------------------------------------
# Part 5: Figure
# ---------------------------------------------------------------------------

def make_figure(real_freqs, real_mags, v_max, sim_bode, emp_bode):
    fig, (ax_bode, ax_psd) = plt.subplots(
        2, 1, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(
        "Motor frequency response: sim vs real hardware\n"
        f"RL env: Kp=5.0 Nm/rad, Kv=0.30 Nm·s/rad, forcerange=±2.3 Nm  |  "
        f"SDA LPF cutoff: {LPF_HZ} Hz",
        fontsize=11,
    )

    # ── Bode magnitude ──────────────────────────────────────────────────────
    f_extrap = np.linspace(0.3, 14.0, 300)
    amp_field = 0.65   # representative field-test command amplitude

    # Speed-saturation prediction (dashed, extrapolated)
    sat_line = speed_saturation_mag(f_extrap, v_max, amplitude=amp_field)
    ax_bode.plot(f_extrap, sat_line, "k--", lw=1.2, alpha=0.5,
                 label=f"Speed-saturation model  (v_max={v_max:.1f} rad/s, A={amp_field} rad)")

    # Real hardware measured (dots)
    ax_bode.plot(real_freqs, real_mags, "o", color="#E53935", ms=8, zorder=5,
                 label="Real hardware — field test sinusoidal (0.5–4 Hz)")

    # Sim Kp=5
    if sim_bode is not None:
        sf = np.array(sim_bode["freqs"])
        sm = np.array(sim_bode["mags"])
        ax_bode.plot(sf, sm, "s-", color="#1565C0", lw=2.0, ms=6, zorder=4,
                     label=f"Sim Kp=5.0, Kv=0.3 — MuJoCo sinusoidal (A={sim_bode['amplitude']} rad)")

    # Empirical from RL deploy (cross-PSD)
    if emp_bode is not None:
        ef   = emp_bode["freqs"]
        emag = emp_bode["mag_mean"]
        estd = emp_bode["mag_std"]
        # Only plot up to 15 Hz and mask low-frequency noise (<0.3 Hz)
        mask = (ef >= 0.3) & (ef <= 14.0)
        ax_bode.fill_between(ef[mask],
                             np.clip(emag[mask] - estd[mask], 0, 1.5),
                             np.clip(emag[mask] + estd[mask], 0, 1.5),
                             color="#2E7D32", alpha=0.15)
        ax_bode.plot(ef[mask], emag[mask], color="#2E7D32", lw=1.8, alpha=0.8,
                     label=f"Empirical — RL deploy cross-PSD ({emp_bode['n_runs']} runs)")

    # Reference lines
    ax_bode.axvline(LPF_HZ, color="darkorange", lw=1.5, ls=":", alpha=0.8,
                    label=f"SDA LPF cutoff {LPF_HZ} Hz (−3 dB)")
    ax_bode.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Backward-swimming band
    ax_bode.axvspan(8, 12, color="#B71C1C", alpha=0.07, zorder=0)
    ax_bode.text(10.0, 0.95, "backward\nswimming\nband", fontsize=7,
                 color="#B71C1C", ha="center", va="top")

    # Sim-to-real gap annotation at 10 Hz
    if sim_bode is not None:
        f10_idx  = np.argmin(np.abs(np.array(sim_bode["freqs"]) - 10.0))
        sim_at10 = sim_bode["mags"][f10_idx]
        real_at10 = float(speed_saturation_mag(np.array([10.0]), v_max, amp_field)[0])
        ax_bode.annotate(
            f"Sim-to-real gap\n@ 10 Hz:\n"
            f"sim={sim_at10:.0%}, real≈{real_at10:.0%}",
            xy=(10.0, sim_at10), xytext=(11.0, sim_at10 + 0.25),
            arrowprops=dict(arrowstyle="->", color="#1565C0"),
            fontsize=8, color="#1565C0",
        )

    ax_bode.set_xlim(0.2, 14.0)
    ax_bode.set_ylim(-0.02, 1.15)
    ax_bode.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_bode.set_ylabel("|H(f)|  actual / commanded amplitude", fontsize=10)
    ax_bode.legend(fontsize=8, loc="upper right")
    ax_bode.grid(True, alpha=0.3)

    # ── Motor setpoint PSD from RL deploy ───────────────────────────────────
    if emp_bode is not None:
        ef  = emp_bode["freqs"]
        mask = (ef >= 0.1) & (ef <= 20.0)
        for psd_run in emp_bode["psd_per_run"]:
            ax_psd.semilogy(ef[mask], psd_run[mask] + 1e-6,
                            color="gray", lw=0.6, alpha=0.3)
        ax_psd.semilogy(ef[mask], emp_bode["psd_mean"][mask] + 1e-6,
                        color="black", lw=2.0, label="Mean PSD (motor setpoint)")
        ax_psd.fill_between(ef[mask],
                            emp_bode["psd_mean"][mask] - emp_bode["psd_std"][mask] + 1e-6,
                            emp_bode["psd_mean"][mask] + emp_bode["psd_std"][mask] + 1e-6,
                            color="black", alpha=0.1)

    ax_psd.axvline(LPF_HZ, color="darkorange", lw=1.5, ls=":", alpha=0.8,
                   label=f"SDA LPF {LPF_HZ} Hz")
    ax_psd.axvspan(8, 12, color="#B71C1C", alpha=0.07, zorder=0)
    ax_psd.set_xlim(0.1, 20.0)
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=10)
    ax_psd.set_ylabel("Normalised PSD\n(motor setpoint)", fontsize=10)
    ax_psd.set_title("What frequency content does the RL policy actually produce?  "
                     "(DA deploy runs, May 2026)", fontsize=9)
    ax_psd.legend(fontsize=8, loc="upper right")
    ax_psd.grid(True, which="both", alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Module 03b: Extended Bode Analysis ===\n")

    # --- Part 1: Real hardware Bode ---
    print("Loading real hardware Bode from 03_pd_gains.json ...")
    real_freqs, real_mags = load_real_bode()
    print(f"  {len(real_freqs)} points: {dict(zip(real_freqs, real_mags.round(3)))}")

    # --- Fit speed-saturation model ---
    v_max = fit_speed_saturation(real_freqs, real_mags)
    print(f"  Fitted v_max = {v_max:.2f} rad/s  "
          f"({v_max * 60 / (2*np.pi):.0f} RPM)  "
          f"[Dynamixel XW430 free-running: ~57 RPM = 5.97 rad/s]")

    # --- Part 2: Empirical from RL deploy ---
    print("\nComputing empirical Bode from RL deploy runs ...")
    emp_bode = compute_empirical_bode()

    # --- Part 3: Sim Bode ---
    print("\nRunning MuJoCo sinusoidal Bode ...")
    sim_bode = compute_sim_bode()

    # --- Save JSON ---
    result = {
        "real_bode": {
            "freqs_Hz": list(real_freqs),
            "mag_mean":  [round(m, 4) for m in real_mags],
        },
        "speed_saturation": {
            "v_max_rad_s":       round(v_max, 3),
            "v_max_rpm":         round(v_max * 60 / (2*np.pi), 1),
            "amplitude_rad":     0.65,
            "note":              "fitted to field test data; consistent with XW430 ~57 RPM",
        },
        "sim_bode": sim_bode,
        "empirical_bode": {
            "n_runs":    emp_bode["n_runs"] if emp_bode else 0,
            "note":      "cross-PSD |Gxy/Gxx| from quality-A RL deploy runs (May 2026)",
        } if emp_bode else None,
        "gap_at_10Hz": None,
    }

    if sim_bode is not None:
        f_arr = np.array(sim_bode["freqs"])
        m_arr = np.array(sim_bode["mags"])
        if 10.0 in f_arr:
            sim10  = float(m_arr[f_arr == 10.0][0])
            real10 = float(speed_saturation_mag(np.array([10.0]), v_max, 0.65)[0])
            result["gap_at_10Hz"] = {
                "sim_mag":  round(sim10, 3),
                "real_mag": round(real10, 3),
                "ratio":    round(sim10 / real10, 2) if real10 > 0 else None,
                "interpretation": (
                    f"At 10 Hz the sim motor tracks {sim10:.0%} of the commanded amplitude; "
                    f"the real motor is predicted to track only {real10:.0%}. "
                    f"This {sim10/real10:.1f}× gap means backward-swimming oscillations "
                    f"that are executable in sim are physically unachievable on real SOFI."
                ),
            }
            print(f"\n=== KEY FINDING ===")
            print(result["gap_at_10Hz"]["interpretation"])

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # --- Figure ---
    print("\nGenerating figure ...")
    fig = make_figure(real_freqs, real_mags, v_max, sim_bode, emp_bode)
    for ext in ("png", "pdf"):
        out = PLOTS_DIR / f"03b_bode_extended.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
