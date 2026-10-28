"""Module 03c — Perfect Bode Plot for Presentation

Generates a clean, single-panel Bode plot comparing the MuJoCo Digital Twin
to the empirical RL deploy runs, with an option to include/exclude the field tests.
Now includes RL-replay into MuJoCo for a true apples-to-apples comparison!
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import csd, welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sysid.motor._loader import load_rl_deploy_runs, load_run_index, get_usable_rl_df

# ---------------------------------------------------------------------------
# Config & Paths
# ---------------------------------------------------------------------------
PLOT_FIELD_TESTS = True  # Flag to include/exclude the original sinusoidal field tests

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PLOTS_DIR = HERE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

BODE_JSON = RESULTS_DIR / "03_pd_gains.json"
FISHCREATE = HERE.parents[2]          # fishsim root
RL_XML = FISHCREATE / "geometry" / "SOFI_da.xml"

def load_real_bode():
    with open(BODE_JSON) as f:
        data = json.load(f)
    bode = data["bode_by_freq_Hz"]
    freqs = np.array([float(k) for k in bode])
    mags = np.array([bode[k]["mag_mean"] for k in bode])
    order = np.argsort(freqs)
    return freqs[order], mags[order]

def _replay_rl_run_in_mujoco(model, data, mj, setpoint_array, dt):
    mj.mj_resetData(model, data)
    motor_act_idx = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "motor_0")
    motor_jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "motor_0")
    motor_jnt_adr = int(model.jnt_qposadr[motor_jnt_id])
    
    timestep = model.opt.timestep
    frame_skip = max(1, int(round(dt / timestep)))
    
    sim_actuals = []
    for ctrl_val in setpoint_array:
        data.ctrl[motor_act_idx] = ctrl_val
        for _ in range(frame_skip):
            mj.mj_step(model, data)
        sim_actuals.append(float(data.qpos[motor_jnt_adr]))
        
    return np.array(sim_actuals)

def compute_empirical_and_sim_bode(nperseg=256):
    try:
        import mujoco as mj
    except ImportError:
        print("mujoco not available")
        return None, None
        
    model = mj.MjModel.from_xml_path(str(RL_XML))
    data = mj.MjData(model)

    run_index = load_run_index()
    runs_raw = load_rl_deploy_runs("20260520_*")
    
    emp_H_per_run = []
    sim_H_per_run = []
    
    n_used = 0
    for run_name, df_raw in runs_raw.items():
        df = get_usable_rl_df(run_name, df_raw, run_index, min_quality=("A",))
        if df is None or len(df) < nperseg * 2: continue
        
        setpoint = df["motor_setpoint"].values
        actual_real = df["motor_pos_rad"].values
        dt = float(df["step_dt_s"].median())
        fs = 1.0 / dt
        
        # 1. Empirical (Real Hardware) Cross-PSD
        f_c, Pxy_real = csd(setpoint, actual_real, fs=fs, nperseg=nperseg, window="hann")
        f_w, Pxx = welch(setpoint, fs=fs, nperseg=nperseg, window="hann")
        mag_real = np.clip(np.abs(Pxy_real) / np.maximum(Pxx, 1e-12), 0.0, 1.5)
        emp_H_per_run.append((f_c, mag_real))
        
        # 2. Replay in MuJoCo (Simulated) Cross-PSD
        actual_sim = _replay_rl_run_in_mujoco(model, data, mj, setpoint, dt)
        f_c_sim, Pxy_sim = csd(setpoint, actual_sim, fs=fs, nperseg=nperseg, window="hann")
        mag_sim = np.clip(np.abs(Pxy_sim) / np.maximum(Pxx, 1e-12), 0.0, 1.5)
        sim_H_per_run.append((f_c_sim, mag_sim))
        
        n_used += 1

    if n_used == 0: return None, None
    
    f_ref = emp_H_per_run[0][0]
    
    emp_mags_stack = np.array([np.interp(f_ref, f_r, m_r) for f_r, m_r in emp_H_per_run])
    emp_res = {"freqs": f_ref, "mag_mean": emp_mags_stack.mean(axis=0), "mag_std": emp_mags_stack.std(axis=0), "n_runs": n_used}
    
    sim_mags_stack = np.array([np.interp(f_ref, f_r, m_r) for f_r, m_r in sim_H_per_run])
    sim_res = {"freqs": f_ref, "mag_mean": sim_mags_stack.mean(axis=0), "mag_std": sim_mags_stack.std(axis=0), "n_runs": n_used}
    
    return emp_res, sim_res

def make_figure(real_freqs, real_mags, emp_bode, sim_bode, dark_mode=True):
    if dark_mode:
        plt.style.use('dark_background')
        text_color = 'white'
        face_color = 'black'
        fig_size = (10, 4.5)
        title_fontsize = 14
        label_fontsize = 12
        legend_fontsize = 10
    else:
        plt.style.use('default')
        text_color = 'black'
        face_color = 'white'
        fig_size = (10, 6)
        title_fontsize = 14
        label_fontsize = 12
        legend_fontsize = 10
        
    fig, ax = plt.subplots(figsize=fig_size)
    fig.suptitle("Motor Actuator Bandwidth: Digital Twin vs. Empirical Hardware Data", fontsize=title_fontsize, fontweight='bold', color=text_color)

    # 1. Sim RL replay (cross-PSD)
    if sim_bode is not None:
        sf = sim_bode["freqs"]
        smag, sstd = sim_bode["mag_mean"], sim_bode["mag_std"]
        smask = (sf >= 0.19) & (sf <= 14.0)
        ax.fill_between(sf[smask], np.clip(smag[smask]-sstd[smask], 0, 1.5), np.clip(smag[smask]+sstd[smask], 0, 1.5), color="#00BFFF", alpha=0.2)
        ax.plot(sf[smask], smag[smask], color="#00BFFF", lw=2.5, label="Digital Twin (Analytical Physics Simulation)")

    # 2. Empirical from RL deploy (cross-PSD)
    if emp_bode is not None:
        ef = emp_bode["freqs"]
        emag, estd = emp_bode["mag_mean"], emp_bode["mag_std"]
        mask = (ef >= 0.19) & (ef <= 14.0)
        ax.fill_between(ef[mask], np.clip(emag[mask]-estd[mask], 0, 1.5), np.clip(emag[mask]+estd[mask], 0, 1.5), color="#00FF7F", alpha=0.2)
        ax.plot(ef[mask], emag[mask], color="#00FF7F", lw=2.5, label="Empirical Hardware Data (RL Deployment Setpoints)")


    # 3. Real hardware measured (dots)
    if PLOT_FIELD_TESTS and real_freqs is not None:
        ax.plot(real_freqs, real_mags, "o", color="#FF5252", ms=10, label="Empirical Hardware Data (Fixed-Amplitude Sine Waves)")

    ax.axhline(1.0, color="#FF2A2A", lw=2.5, ls="--", alpha=0.9, label="Old MuJoCo Value (Perfect Tracking)")
    
    ax.set_xscale('log')
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 14])
    ax.set_xlim(0.2, 14.0)
    ax.set_ylim(-0.02, 1.15)
    ax.set_xlabel("Command Frequency (Hz) [Log Scale]", fontsize=label_fontsize, color=text_color)
    ax.set_ylabel("Tracking Ratio (Actual / Commanded)", fontsize=label_fontsize, color=text_color)
    
    # Optional: order legend elements
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=legend_fontsize, loc="lower left", facecolor=face_color, edgecolor=text_color)
    ax.grid(True, alpha=0.2, color=text_color)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def main():
    print("Computing Bode plot elements (this may take 10-20 seconds to run the physics engine)...")
    real_freqs, real_mags = load_real_bode()
    emp_bode, sim_bode = compute_empirical_and_sim_bode()
    
    print("Generating perfect figures (dark and light)...")
    fig_dark = make_figure(real_freqs, real_mags, emp_bode, sim_bode, dark_mode=True)
    out_dark = PLOTS_DIR / "03f_bode_final_dark.png"
    fig_dark.savefig(out_dark, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_dark}")
    
    fig_light = make_figure(real_freqs, real_mags, emp_bode, sim_bode, dark_mode=False)
    out_light = PLOTS_DIR / "03f_bode_final_light.png"
    out_pdf = PLOTS_DIR / "03f_bode_final_light.pdf"
    fig_light.savefig(out_light, dpi=200, bbox_inches="tight")
    fig_light.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_light} and {out_pdf}")

if __name__ == "__main__":
    main()
