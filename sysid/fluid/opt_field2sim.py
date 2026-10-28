#!/usr/bin/env python3
"""
Field-to-Sim Fluid Coefficient Optimizer
=========================================
Optimizes MuJoCo fluidcoef parameters via Bayesian Optimization (BO) or CMA-ES.

Choose optimizer with --optimizer bo|cma  (default: cma, recommended for 10-15 coefs).

Run from the fishsim root (with fishdeploy cloned next to it):

  # CMA-ES, 10 coefs, 6 training runs
  python sysid/fluid/opt_field2sim.py \\
      ../fishdeploy/data/01_field_openloop_20260401/run_01_1.0Hz_5.0mm \\
      ../fishdeploy/data/01_field_openloop_20260401/run_03_1.0Hz_9.0mm \\
      ../fishdeploy/data/01_field_openloop_20260401/run_04_2.0Hz_9.0mm \\
      ../fishdeploy/data/01_field_openloop_20260401/run_07_3.0Hz_7.0mm \\
      ../fishdeploy/data/01_field_openloop_20260401/run_10_4.0Hz_7.0mm \\
      ../fishdeploy/data/01_field_openloop_20260401/run_11_2.0Hz_12.0mm \\
      --n_coef 10 --optimizer cma --physics_bounds --iter 300

Suggested train/validate split (11 runs total):
  Train (6):    run_1, run_3, run_4, run_7, run_10, run_11   (diverse Hz + amplitude)
  Validate (5): run_2, run_5, run_8, run_9, run_12
"""

import argparse
import copy
import os
import re
import sys
import time
import yaml
import signal
import numpy as np
from collections import defaultdict
import mujoco
import pandas as pd
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm
from datetime import datetime
import cma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geometry.sofi_twin import generate_xml, SYSTEMPARAMETERS
from sysid.fluid.es_ekf import FishESEKF

plt.rcParams.update({
    "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.usetex": True, "font.family": 'serif',
})
mm = 1 / 25.4

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_TMP_CONFIG = os.path.join(_REPO_ROOT, "sysid", "fluid", "_opt_tmp.yml")
SYSTEMPARAMETERS["filename"] = os.path.join(_REPO_ROOT, SYSTEMPARAMETERS["filename"])
_SEG_NAMES = {5: ["all"], 10: ["head", "tail+fin"], 15: ["head", "tail", "fin"]}
_PARAM_LABELS = ["blunt", "slender", "angular", "kutta", "magnus"]

_GLOBAL_POOL = None

def signal_handler(sig, frame):
    print("\n\n  [!] Interrupt received. Shutting down workers and saving best result...")
    if _GLOBAL_POOL:
        _GLOBAL_POOL.terminate()
        _GLOBAL_POOL.join()
    raise KeyboardInterrupt

def _sin_model(t, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * t + phase) + offset

def fit_sinusoid(signal, dt):
    """Fit A*sin(2π*f*t + φ) + offset to signal. Returns dict with amp/freq/phase/offset."""
    N = len(signal)
    t = np.arange(N) * dt
    freqs = rfftfreq(N, d=dt)
    spectrum = np.abs(rfft(signal))
    spectrum[0] = 0  # ignore DC for peak search
    fund_bin = int(np.argmax(spectrum))
    freq_guess = float(freqs[fund_bin]) if fund_bin > 0 else 1.0
    amp_guess = float(np.std(signal) * np.sqrt(2))
    offset_guess = float(np.mean(signal))
    try:
        popt, _ = curve_fit(
            _sin_model, t, signal,
            p0=[amp_guess, freq_guess, 0.0, offset_guess],
            bounds=([0.0, 0.2, -np.pi, -1.5], [np.pi, 10.0, np.pi, 1.5]),
            maxfev=8000,
        )
        amp, freq, phase, offset = popt
        return {'amp': float(abs(amp)), 'freq': float(abs(freq)), 'phase': float(phase), 'offset': float(offset)}
    except Exception:
        return {'amp': amp_guess, 'freq': freq_guess, 'phase': 0.0, 'offset': offset_guess}

def compute_thd(signal, dt, fundamental_freq):
    """Total Harmonic Distortion: sqrt(sum(H2..H9)²) / H1 (0→pure sine, →1 more distorted)."""
    N = len(signal)
    freqs = rfftfreq(N, d=dt)
    spectrum = np.abs(rfft(signal))
    df = freqs[1] if len(freqs) > 1 else 1.0
    fund_bin = max(1, min(int(round(fundamental_freq / df)), len(spectrum) - 1))
    H1 = spectrum[fund_bin]
    if H1 < 1e-9:
        return 0.0
    harmonic_power = sum(
        spectrum[fund_bin * n] ** 2
        for n in range(2, 10)
        if fund_bin * n < len(spectrum)
    )
    return float(np.clip(np.sqrt(harmonic_power) / H1, 0.0, 1.0))

def precompute_sinusoid_params(csv_paths, dt):
    """Fit a sinusoid to Ghost_Alpha for each CSV and compute THD. Run once before optimization."""
    params = {}
    print("  [sinusoid] Fitting sinusoids to training runs:")
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        signal = df['Ghost_Alpha'].values
        fit = fit_sinusoid(signal, dt)
        thd = compute_thd(signal, dt, fit['freq'])
        run_label = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        print(f"    {run_label}: A={fit['amp']:.3f} f={fit['freq']:.2f}Hz "
              f"φ={fit['phase']:.2f} off={fit['offset']:.3f}  THD={thd:.3f}")
        params[csv_path] = {**fit, 'thd': thd}
    return params

def get_joint_indices(model, fish_id):
    qpos_indices, qvel_indices = [], []
    joint_names = [
        f"headX_{fish_id}", f"headY_{fish_id}", f"headZ_{fish_id}",
        f"headRollX_{fish_id}", f"headRollY_{fish_id}", f"headRollZ_{fish_id}",
        f"motor_{fish_id}",
        f"joint0_{fish_id}", f"joint1_{fish_id}", f"joint2_{fish_id}",
        f"joint3_{fish_id}", f"joint4_{fish_id}",
    ]
    for j_name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id == -1: continue
        qpos_adr, qvel_adr = model.jnt_qposadr[j_id], model.jnt_dofadr[j_id]
        j_type = model.jnt_type[j_id]
        if j_type == mujoco.mjtJoint.mjJNT_FREE:   n_q, n_v = 7, 6
        elif j_type == mujoco.mjtJoint.mjJNT_BALL: n_q, n_v = 4, 3
        else:                                       n_q, n_v = 1, 1
        for k in range(n_q): qpos_indices.append(qpos_adr + k)
        for k in range(n_v): qvel_indices.append(qvel_adr + k)
    return qpos_indices, qvel_indices

def simulate_headless(config, xml_path, field_csv, oscillation_weight=0.0, backward_penalty=0.0, start_weight=1.0, sinusoid_override=None):
    """Run one simulation. If sinusoid_override is provided (dict with amp/freq/phase/offset),
    Ghost_Alpha is replaced with the fitted sinusoid while ghost positions remain from the CSV."""
    df = pd.read_csv(field_csv)
    sync_cfg, sensor_cfg, ekf_cfg = config['synchronization'], config['sensors'], config['ekf_tuning']
    Kp_pos, Kd_pos = sync_cfg['translation']['stiffness'], sync_cfg['translation']['damping']
    Kp_yaw, Kd_yaw = sync_cfg['rotation']['stiffness'], sync_cfg['rotation']['damping']
    Kp_motor = sync_cfg['motor']['stiffness']
    twin_actuation = config.get('motor', {}).get('twin_actuation', 'measured')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    f0_qpos, f0_qvel = get_joint_indices(model, 0)
    f1_qpos, f1_qvel = get_joint_indices(model, 1)

    ekf = FishESEKF(
        dt=model.opt.timestep, active_sensors=sensor_cfg['active'],
        process_noise_std=ekf_cfg['process_noise_std'], sensor_noise_stds=sensor_cfg['noise'],
        bias_process_noise=ekf_cfg['bias_process_noise'], initial_covariance=ekf_cfg['initial_covariance'],
    )
    mujoco.mj_forward(model, data)

    active, N = sensor_cfg['active'], len(df)
    ghost_x, ghost_y, ghost_yaw = np.empty(N), np.empty(N), np.empty(N)
    twin_x, twin_y, twin_yaw = np.empty(N), np.empty(N), np.empty(N)

    # Store local velocities for backward penalty
    twin_vx_local = np.empty(N)

    dt_sim = model.opt.timestep
    # Pre-build sinusoid signal if override provided (avoids per-step trig in loop)
    if sinusoid_override is not None:
        so = sinusoid_override
        t_arr = np.arange(N) * dt_sim
        _sin_alpha  = so['amp'] * np.sin(2 * np.pi * so['freq'] * t_arr + so['phase']) + so['offset']
        _sin_dalpha = so['amp'] * 2 * np.pi * so['freq'] * np.cos(2 * np.pi * so['freq'] * t_arr + so['phase'])
    else:
        _sin_alpha = _sin_dalpha = None

    for i in range(N):
        row = df.iloc[i]
        ghost_alpha  = _sin_alpha[i]  if _sin_alpha  is not None else row['Ghost_Alpha']
        ghost_dalpha = _sin_dalpha[i] if _sin_dalpha is not None else row['Ghost_dAlpha']

        data.qpos[f0_qpos[0]], data.qpos[f0_qpos[1]] = row['Ghost_X'], row['Ghost_Y']
        data.qpos[f0_qpos[5]], data.qpos[f0_qpos[6]] = row['Ghost_Yaw'], ghost_alpha
        data.qvel[f0_qvel[0]], data.qvel[f0_qvel[1]] = row['Ghost_vX'], row['Ghost_vY']
        data.qvel[f0_qvel[5]], data.qvel[f0_qvel[6]] = row['Ghost_Omega'], ghost_dalpha

        tw_yaw = data.qpos[f1_qpos[5]]
        tw_vxg, tw_vyg = data.qvel[f1_qvel[0]], data.qvel[f1_qvel[1]]
        c, s = np.cos(tw_yaw), np.sin(tw_yaw)
        
        # Local forward velocity (MuJoCo X is swim axis)
        twin_vx_local[i] = tw_vxg * c + tw_vyg * s

        state_old = np.array([
            tw_yaw, twin_vx_local[i], -tw_vxg*s + tw_vyg*c,
            data.qvel[f1_qvel[5]], data.qpos[f1_qpos[6]], data.qvel[f1_qvel[6]],
        ])

        data.ctrl[0] = ghost_alpha

        if twin_actuation == 'teleport':
            # Directly set motor position — zero tracking error, best for SysID
            data.qpos[f1_qpos[6]] = ghost_alpha
            data.qvel[f1_qvel[6]] = ghost_dalpha
            data.ctrl[1] = ghost_alpha
        else:
            # 'measured': high-gain PD tracking of Ghost_Alpha
            ctrl1 = ghost_alpha + Kp_motor * (ekf.x[4] - data.qpos[f1_qpos[6]])
            if not np.isfinite(ctrl1): return 1e6, 1e6, 1e6, 1e6
            data.ctrl[1] = ctrl1

        ekf_vxg, ekf_vyg = ekf.x[1]*c - ekf.x[2]*s, ekf.x[1]*s + ekf.x[2]*c
        data.qfrc_applied[f1_qvel[0]] = Kd_pos*(ekf_vxg - tw_vxg) + Kp_pos*(ekf.x[9]  - data.qpos[f1_qpos[0]])
        data.qfrc_applied[f1_qvel[1]] = Kd_pos*(ekf_vyg - tw_vyg) + Kp_pos*(ekf.x[10] - data.qpos[f1_qpos[1]])
        data.qfrc_applied[f1_qvel[5]] = (
            Kp_yaw * ((ekf.x[0] - tw_yaw + np.pi) % (2*np.pi) - np.pi)
            + Kd_yaw * (ekf.x[3] - data.qvel[f1_qvel[5]])
        )

        mujoco.mj_step(model, data)
        if not all(np.isfinite(data.qpos[idx]) for idx in f1_qpos): return 1e6, 1e6, 1e6, 1e6

        tw_yaw_n = data.qpos[f1_qpos[5]]
        cn, sn = np.cos(tw_yaw_n), np.sin(tw_yaw_n)
        state_new = np.array([
            tw_yaw_n,
            data.qvel[f1_qvel[0]]*cn + data.qvel[f1_qvel[1]]*sn,
            -data.qvel[f1_qvel[0]]*sn + data.qvel[f1_qvel[1]]*cn,
            data.qvel[f1_qvel[5]], data.qpos[f1_qpos[6]], data.qvel[f1_qvel[6]],
        ])
        delta = state_new - state_old
        delta[0] = (delta[0] + np.pi) % (2*np.pi) - np.pi
        delta[4] = (delta[4] + np.pi) % (2*np.pi) - np.pi
        ekf.predict(delta)

        z = {}
        if row['Vision_Update']:
            if 'global_position' in active: z['global_position'] = [row['Meas_X'], row['Meas_Y']]
            if 'heading' in active: z['heading'] = row['Meas_Yaw']
        if row['IMU_Update']:
            if 'motor_encoder' in active: z['motor_encoder'] = [row['Meas_Alpha'], ghost_dalpha]
            if 'imu' in active: z['imu'] = [0.0, 0.0, row['Meas_GyrZ']]
        if z:
            if 'imu' in active and 'imu' not in z: z['imu'] = [0.0, 0.0, ekf.x[3] + ekf.x[8]]
            if 'motor_encoder' in active and 'motor_encoder' not in z: z['motor_encoder'] = [ekf.x[4], ekf.x[5]]
            if 'global_position' in active and 'global_position' not in z: z['global_position'] = [ekf.x[9], ekf.x[10]]
            if 'heading' in active and 'heading' not in z: z['heading'] = ekf.x[0]
            ekf.update(z, data.sensordata[6:8])

        ghost_x[i], ghost_y[i], ghost_yaw[i] = row['Ghost_X'], row['Ghost_Y'], row['Ghost_Yaw']
        twin_x[i], twin_y[i], twin_yaw[i] = data.qpos[f1_qpos[0]], data.qpos[f1_qpos[1]], tw_yaw_n

    # Time-weighted position MSE: exponential decay from start_weight→1.0 over the run.
    # Prevents "swim-backward-then-sprint" exploit: early position errors count more,
    # so the optimizer can never afford a startup detour regardless of how fast it catches up.
    if start_weight > 1.0:
        t = np.arange(N, dtype=float)
        w = np.exp(np.log(start_weight) * (1.0 - t / max(N - 1, 1)))
        w /= w.mean()  # normalize: total loss scale unchanged vs uniform weighting
    else:
        w = np.ones(N)
    mse_pos = float(np.mean(w * ((ghost_x - twin_x)**2 + (ghost_y - twin_y)**2)))
    mse_yaw = float(np.mean(((ghost_yaw - twin_yaw + np.pi) % (2*np.pi) - np.pi)**2))

    dt = model.opt.timestep
    ghost_yaw_rate = np.diff(np.unwrap(ghost_yaw)) / dt
    twin_yaw_rate  = np.diff(np.unwrap(twin_yaw)) / dt
    ghost_osc_std = np.std(ghost_yaw_rate)
    twin_osc_std  = np.std(twin_yaw_rate)
    osc_penalty = ((twin_osc_std - ghost_osc_std) / ghost_osc_std)**2 if twin_osc_std > ghost_osc_std else 0.0

    # Backward penalty: fraction of timesteps with local forward velocity below -5 mm/s.
    # Fraction-based (0–1) so backward_penalty weight is directly interpretable vs position loss.
    back_penalty = float(np.mean(twin_vx_local < -0.005))

    return mse_pos, mse_yaw, oscillation_weight * osc_penalty, backward_penalty * back_penalty

def to_native(obj):
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_native(v) for v in obj]
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    return obj

def expand_coef(prev_coef, target_n):
    n = len(prev_coef)
    if n == target_n: return list(prev_coef)
    if n == 5 and target_n == 10: return list(prev_coef) * 2
    if n == 5 and target_n == 15: return list(prev_coef) * 3
    if n == 10 and target_n == 15: return list(prev_coef[:5]) + list(prev_coef[5:10]) + list(prev_coef[5:10])
    raise ValueError(f"Cannot expand {n} -> {target_n}")

def get_worker_paths():
    pid = multiprocessing.current_process().pid
    if pid is None or multiprocessing.current_process().name == 'MainProcess':
        return _TMP_CONFIG, SYSTEMPARAMETERS["filename"]
    return (
        _TMP_CONFIG.replace(".yml", f"_{pid}.yml"),
        SYSTEMPARAMETERS["filename"].replace(".xml", f"_{pid}.xml"),
    )

def evaluate_params(coef, base_config, field_csvs, yaw_weight, oscillation_weight, backward_penalty, n_coef, start_weight=1.0, sinusoid_weight=0.0, sinusoid_params=None):
    """Evaluate all CSVs sequentially in one worker. Used for BO init phase."""
    tmp_cfg, tmp_xml = get_worker_paths()
    cfg = copy.deepcopy(base_config)
    cfg['fluid_tuning']['coefficients'] = [float(c) for c in coef]
    with open(tmp_cfg, 'w') as f: yaml.dump(cfg, f)
    try:
        sys_params = copy.deepcopy(SYSTEMPARAMETERS)
        sys_params["filename"] = tmp_xml
        generate_xml(sys_params, config_path=tmp_cfg)
    except Exception as e:
        return coef, 1e6, [f"XML_FAIL({type(e).__name__}: {e})"]

    total_loss, details = 0.0, []
    for csv in field_csvs:
        try:
            mse_pos, mse_yaw, p_osc, p_back = simulate_headless(cfg, tmp_xml, csv, oscillation_weight, backward_penalty, start_weight)
            if mse_pos >= 1e5:
                details.append("UNSTABLE"); total_loss += 1e6
            else:
                if sinusoid_weight > 0.0 and sinusoid_params and csv in sinusoid_params:
                    sp = sinusoid_params[csv]
                    sin_w = sinusoid_weight * (1.0 - sp['thd'])
                    try:
                        s_pos, s_yaw, s_osc, s_back = simulate_headless(cfg, tmp_xml, csv, oscillation_weight, backward_penalty, start_weight, sinusoid_override=sp)
                        if s_pos < 1e5:
                            mse_pos = (1 - sin_w) * mse_pos + sin_w * s_pos
                            mse_yaw = (1 - sin_w) * mse_yaw + sin_w * s_yaw
                            p_osc   = (1 - sin_w) * p_osc   + sin_w * s_osc
                            p_back  = (1 - sin_w) * p_back  + sin_w * s_back
                    except Exception:
                        pass
                loss = mse_pos + yaw_weight * mse_yaw + p_osc + p_back
                total_loss += loss
                detail = f"wpos={mse_pos:.5f}"
                if yaw_weight > 0: detail += f" yaw={mse_yaw:.5f}"
                if p_osc > 0:      detail += f" osc={p_osc:.5f}"
                if p_back > 0:     detail += f" back={p_back:.5f}"
                details.append(detail)
        except Exception as e: total_loss += 1e6; details.append(f"SIM_FAIL({type(e).__name__}: {e})")

    for p in [tmp_cfg, tmp_xml]:
        if os.path.exists(p): os.remove(p)
    return coef, total_loss / len(field_csvs), details

def evaluate_single(job_id, coef, base_config, field_csv, yaw_weight, oscillation_weight, backward_penalty, n_coef, start_weight=1.0, sinusoid_weight=0.0, sinusoid_params=None):
    """Evaluate one coefficient set on one CSV. Used for parallel BO and CMA-ES.

    If sinusoid_weight > 0 and sinusoid_params contains an entry for field_csv, runs the
    simulation twice (real Ghost_Alpha + fitted sinusoid) and blends the losses:
        effective_sin_weight = sinusoid_weight * (1 - THD)
        loss = (1 - w) * real_loss + w * sin_loss
    THD-weighting: high harmonic distortion → sinusoid is less representative → less weight.
    """
    tmp_cfg, tmp_xml = get_worker_paths()
    cfg = copy.deepcopy(base_config)
    cfg['fluid_tuning']['coefficients'] = [float(c) for c in coef]
    with open(tmp_cfg, 'w') as f: yaml.dump(cfg, f)
    try:
        sys_params = copy.deepcopy(SYSTEMPARAMETERS)
        sys_params["filename"] = tmp_xml
        generate_xml(sys_params, config_path=tmp_cfg)
    except Exception as e:
        for p in [tmp_cfg, tmp_xml]:
            if os.path.exists(p): os.remove(p)
        return job_id, coef, 1e6, 1e6, 1e6, 1e6, f"XML_FAIL({type(e).__name__}: {e})"
    try:
        mse_pos, mse_yaw, p_osc, p_back = simulate_headless(cfg, tmp_xml, field_csv, oscillation_weight, backward_penalty, start_weight)
        if mse_pos >= 1e5:
            for p in [tmp_cfg, tmp_xml]:
                if os.path.exists(p): os.remove(p)
            return job_id, coef, 1e6, 1e6, 1e6, 1e6, "UNSTABLE"

        # Dual loss: also simulate with fitted sinusoid, blend by THD-adjusted weight
        if sinusoid_weight > 0.0 and sinusoid_params and field_csv in sinusoid_params:
            sp = sinusoid_params[field_csv]
            sin_w = sinusoid_weight * (1.0 - sp['thd'])  # THD-adjusted: more harmonic distortion → less weight
            try:
                s_pos, s_yaw, s_osc, s_back = simulate_headless(
                    cfg, tmp_xml, field_csv, oscillation_weight, backward_penalty, start_weight,
                    sinusoid_override=sp,
                )
                if s_pos < 1e5:
                    mse_pos = (1 - sin_w) * mse_pos + sin_w * s_pos
                    mse_yaw = (1 - sin_w) * mse_yaw + sin_w * s_yaw
                    p_osc   = (1 - sin_w) * p_osc   + sin_w * s_osc
                    p_back  = (1 - sin_w) * p_back  + sin_w * s_back
                    detail_sin = f" sin(w={sin_w:.2f},pos={s_pos:.5f})"
                else:
                    detail_sin = " sin=UNSTABLE"
            except Exception as e:
                detail_sin = f" sin=FAIL({type(e).__name__})"
        else:
            detail_sin = ""

        detail = f"wpos={mse_pos:.5f}"
        if yaw_weight > 0:       detail += f" yaw={mse_yaw:.5f}"
        if p_osc > 0:            detail += f" osc={p_osc:.5f}"
        if p_back > 0:           detail += f" back={p_back:.5f}"
        detail += detail_sin
    except Exception as e:
        mse_pos, mse_yaw, p_osc, p_back, detail = 1e6, 1e6, 1e6, 1e6, f"SIM_FAIL({type(e).__name__}: {e})"
    for p in [tmp_cfg, tmp_xml]:
        if os.path.exists(p): os.remove(p)
    return job_id, coef, mse_pos, mse_yaw, p_osc, p_back, detail

def main():
    global _GLOBAL_POOL
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs='+')
    parser.add_argument("--n_coef", type=int, default=10, choices=[5, 10, 15])
    parser.add_argument("--yaw_weight", type=float, default=0.0)
    parser.add_argument("--oscillation_weight", type=float, default=0.0,
                        help="Weight for head-yaw oscillation amplitude penalty.")
    parser.add_argument("--backward_penalty", type=float, default=0.0,
                        help="Weight for backward swimming penalty (fraction-based: 0–1 * weight).")
    parser.add_argument("--sinusoid_weight", type=float, default=0.0,
                        help="Dual-loss weight for sinusoidal motor input (0=off, 0.4=recommended). "
                             "Each run is evaluated twice (real + fitted sinusoid); losses blended by "
                             "sinusoid_weight*(1-THD) to give less weight when the real signal is "
                             "more distorted (triangular). Encourages generalization to clean sinusoids.")
    parser.add_argument("--start_weight", type=float, default=1.0,
                        help="Exponential time-weighting for position loss. t=0 counts start_weight× more than t=end. 1.0=uniform.")
    parser.add_argument("--iter", type=int, default=300)
    parser.add_argument("--config", type=str, default="sysid/fluid/field2sim.yml")
    parser.add_argument("--output_dir", type=str, default="outputs/sysid")
    parser.add_argument("--upper_bound", type=float, default=10.0)
    parser.add_argument("--physics_bounds", action="store_true",
                        help="Use per-parameter physics-motivated bounds (recommended).")
    parser.add_argument("--bound_range", type=float, default=None,
                        help="Zoom bounds to ±N around warm_start coef.")


    # --- CMA-ES ---
    parser.add_argument("--cma_sigma", type=float, default=2.0,
                        help="CMA-ES initial step size. ~1/3 of search range is a good start.")
    parser.add_argument("--cma_popsize", type=int, default=None,
                        help="CMA-ES population size. Default: auto (4 + 3*ln(n_coef)).")

    # --- warm start / seeding ---
    parser.add_argument("--warm_start", type=str, nargs='+', default=None)
    parser.add_argument("--seed_coef", type=str, action='append', default=None)
    parser.add_argument("--random_start", action="store_true",
                        help="Start CMA-ES from a random point within the bounds.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes. Default: all CPU cores. "
                             "Set to cpu_count//N when running N instances in parallel.")

    args = parser.parse_args()

    field_csvs = [os.path.join(rd, "processed", "03_field2sim_ready.csv") for rd in args.run_dirs]
    with open(args.config, 'r') as f: base_config = yaml.safe_load(f)

    sync_cfg = base_config.get('synchronization', {})
    ky = sync_cfg.get('rotation', {}).get('stiffness', 0)
    print(f"\n{'='*62}")
    print(f"  Optimizer      : CMA-ES")
    print(f"  Coefficients   : {args.n_coef}  ({' | '.join(_SEG_NAMES[args.n_coef])})")
    print(f"  Training runs  : {len(field_csvs)}")
    print(f"  Motor mode     : {base_config.get('motor', {}).get('twin_actuation', 'measured')}")
    print(f"  Kp_yaw         : {ky}")
    if ky > 0:
        print("  ! WARNING: Kp_yaw > 0. Set rotation.stiffness=0 for SysID.")
    print(f"  Weights        : yaw={args.yaw_weight}, osc={args.oscillation_weight}, back={args.backward_penalty}, start_weight={args.start_weight}, sinusoid={args.sinusoid_weight}")
    print(f"{'='*62}\n")

    # --- Physics-motivated bounds ---
    # Updated upper limits for kutta and slender based on supervisor feedback:
    # fish fins with leading-edge vortices can exceed thin-foil theory (2π≈6.28)
    # MuJoCo fluidcoef defaults: blunt=0.5, slender=0.25, angular=1.5, kutta=1.5, magnus=1.0
    # For a fish:
    #   blunt:   streamlined body → below default (0-0.5)
    #   slender: fish body has real added-mass → allow up to 2.0 for head, 15 for oscillating tail
    #   angular: rotational drag, default 1.5 → head (0-3), tail non-zero lower bound
    #   kutta:   head produces little lift (0-5), tail+fin is main thrust driver (0-20)
    #   magnus:  fish segments don't spin → keep tight (0-0.5 head, 0-3 tail)
    _PHYS_BOUNDS_10 = [
        # head-body segment
        # (empirical evidence from 4 runs shown in brackets)
        (0.00, 0.15),  # c0  blunt   — max ever seen: 0.056; streamlined head has low blunt drag
        (0.00, 2.0),   # c1  slender — max ever seen: 1.38; head added-mass, no evidence above 1.5
        (1.00, 5.0),   # c2  angular — min ever seen: 1.07; raised lower to ensure head damping
        (0.00, 2.0),   # c3  kutta   — raised upper from 1.0; head acts as lifting surface during turns, straight-swim loss never explored this
        (0.00, 0.5),   # c4  magnus  — fine as-is; fish head doesn't spin
        # tail+fin segment
        (0.00, 0.2),   # c5  blunt   — max ever seen: 0.087; thin tail is streamlined
        (0.00, 5.0),   # c6  slender — E5+E6 both converged to ~0 independently; likely real signal not degeneracy; turning fix comes from training data not forced lower bound
        (2.00, 20.0),  # c7  angular — Entry 5 hit ceiling (12.0) → widened upper; true value likely higher
        (0.00, 12.0),  # c8  kutta   — values >12 cause instability; sprint cheat drove it up
        (0.00, 6.0),   # c9  magnus  — Entry 5 hit ceiling (4.0) → widened upper
    ]
    _PHYS_BOUNDS_15 = [
        # head segment — mirrors 10-coeff head bounds (well-validated across E2–E7)
        (0.00, 0.15),  # c0  head blunt   — max ever seen: 0.056; streamlined head
        (0.00, 2.0),   # c1  head slender — 10-coeff consistently ~0.6; upper kept generous
        (0.00, 5.0),   # c2  head angular — E9 hit lower bound (1.048≈1.0); relaxed to 0.0 (heavier head may need less forced damping)
        (0.00, 2.0),   # c3  head kutta   — E7 found 0.15 with turning run; head is mild lifting surface
        (0.00, 1.0),   # c4  head magnus  — E9 hit upper bound (0.4999≈0.5); expanded to 1.0
        # tail body segment — same bounds as 10-coeff tail+fin; no empirical basis to assume split a priori
        (0.00, 0.2),   # c5  tail blunt
        (0.00, 5.0),   # c6  tail slender
        (2.00, 20.0),  # c7  tail angular — mirrors 10-coeff [2,20]; let optimizer find split with fin
        (0.00, 12.0),  # c8  tail kutta   — same instability ceiling as 10-coeff
        (0.00, 6.0),   # c9  tail magnus  — mirrors 10-coeff [0,6]
        # fin segment — same bounds as tail; no empirical data to justify asymmetry
        (0.00, 0.2),   # c10 fin blunt
        (0.00, 5.0),   # c11 fin slender
        (2.00, 20.0),  # c12 fin angular  — lowered from 40 back to 20; E13+ has correct stiffness (4-5× stiffer joints) so c_angular no longer needed as synthetic restoring spring
        (0.00, 12.0),  # c13 fin kutta    — same instability ceiling
        (0.00, 10.0),  # c14 fin magnus   — E9 hit upper bound (5.848≈6.0); expanded to 10.0
    ]
    # 5-coeff: single set applied to ALL geoms (head + tail + fin share identical coefficients).
    # angular drag upper bound is generous (30) because SoFi tail geoms are ~9.6x less effective
    # than TendonFish per unit coefficient, even when all zones share the same value.
    _PHYS_BOUNDS_5 = [
        (0.00, 2.0),   # c0  blunt   — streamlined fish body, small
        (0.00, 15.0),  # c1  slender — added mass; TendonFish was 7.79, SoFi may need more
        (0.00, 30.0),  # c2  angular — shared across all geoms; SoFi ~9.6x less effective than TendonFish per unit
        (0.00, 15.0),  # c3  kutta   — fin generates most thrust; needs generous upper
        (0.00, 3.0),   # c4  magnus  — typically small; TendonFish was 0.27
    ]

    if args.physics_bounds:
        if args.n_coef == 10:
            pbounds = {f'c{i}': _PHYS_BOUNDS_10[i] for i in range(10)}
            print("  [physics_bounds] Using per-parameter bounds (10-coef mode).")
        elif args.n_coef == 15:
            pbounds = {f'c{i}': _PHYS_BOUNDS_15[i] for i in range(15)}
            print("  [physics_bounds] Using per-parameter bounds (15-coef mode).")
        elif args.n_coef == 5:
            pbounds = {f'c{i}': _PHYS_BOUNDS_5[i] for i in range(5)}
            print("  [physics_bounds] Using per-parameter bounds (5-coef mode).")
        else:
            pbounds = {f'c{i}': (0.0, args.upper_bound) for i in range(args.n_coef)}
    else:
        pbounds = {f'c{i}': (0.0, args.upper_bound) for i in range(args.n_coef)}

    # --- Warm start / initial coef ---
    warm_coef, prev_evals, evals_per_file = None, [], []
    if args.warm_start:
        best_loss_ws = np.inf
        for ws_path in args.warm_start:
            with open(ws_path, 'r') as f: prev = yaml.load(f, Loader=yaml.UnsafeLoader)
            file_evals = prev.get('all_evaluations', [])
            evals_per_file.append(file_evals)
            prev_evals.extend(file_evals)
            if prev.get('best_loss', np.inf) < best_loss_ws:
                best_loss_ws = prev['best_loss']
                warm_coef = expand_coef(prev['best_coef'], args.n_coef)
        print(f"  [warm_start] Loaded {len(prev_evals)} evals. Best loss: {best_loss_ws:.6f}")
    else:
        try: warm_coef = expand_coef(base_config['fluid_tuning']['coefficients'], args.n_coef)
        except: pass

    if args.bound_range and warm_coef:
        pbounds = {f'c{i}': (max(0.0, warm_coef[i]-args.bound_range), warm_coef[i]+args.bound_range) for i in range(args.n_coef)}
        print(f"  [bound_range] Zoomed ±{args.bound_range} around warm_start coef.")

    # Shared tracking — updated by both BO and CMA-ES paths
    eval_count = [0]
    best_loss  = [np.inf]
    best_c     = [warm_coef[:] if warm_coef else None]
    loss_history, eval_history = [], []
    t_start = time.time()
    n_csvs = len(field_csvs)

    # --- Live convergence plot (file-only, no popup) ---
    os.makedirs(args.output_dir, exist_ok=True)
    ts_live = datetime.now().strftime("%Y%m%d_%H%M%S")
    _live_plot_path  = os.path.join(args.output_dir, f"live_convergence_{ts_live}.png")
    _checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{ts_live}.yml")
    _live_fig = _live_ax = None
    try:
        matplotlib.use('Agg')
        _live_fig, _live_ax = plt.subplots(figsize=(7, 4))
        print(f"  [plot]       Saving live convergence to {_live_plot_path}")
    except Exception:
        _live_fig = _live_ax = None
    print(f"  [checkpoint] Checkpointing best to    {_checkpoint_path}")

    def update_live_plot():
        if _live_fig is None or not loss_history: return
        _live_ax.clear()
        evals = list(range(1, len(loss_history) + 1))
        best_so_far = np.minimum.accumulate(loss_history)
        _live_ax.plot(evals, loss_history, color='steelblue', alpha=0.35, lw=0.8, label='Eval')
        _live_ax.plot(evals, best_so_far, color='crimson', lw=1.5, label=f'Best = {best_loss[0]:.5f}')
        _live_ax.set_xlabel("Evaluation")
        _live_ax.set_ylabel("Loss")
        _live_ax.set_title(f"CMA-ES | n_coef={args.n_coef} | runs={n_csvs}")
        _live_ax.legend(loc='upper right', fontsize=7)
        _live_ax.grid(True, alpha=0.3)
        _live_fig.tight_layout()
        try:
            _live_fig.savefig(_live_plot_path, dpi=150, bbox_inches='tight')
        except Exception:
            pass
        save_checkpoint()

    def save_checkpoint():
        if best_c[0] is None: return
        ckpt = {
            'timestamp':          ts_live,
            'is_checkpoint':      True,
            'optimizer':          'cma',
            'n_coef':             args.n_coef,
            'run_dirs':           args.run_dirs,
            'yaw_weight':         args.yaw_weight,
            'oscillation_weight': args.oscillation_weight,
            'backward_penalty':   args.backward_penalty,
            'sinusoid_weight':    args.sinusoid_weight,
            'start_weight':       args.start_weight,
            'best_loss':          float(best_loss[0]),
            'best_coef':          [float(c) for c in best_c[0]],
            'all_evaluations':    to_native(prev_evals + eval_history),
        }
        tmp = _checkpoint_path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                yaml.safe_dump(ckpt, f)
            os.replace(tmp, _checkpoint_path)
        except Exception as e:
            tqdm.write(f"  [warn] Checkpoint save failed: {e}")

    # Precompute sinusoid fits and THD for each training CSV (done once, before worker pool)
    sinusoid_params = None
    if args.sinusoid_weight > 0.0:
        _env_dt = base_config.get('environment', {}).get('dt', 0.001)
        sinusoid_params = precompute_sinusoid_params(field_csvs, _env_dt)
        print()

    n_workers = args.workers or multiprocessing.cpu_count()
    print(f"  Workers        : {n_workers} / {multiprocessing.cpu_count()} CPU cores\n")
    _GLOBAL_POOL = multiprocessing.Pool(processes=n_workers)

    try:
        # ==================================================================
        # CMA-ES path
        # ==================================================================
        if args.seed_coef:
            # Use the last provided seed_coef as x0
            last_seed = args.seed_coef[-1]
            x0 = [float(x) for x in last_seed.split(',')]
            if len(x0) != args.n_coef:
                print(f"  [CMA-ES] WARNING: seed_coef length ({len(x0)}) != n_coef ({args.n_coef}).")
                x0 = expand_coef(x0, args.n_coef)
            print(f"  [CMA-ES] Initializing with SEED x0 from command line.")
        elif args.random_start:
            x0 = [
                np.random.uniform(pbounds[f'c{i}'][0], pbounds[f'c{i}'][1])
                for i in range(args.n_coef)
            ]
            print(f"  [CMA-ES] Initializing with RANDOM x0 within bounds.")
        else:
            x0 = warm_coef if warm_coef else [
                (pbounds[f'c{i}'][0] + pbounds[f'c{i}'][1]) / 2 for i in range(args.n_coef)
            ]
        
        x0 = [float(np.clip(x0[i], pbounds[f'c{i}'][0], pbounds[f'c{i}'][1])) for i in range(args.n_coef)]
        lb = [pbounds[f'c{i}'][0] for i in range(args.n_coef)]
        ub = [pbounds[f'c{i}'][1] for i in range(args.n_coef)]

        popsize = args.cma_popsize or (4 + int(3 * np.log(args.n_coef)))
        print(f"  [CMA-ES] sigma0={args.cma_sigma}, popsize={popsize}, maxiter={args.iter}")
        print(f"  [CMA-ES] x0 = {[f'{v:.4f}' for v in x0]}\n")

        opts = cma.CMAOptions()
        opts['bounds'] = [lb, ub]
        opts['maxiter'] = args.iter
        opts['popsize'] = popsize
        opts['seed'] = 42
        opts['verbose'] = -9  # suppress internal cma output

        es = cma.CMAEvolutionStrategy(x0, args.cma_sigma, opts)
        pbar = tqdm(total=args.iter, desc="CMA-ES", unit="gen",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        while not es.stop():
            solutions = es.ask()

            # Evaluate all (solution × csv) pairs in parallel
            all_jobs = [
                (s_idx, sol, base_config, csv, args.yaw_weight, args.oscillation_weight, args.backward_penalty, args.n_coef, args.start_weight, args.sinusoid_weight, sinusoid_params)
                for s_idx, sol in enumerate(solutions)
                for csv in field_csvs
            ]
            raw = _GLOBAL_POOL.starmap(evaluate_single, all_jobs)

            per_sol = defaultdict(list)
            for job_id, coef, mse_pos, mse_yaw, p_osc, p_back, detail in raw:
                per_sol[job_id].append((coef, mse_pos, mse_yaw, p_osc, p_back, detail))

            fitnesses = []
            for s_idx, sol in enumerate(solutions):
                runs = per_sol[s_idx]
                total = sum(1e6 if r[1] >= 1e5 else r[1] + args.yaw_weight * r[2] + r[3] + r[4] for r in runs)
                loss = total / n_csvs
                fitnesses.append(loss)
                eval_count[0] += 1
                pdct = {f'c{j}': float(sol[j]) for j in range(args.n_coef)}
                eval_history.append({'params': pdct, 'loss': float(loss)})
                loss_history.append(loss)
                if loss < best_loss[0]:
                    best_loss[0] = loss
                    best_c[0] = list(sol)
                    details_str = ' | '.join(r[5] for r in runs)
                    tqdm.write(f"  [{eval_count[0]:4d}] *** NEW BEST *** loss={loss:.6f}  ({details_str})  t={time.time()-t_start:.0f}s")

            es.tell(solutions, fitnesses)
            update_live_plot()
            pbar.set_postfix({"best": f"{best_loss[0]:.4f}", "σ": f"{es.sigma:.3f}"})
            pbar.update(1)

        pbar.close()
        if es.result.xbest is not None and es.result.fbest < best_loss[0]:
            best_c[0] = list(es.result.xbest)
            best_loss[0] = float(es.result.fbest)

    except KeyboardInterrupt:
        print("\n\n  [!] Interrupted — saving current best before exit...")

    finally:
        try:
            _GLOBAL_POOL.terminate()
            _GLOBAL_POOL.join()
        except: pass

    # --- Save results ---
    if best_c[0] is None or not eval_history:
        print("  No evaluations completed — nothing to save.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    rpath = os.path.join(args.output_dir, f"opt_result_{ts}.yml")

    save_data = {
        'timestamp': ts,
        'optimizer': 'cma',
        'n_coef': args.n_coef,
        'run_dirs': args.run_dirs,
        'yaw_weight': args.yaw_weight,
        'oscillation_weight': args.oscillation_weight,
        'backward_penalty': args.backward_penalty,
        'sinusoid_weight': args.sinusoid_weight,
        'start_weight': args.start_weight,
        'best_loss': float(best_loss[0]),
        'best_coef': [float(c) for c in best_c[0]],
        'all_evaluations': prev_evals + eval_history,
    }

    # Write best coef back to field2sim.yml — done first so it's never lost
    with open(args.config, 'r') as f: txt = f.read()
    with open(args.config, 'w') as f:
        f.write(re.sub(r'(coefficients\s*:).*', rf'\1 [{", ".join(f"{c:.6g}" for c in best_c[0])}]', txt))

    with open(rpath, 'w') as f:
        yaml.safe_dump(to_native(save_data), f)

    if loss_history:
        try:
            plt.rcParams.update({"text.usetex": False})
            fig, ax = plt.subplots(figsize=(88*mm, 55*mm))
            ax.plot(range(1, len(loss_history)+1), loss_history, alpha=0.5, label='Eval')
            ax.plot(range(1, len(loss_history)+1), np.minimum.accumulate(loss_history), '--', lw=1.5, label='Best')
            ax.set_xlabel("Evaluation"); ax.set_ylabel("Loss"); ax.grid(); ax.legend()
            fig.savefig(os.path.join(args.output_dir, f"convergence_{ts}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  [warn] Convergence plot skipped: {e}")

    print(f"\n  Best loss : {best_loss[0]:.6f}")
    print(f"  Best coef : {[f'{c:.4f}' for c in best_c[0]]}")
    print(f"  Saved to  : {rpath}")

if __name__ == "__main__":
    main()


