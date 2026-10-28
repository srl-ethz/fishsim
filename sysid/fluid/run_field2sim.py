"""
Field-to-Sim Synchronization Orchestrator
Mirrors run_sim2sim.py architecture exactly.
Differences from sim2sim (marked with # [F2S]):
  - Ghost Fish (0) is force-teleported from field CSV (not physics-simulated).
  - EKF measurements come from CSV Meas_* columns, gated by Vision_Update /
    IMU_Update flags (actual measurement timestamps, not fixed frequency).
  - Spline columns (Ghost_X/Y/Alpha) drive Ghost teleportation only - never EKF.
"""

import mujoco
import numpy as np
import cv2
import os
import sys
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geometry.auto_SOFI_field2sim import generate_xml, SYSTEMPARAMETERS
from sysid.fluid.es_ekf import FishESEKF

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "text.usetex": True, "font.family": 'serif'})
mm = 1 / 25.4


def pi_formatter(x, pos):
    if np.isclose(x, 0): return "0"
    return f"{x/np.pi:g}" + r"$\pi$"


def get_joint_indices(model, fish_id):
    """Dynamically resolves qpos and qvel indices for a fish by joint name."""
    qpos_indices = []
    qvel_indices = []
    joint_names = [
        f"headX_{fish_id}", f"headY_{fish_id}", f"headZ_{fish_id}",
        f"headRollX_{fish_id}", f"headRollY_{fish_id}", f"headRollZ_{fish_id}",
        f"motor_{fish_id}",
        f"joint0_{fish_id}", f"joint1_{fish_id}", f"joint2_{fish_id}",
        f"joint3_{fish_id}", f"joint4_{fish_id}",
    ]
    for j_name in joint_names:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if j_id == -1:
            print(f"Warning: Joint {j_name} not found in model.")
            continue
        qpos_adr = model.jnt_qposadr[j_id]
        qvel_adr = model.jnt_dofadr[j_id]
        j_type   = model.jnt_type[j_id]
        if j_type == mujoco.mjtJoint.mjJNT_FREE:
            n_qpos, n_qvel = 7, 6
        elif j_type == mujoco.mjtJoint.mjJNT_BALL:
            n_qpos, n_qvel = 4, 3
        else:
            n_qpos, n_qvel = 1, 1
        for k in range(n_qpos): qpos_indices.append(qpos_adr + k)
        for k in range(n_qvel): qvel_indices.append(qvel_adr + k)
    return qpos_indices, qvel_indices


def run_field2sim():
    # ------------------------------------------------------------------
    # 1. LOAD CONFIG
    # ------------------------------------------------------------------
    config_path = "sysid/fluid/field2sim.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_cfg    = config['environment']
    paths_cfg  = config['paths']
    sync_cfg   = config['synchronization']
    motor_cfg  = config['motor']
    sensor_cfg = config['sensors']
    ekf_cfg    = config['ekf_tuning']

    Kp_pos   = sync_cfg['translation']['stiffness']
    Kd_pos   = sync_cfg['translation']['damping']
    Kp_yaw   = sync_cfg['rotation']['stiffness']
    Kd_yaw   = sync_cfg['rotation']['damping']
    Kp_motor = sync_cfg['motor']['stiffness']

    twin_actuation = motor_cfg.get('twin_actuation', 'measured')

    active_sensors    = sensor_cfg['active']
    sensor_noise_stds = sensor_cfg['noise']

    # ------------------------------------------------------------------
    # 2. GENERATE XML
    # ------------------------------------------------------------------
    print("Generating XML from config...")
    generate_xml(SYSTEMPARAMETERS, config_path=config_path)
    model_path = paths_cfg['xml_filename']

    # ------------------------------------------------------------------
    # 3. LOAD FIELD DATA  [F2S]
    # ------------------------------------------------------------------
    print(f"Loading field data from: {paths_cfg['field_data']}")
    df = pd.read_csv(os.path.expandvars(paths_cfg['field_data']))

    # ------------------------------------------------------------------
    # 4. INITIALIZE MUJOCO
    # ------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    # ------------------------------------------------------------------
    # 5. DYNAMIC JOINT RESOLUTION
    # ------------------------------------------------------------------
    f0_qpos_idx, f0_qvel_idx = get_joint_indices(model, 0)
    f1_qpos_idx, f1_qvel_idx = get_joint_indices(model, 1)
    print(f"Fish 0 QPos Indices: {f0_qpos_idx}")
    print(f"Fish 1 QPos Indices: {f1_qpos_idx}")

    # ------------------------------------------------------------------
    # 6. INITIALIZE EKF
    # ------------------------------------------------------------------
    ekf = FishESEKF(
        dt=dt,
        active_sensors=active_sensors,
        process_noise_std=ekf_cfg['process_noise_std'],
        sensor_noise_stds=sensor_noise_stds,
        bias_process_noise=ekf_cfg['bias_process_noise'],
        initial_covariance=ekf_cfg['initial_covariance'],
        min_measurement_noise=ekf_cfg.get('min_measurement_noise', 1e-6),
    )

    # ------------------------------------------------------------------
    # 7. SIMULATION SETUP
    # ------------------------------------------------------------------
    width, height = 1200, 800
    renderer = mujoco.Renderer(model, height, width)

    mujoco.mj_forward(model, data)

    # [F2S FIX] Warm-start EKF and twin fish from first CSV row so that both
    # fish spawn at the same real-world position and the coupling forces are
    # non-zero and correctly directed from step 1 onward.
    first = df.iloc[0]
    ekf.x[9]  = first['Ghost_X']
    ekf.x[10] = first['Ghost_Y']
    ekf.x[0]  = first['Ghost_Yaw']
    data.qpos[f1_qpos_idx[0]] = first['Ghost_X']
    data.qpos[f1_qpos_idx[1]] = first['Ghost_Y']
    data.qpos[f1_qpos_idx[5]] = first['Ghost_Yaw']
    mujoco.mj_forward(model, data)

    log      = []
    deep_log = []
    images   = []
    frame_count = 0

    print(f"Starting Field-to-Sim Loop ({len(df)} steps at {1/dt:.0f} Hz)...")

    # ------------------------------------------------------------------
    # 8. MAIN LOOP
    # ------------------------------------------------------------------
    for i in range(len(df)):
        row = df.iloc[i]

        # --- A. GHOST FISH (Fish 0) - FORCED REPLAY  [F2S] ---
        # Spline columns used ONLY for visual teleportation, not for EKF.
        data.qpos[f0_qpos_idx[0]] = row['Ghost_X']
        data.qpos[f0_qpos_idx[1]] = row['Ghost_Y']
        data.qpos[f0_qpos_idx[5]] = row['Ghost_Yaw']
        data.qpos[f0_qpos_idx[6]] = row['Ghost_Alpha']

        data.qvel[f0_qvel_idx[0]] = row['Ghost_vX']
        data.qvel[f0_qvel_idx[1]] = row['Ghost_vY']
        data.qvel[f0_qvel_idx[5]] = row['Ghost_Omega']
        data.qvel[f0_qvel_idx[6]] = row['Ghost_dAlpha']

        # --- B. CAPTURE TWIN STATE (pre-step) ---
        twin_yaw_old    = data.qpos[f1_qpos_idx[5]]
        twin_vx_g_old   = data.qvel[f1_qvel_idx[0]]
        twin_vy_g_old   = data.qvel[f1_qvel_idx[1]]
        c_old, s_old    = np.cos(twin_yaw_old), np.sin(twin_yaw_old)
        twin_vx_l_old   = twin_vx_g_old * c_old + twin_vy_g_old * s_old
        twin_vy_l_old   = -twin_vx_g_old * s_old + twin_vy_g_old * c_old
        twin_omega_old  = data.qvel[f1_qvel_idx[5]]
        twin_alpha_old  = data.qpos[f1_qpos_idx[6]]
        twin_d_alpha_old = data.qvel[f1_qvel_idx[6]]
        twin_state_old  = np.array([twin_yaw_old, twin_vx_l_old, twin_vy_l_old,
                                    twin_omega_old, twin_alpha_old, twin_d_alpha_old])

        # --- C. MOTOR CONTROL ---
        # Ghost: set ctrl to match force-teleported position → zero actuator error
        data.ctrl[0] = row['Ghost_Alpha']

        # Twin actuation target depends on mode:
        #   'teleport'  → directly set motor qpos/qvel each step, zero tracking error.
        #                 Must match simulate_headless in opt_field2sim.py exactly.
        #   'measured'  → Ghost_Alpha (spline of real encoder): what the fish actually did.
        #                 In real-time RL, replace with live Meas_Alpha.
        #   'commanded' → Commanded_Alpha column if present, else falls back to Ghost_Alpha.
        #                 In real-time RL, replace with raw policy output (alpha_target).
        if twin_actuation == 'teleport':
            data.qpos[f1_qpos_idx[6]] = row['Ghost_Alpha']
            data.qvel[f1_qvel_idx[6]] = row['Ghost_dAlpha']
            data.ctrl[1] = row['Ghost_Alpha']
        elif twin_actuation == 'commanded' and 'Commanded_Alpha' in row.index:
            ctrl_target = row['Commanded_Alpha']
            data.ctrl[1] = ctrl_target + Kp_motor * (ekf.x[4] - data.qpos[f1_qpos_idx[6]])
        else:
            ctrl_target = row['Ghost_Alpha']
            data.ctrl[1] = ctrl_target + Kp_motor * (ekf.x[4] - data.qpos[f1_qpos_idx[6]])

        # --- D. SPATIAL COUPLING (EKF-driven; all stiffness=0 for SysID Step A) ---
        ekf_yaw      = ekf.x[0]
        ekf_vx_l     = ekf.x[1];  ekf_vy_l = ekf.x[2]
        ekf_omega    = ekf.x[3]
        ekf_px       = ekf.x[9];  ekf_py   = ekf.x[10]

        twin_x   = data.qpos[f1_qpos_idx[0]]
        twin_y   = data.qpos[f1_qpos_idx[1]]
        twin_vx_g = twin_vx_g_old
        twin_vy_g = twin_vy_g_old
        twin_omega = twin_omega_old

        c_twin, s_twin = np.cos(twin_yaw_old), np.sin(twin_yaw_old)
        ekf_vx_g = ekf_vx_l * c_twin - ekf_vy_l * s_twin
        ekf_vy_g = ekf_vx_l * s_twin + ekf_vy_l * c_twin

        error_yaw = (ekf_yaw - twin_yaw_old + np.pi) % (2 * np.pi) - np.pi

        data.qfrc_applied[f1_qvel_idx[0]] = (Kd_pos * (ekf_vx_g - twin_vx_g)
                                              + Kp_pos * (ekf_px - twin_x))
        data.qfrc_applied[f1_qvel_idx[1]] = (Kd_pos * (ekf_vy_g - twin_vy_g)
                                              + Kp_pos * (ekf_py - twin_y))
        data.qfrc_applied[f1_qvel_idx[5]] = (Kp_yaw * error_yaw
                                              + Kd_yaw * (ekf_omega - twin_omega))

        # --- E. STEP ---
        mujoco.mj_step(model, data)

        # --- F. CAPTURE TWIN STATE (post-step) ---
        twin_yaw     = data.qpos[f1_qpos_idx[5]]
        twin_vx_g    = data.qvel[f1_qvel_idx[0]]
        twin_vy_g    = data.qvel[f1_qvel_idx[1]]
        c, s         = np.cos(twin_yaw), np.sin(twin_yaw)
        twin_vx_l    = twin_vx_g * c + twin_vy_g * s
        twin_vy_l    = -twin_vx_g * s + twin_vy_g * c
        twin_omega   = data.qvel[f1_qvel_idx[5]]
        twin_alpha   = data.qpos[f1_qpos_idx[6]]
        twin_d_alpha = data.qvel[f1_qvel_idx[6]]
        twin_state_new = np.array([twin_yaw, twin_vx_l, twin_vy_l,
                                   twin_omega, twin_alpha, twin_d_alpha])

        twin_delta    = twin_state_new - twin_state_old
        twin_delta[0] = (twin_delta[0] + np.pi) % (2 * np.pi) - np.pi
        twin_delta[4] = (twin_delta[4] + np.pi) % (2 * np.pi) - np.pi

        # --- G. EKF PREDICT (every step, same as sim2sim) ---
        ekf.predict(twin_delta)

        # --- H. EKF UPDATE (only at real measurement timestamps)  [F2S] ---
        # Measurements come from actual field data, NOT from simulation sensors.
        # Vision_Update / IMU_Update flags mark the true measurement instants.
        z_dict = {}
        if row['Vision_Update']:
            if 'global_position' in active_sensors:
                z_dict['global_position'] = [row['Meas_X'], row['Meas_Y']]
            if 'heading' in active_sensors:
                z_dict['heading'] = row['Meas_Yaw']
        if row['IMU_Update']:
            if 'motor_encoder' in active_sensors:
                # Meas_Alpha = real encoder reading; Ghost_dAlpha = best available velocity proxy
                z_dict['motor_encoder'] = [row['Meas_Alpha'], row['Ghost_dAlpha']]
            if 'imu' in active_sensors:
                # Only Omega (GyrZ) is used by the ES-EKF; aX/aY slots are dummies
                z_dict['imu'] = [0.0, 0.0, row['Meas_GyrZ']]

        if z_dict:
            # EKF expects ALL active sensors in z_dict. For sensors whose measurement
            # did not arrive this step, inject the EKF's own prediction → zero residual,
            # no state change (sparse measurement handling). [F2S]
            if 'imu' in active_sensors and 'imu' not in z_dict:
                z_dict['imu'] = [0.0, 0.0, ekf.x[3] + ekf.x[8]]
            if 'motor_encoder' in active_sensors and 'motor_encoder' not in z_dict:
                z_dict['motor_encoder'] = [ekf.x[4], ekf.x[5]]
            if 'global_position' in active_sensors and 'global_position' not in z_dict:
                z_dict['global_position'] = [ekf.x[9], ekf.x[10]]
            if 'heading' in active_sensors and 'heading' not in z_dict:
                z_dict['heading'] = ekf.x[0]

            twin_accel = data.sensordata[6:8]   # accel_1 XY (same index as sim2sim)
            _, _, _, _ = ekf.update(z_dict, twin_accel)

        # --- I. LOGGING (100 Hz) ---
        if frame_count % 10 == 0:
            deep_row = {'Time': row['Time']}
            for idx, q_idx in enumerate(f0_qpos_idx):
                deep_row[f'F0_qpos_{idx}'] = data.qpos[q_idx]
            for idx, q_idx in enumerate(f1_qpos_idx):
                deep_row[f'F1_qpos_{idx}'] = data.qpos[q_idx]
            deep_row['Ctrl_F1_Motor'] = data.ctrl[1]
            deep_row['EKF_X']   = ekf.x[9]
            deep_row['EKF_Y']   = ekf.x[10]
            deep_row['EKF_Yaw'] = ekf.x[0]
            deep_log.append(deep_row)

            log.append({
                'Time':      row['Time'],
                'Ghost_X':   row['Ghost_X'],   'Ghost_Y':   row['Ghost_Y'],
                'Ghost_Yaw': row['Ghost_Yaw'],
                'Twin_X':    data.qpos[f1_qpos_idx[0]],
                'Twin_Y':    data.qpos[f1_qpos_idx[1]],
                'Twin_Yaw':  twin_yaw,
                'EKF_X':     ekf.x[9],
                'EKF_Y':     ekf.x[10],
            })

        # --- J. RENDER (~30 Hz) ---
        if frame_count % 33 == 0:
            renderer.update_scene(data, camera="overhead")
            images.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

        frame_count += 1

    # ------------------------------------------------------------------
    # 9. POST-PROCESSING
    # ------------------------------------------------------------------
    res_df  = pd.DataFrame(log)
    deep_df = pd.DataFrame(deep_log)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(paths_cfg['output_dir'], f"Run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    deep_df.to_csv(f"{out_dir}/deep_analysis_log.csv", index=False)
    print(f"Deep analysis log saved to {out_dir}/deep_analysis_log.csv")

    mse_pos = np.mean(
        (res_df['Ghost_X'] - res_df['Twin_X'])**2 +
        (res_df['Ghost_Y'] - res_df['Twin_Y'])**2
    )
    print(f"\nSimulation Finished! Final Position MSE: {mse_pos:.6f} m^2")

    # Trajectory plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['Ghost_X'], res_df['Ghost_Y'], 'b--', label='Field (Ghost)', linewidth=1.2)
    plt.plot(res_df['Twin_X'],  res_df['Twin_Y'],  'r-',  label='Sim (Twin)',    linewidth=1.2)
    plt.plot(res_df['EKF_X'],   res_df['EKF_Y'],   'g:',  label='EKF Estimate', linewidth=1.2)
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.legend(); plt.grid(True)
    plt.title(f"Trajectory Comparison | Position MSE: {mse_pos:.6f} m$^2$")
    plt.savefig(f"{out_dir}/trajectory.png", dpi=150)
    plt.close()

    # Video
    video = cv2.VideoWriter(
        f"{out_dir}/replay.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30, (width, height)
    )
    for img in images:
        video.write(img)
    video.release()

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    run_field2sim()


