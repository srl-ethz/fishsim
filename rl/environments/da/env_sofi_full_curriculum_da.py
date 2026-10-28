"""SOFI DA-MotorID-S01 curriculum: Direct Action with the re-identified motor model.

Returns to the Direct-Action regime (post-S28), retiring the SDA/LPF detour. Key
differences from the SDA (LPF) curriculum it is based on:

1. NO action low-pass filter and NO gear-compliance EMA. The motor bandwidth comes
   entirely from the re-identified MuJoCo position actuator (kp=4.09, kv=0.340,
   armature=0.0044, forcerange ±2.3 in geometry/auto_SOFI_da.py; fluidID/scripts/
   23_motor_id.py, 2026-06-13) plus an explicit transport delay.

2. MOTOR TRANSPORT DELAY (motor_deadtime_steps, physics steps of 1 ms): identified
   ~28 ms; domain-randomized 20–40 ms (the cmd→meas lag is ~73 ms in RL deployment /
   ~60 ms field; the delay-vs-pole split is ambiguous without step inputs, so the
   delay band is randomized). Applied as a substep ring buffer on the setpoint.

3. MOTOR DR RECENTERED on the identified means: motor_kp_range and motor_kv_range are
   centred on kp≈4.09 / kv≈0.340 (was kp≈5 / kv≈0.45 for the SDA curriculum).

Curriculum structure: identical 10 stages to the SDA curriculum (DA-v4 lineage),
CPG-style reward (r = -distance + rewardSuccess).
"""

import numpy as np
from gymnasium import utils

from .env_sofi_da_phase1_2_physicsDR import FishEnv as _Phase1_2

_CLEAN = dict(
    enable_imu=False, enable_encoder=False,
    enable_vision=True, enable_vision_noise=False, enable_latency=True,
    latency_min=0.010, latency_max=0.050, frame_drop_prob=0.00,
)
_FULL_SENSOR = dict(
    enable_imu=True, enable_encoder=True,
    enable_vision=True, enable_vision_noise=True, enable_latency=True,
    latency_min=0.010, latency_max=0.050, frame_drop_prob=0.10,
    enable_fps_dr=True,
)

# Motor DR recentred on the re-identified joint-damping model (2026-06-19): kp≈5.41,
# kv≈0 (back-EMF moved to fixed joint damping b=0.48 in the XML → velocity cap ~4.8 rad/s).
# v_max now scales with forcerange (b fixed): forcerange_range [1.61–2.30] → v_max [3.4–4.8],
# physically coupled to voltage sag. armature (0.0113) + b are XML-fixed (not DR'd here).
# Early stages narrower; full/fluid stages wider.
_BASE_DR = dict(
    enable_init_vel=False, enable_inertial=True,
    enable_mechanics=True, enable_fluid=True, enable_motor=True,
    enable_extra_frames=True,
    fluid_std=0.10,
    motor_kp_range=[5.0, 7.0],
    motor_kv_range=[0.0, 0.10],
    motor_deadtime_range=[10, 16],
    motor_armature_range=[0.010, 0.013],
    motor_forcerange_range=[1.84, 2.30],
    motor_zero_offset_std=0.025,
)
_BASE_DR_INITVEL = dict(_BASE_DR, enable_init_vel=True)
_MOTOR_DR_FULL = dict(
    enable_init_vel=True, enable_inertial=True,
    enable_mechanics=True, enable_fluid=True, enable_motor=True,
    enable_extra_frames=True,
    fluid_std=0.10,
    motor_kp_range=[5.0, 7.0],
    motor_kv_range=[0.0, 0.12],
    motor_deadtime_range=[8, 18],
    motor_armature_range=[0.009, 0.014],
    motor_forcerange_range=[1.61, 2.30],
    motor_zero_offset_std=0.038,
)
_FLUID_15PCT = dict(_MOTOR_DR_FULL, fluid_std=0.15)
_FLUID_20PCT = dict(_MOTOR_DR_FULL, fluid_std=0.20)


class FishEnv(_Phase1_2, utils.EzPickle):
    """DA-MotorID-S01 curriculum: 10 stages (DA-v4 lineage), re-identified motor model."""

    FRAME_BUFFER_LEN   = 5
    TRACK_ACTUAL_MOTOR = True
    N_ACTIVE_AT_INIT   = 40

    CURRICULUM_STAGES = [
        dict(name="fwd60",          r_min=0.50, r_max=3.00, theta_max_deg=60.0,
             **_CLEAN, **_BASE_DR),
        dict(name="semi90",         r_min=0.50, r_max=3.00, theta_max_deg=90.0,
             **_CLEAN, **_BASE_DR),
        dict(name="lateral120",     r_min=0.50, r_max=3.00, theta_max_deg=120.0,
             reset_lr=True, **_CLEAN, **_BASE_DR),
        dict(name="wide150",        r_min=0.50, r_max=3.00, theta_max_deg=150.0,
             success_window=150, **_CLEAN, **_BASE_DR),
        dict(name="full180",        r_min=0.50, r_max=3.00, theta_max_deg=180.0,
             reset_lr=True, success_window=150, **_CLEAN, **_BASE_DR),
        dict(name="full180_noise",  r_min=0.50, r_max=3.00, theta_max_deg=180.0,
             reset_lr=True, success_window=150, **_FULL_SENSOR, **_BASE_DR_INITVEL),
        dict(name="motor_dr_full",  r_min=0.50, r_max=3.00, theta_max_deg=180.0,
             reset_lr=True, success_window=150, min_stage_steps=250_000,
             **_FULL_SENSOR, **_MOTOR_DR_FULL),
        dict(name="fluid_dr_15pct", r_min=0.50, r_max=3.00, theta_max_deg=180.0,
             reset_lr=True, success_window=200, min_stage_steps=250_000,
             **_FULL_SENSOR, **_FLUID_15PCT),
        dict(name="fluid_dr_20pct", r_min=0.50, r_max=3.00, theta_max_deg=180.0,
             reset_lr=True, success_window=200, min_stage_steps=250_000,
             **_FULL_SENSOR, **_FLUID_20PCT),
        dict(name="mastery_20pct",  r_min=0.50, r_max=3.00, theta_max_deg=180.0,
             reset_lr=True, success_window=500, min_stage_steps=500_000,
             **_FULL_SENSOR, **_FLUID_20PCT),
    ]

    def __init__(self, r_min=0.50, r_max=3.00, theta_max_deg=60.0,
                 use_distance_reward=True, **kwargs):
        utils.EzPickle.__init__(self, r_min=r_min, r_max=r_max,
                                theta_max_deg=theta_max_deg,
                                use_distance_reward=use_distance_reward,
                                **kwargs)
        super().__init__(use_distance_reward=use_distance_reward, **kwargs)
        self.r_min     = float(r_min)
        self.r_max     = float(r_max)
        self.theta_max = float(np.deg2rad(theta_max_deg))

    def _sample_target(self, initial_com: np.ndarray, rot: np.ndarray) -> np.ndarray:
        r     = self.np_random.uniform(self.r_min, self.r_max)
        theta = self.np_random.uniform(-self.theta_max, self.theta_max)
        return initial_com + rot @ np.array([-r * np.cos(theta), r * np.sin(theta)])
