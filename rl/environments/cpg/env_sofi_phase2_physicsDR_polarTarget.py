"""SOFI phase 2 with physics DR — polar target widening on top of phase 1.2.

Inherits phase 1.2 (full sensor noise + physics DR), then adds the same
four-stage polar target widening as env_sofi_phase2_polarTarget.  The
result is a curriculum that simultaneously maintains full DR robustness
while expanding the target distribution.

Intended as the continuation environment when finetuning from a phase 1.2
checkpoint (sensor noise + physics DR already mastered).

Curriculum
----------
All four stages run with full sensor noise AND full physics DR from the
start — both inherited from phase 1.2 stage 2 "full":

    fwd_narrow   (30°,  1–2 m)     reward_threshold 400
    fwd_wide     (60°,  0.5–2.5 m) reward_threshold 400
    semi_lateral (85°,  0.5–2 m)   reward_threshold 300
    lateral      (120°, 0.5–2.5 m) reward_threshold 250
    full_range   (180°, 0.3–4 m)   reward_threshold 150
"""

import numpy as np
from gymnasium import utils

from .env_sofi_phase1_2_physicsDR import FishEnv as _Phase1_2

_HEADING_REWARD_D0 = 0.5   # m — distance scale for heading reward fade near target

# Gated-progress reward: heading_gate = max(0, cos(heading_error))
# Reward is heading_gate * progress_weight * (prev_dist - curr_dist).
# This means the fish is only penalised for regressing when it is already
# facing the target; during a turn (gate ≈ 0) backward drift has no penalty.

_FULL_DR = dict(
    enable_init_vel=True, enable_inertial=True,
    enable_mechanics=True, enable_fluid=True,
)

_FULL_SENSOR = dict(
    enable_imu=True, enable_encoder=True,
    enable_vision=True, enable_vision_noise=True, enable_latency=True,
    latency_min=0.010, latency_max=0.050, frame_drop_prob=0.10,
)


class FishEnv(_Phase1_2, utils.EzPickle):
    """Phase 2 + physics DR — polar target widening with full DR throughout."""

    CURRICULUM_STAGES = [
        dict(name="fwd_narrow",   r_min=1.0, r_max=2.0, theta_max_deg=30.0,  reward_threshold=400.0,
             **_FULL_DR, **_FULL_SENSOR),
        dict(name="fwd_wide",     r_min=0.5, r_max=2.5, theta_max_deg=60.0,  reward_threshold=400.0,
             **_FULL_DR, **_FULL_SENSOR),
        dict(name="semi_lateral", r_min=0.5, r_max=2.0, theta_max_deg=85.0,  reward_threshold=300.0,
             **_FULL_DR, **_FULL_SENSOR),
        dict(name="lateral",      r_min=0.5, r_max=2.5, theta_max_deg=120.0, reward_threshold=250.0,
             **_FULL_DR, **_FULL_SENSOR),
        dict(name="full_range",   r_min=0.3, r_max=4.0, theta_max_deg=180.0, reward_threshold=150.0,
             **_FULL_DR, **_FULL_SENSOR),
    ]

    def __init__(
        self,
        r_min: float = 1.0,
        r_max: float = 2.0,
        theta_max_deg: float = 30.0,
        heading_reward_weight: float = 0.0,
        progress_reward_weight: float = 0.0,
        distance_reward_weight: float = 1.0,
        time_penalty_per_step: float  = 0.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, r_min=r_min, r_max=r_max, theta_max_deg=theta_max_deg,
            heading_reward_weight=heading_reward_weight,
            progress_reward_weight=progress_reward_weight,
            distance_reward_weight=distance_reward_weight,
            time_penalty_per_step=time_penalty_per_step,
            **kwargs,
        )
        super().__init__(**kwargs)
        self.r_min                   = float(r_min)
        self.r_max                   = float(r_max)
        self.theta_max               = float(np.deg2rad(theta_max_deg))
        self._heading_reward_weight  = float(heading_reward_weight)
        self._progress_reward_weight = float(progress_reward_weight)
        self._distance_reward_weight = float(distance_reward_weight)
        self._time_penalty_per_step  = float(time_penalty_per_step)
        self._prev_distance: float   = 0.0

    def step(self, action):
        prev_distance = self._prev_distance
        obs, reward, terminated, truncated, info = super().step(action)

        com_pos    = self.data.site("COM_0").xpos[:2]
        head_angle = self.data.joint("headRollZ_0").qpos[0]
        vec_world  = self.target - com_pos
        distance   = np.linalg.norm(vec_world)
        self._prev_distance = distance

        if distance > 1e-6:
            cos_err = -(np.cos(head_angle) * vec_world[0] + np.sin(head_angle) * vec_world[1]) / distance

            if not terminated:
                # Scale down the base -distance term if requested
                if self._distance_reward_weight != 1.0:
                    reward += distance * (1.0 - self._distance_reward_weight)

                # Heading alignment reward, faded to zero near target
                if self._heading_reward_weight > 0.0:
                    fade = 1.0 - np.exp(-distance / _HEADING_REWARD_D0)
                    reward += self._heading_reward_weight * cos_err * fade

                # Gated progress reward: penalise/reward progress only when facing target
                if self._progress_reward_weight > 0.0:
                    heading_gate = max(0.0, cos_err)
                    progress     = prev_distance - distance
                    reward      += self._progress_reward_weight * heading_gate * progress

                # Fixed time cost per step to incentivise reaching target quickly
                reward -= self._time_penalty_per_step

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        obs = super().reset_model()
        com_pos = self.data.site("COM_0").xpos[:2]
        self._prev_distance = float(np.linalg.norm(self.target - com_pos))
        return obs

    def _sample_target(self, initial_com: np.ndarray, rot: np.ndarray) -> np.ndarray:
        r     = self.np_random.uniform(self.r_min, self.r_max)
        theta = self.np_random.uniform(-self.theta_max, self.theta_max)
        return initial_com + rot @ np.array([-r * np.cos(theta), r * np.sin(theta)])
