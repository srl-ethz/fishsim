"""SOFI phase 1.2 — direct-action variant — physics domain randomization.

Same 24-dim DA obs as phase 1.1 (inherited). Adds per-episode physics DR
applied in-memory. Identical to the CPG phase 1.2 structure.
"""

import numpy as np
from gymnasium import utils

from .env_sofi_da_phase1_1_noiseAndDelay import FishEnv as _Phase1_1
from ...models.physics_model import PhysicsModel

_FULL_SENSOR = dict(
    enable_imu=True, enable_encoder=True,
    enable_vision=True, enable_vision_noise=True, enable_latency=True,
    enable_fps_dr=True,
    latency_min=0.010, latency_max=0.050, frame_drop_prob=0.10,
)


class FishEnv(_Phase1_1, utils.EzPickle):

    CURRICULUM_STAGES = [
        dict(name="init_vel",  reward_threshold=400,
             enable_init_vel=True,  enable_inertial=False,
             enable_mechanics=False, enable_fluid=False, enable_motor=True, **_FULL_SENSOR),
        dict(name="physical",  reward_threshold=350,
             enable_init_vel=True,  enable_inertial=True,
             enable_mechanics=True, enable_fluid=False,  enable_motor=True, **_FULL_SENSOR),
        dict(name="full",      reward_threshold=300,
             enable_init_vel=True,  enable_inertial=True,
             enable_mechanics=True, enable_fluid=True,   enable_motor=True, **_FULL_SENSOR),
    ]

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self._physics_model = PhysicsModel(
            enable_init_vel=False, enable_inertial=False,
            enable_mechanics=False, enable_fluid=False,
        )
        super().__init__(**kwargs)
        self._physics_model.store_originals(self.model)
        self._step_dr_log: dict = {}

    def reset_model(self):
        self._physics_model.restore_originals(self.model)
        self._physics_model.reset()
        self._physics_model.apply_to_model(self.model)

        # Motor dead-time DR: when motor DR is active, adopt the per-episode sampled
        # transport delay before super().reset_model() rebuilds the setpoint ring buffer.
        if self._physics_model.enable_motor:
            self._deadtime_steps = int(self._physics_model.motor_deadtime_steps)

        super().reset_model()

        init_vel = self._physics_model.get_init_vel()
        if np.any(init_vel != 0.0):
            qvel = self.data.qvel.copy()
            qvel[0] += init_vel[0]
            qvel[1] += init_vel[1]
            qvel[5] += init_vel[2]
            self.set_state(self.data.qpos, qvel)

        self._step_dr_log = self._physics_model.log_sampled_values()
        return self._get_obs()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self._step_dr_log)
        return obs, reward, terminated, truncated, info
