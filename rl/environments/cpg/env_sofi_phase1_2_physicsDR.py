"""SOFI phase 1.2 — phase 1.1 sensor noise + physics domain randomization.

Same 24-dim obs as phase 1.1 (inherited). Adds per-episode physics DR
applied in-memory to the loaded MuJoCo model — no XML regeneration at
runtime.

Initial pose jitter is intentionally absent: the target is always sampled
relative to the fish's initial COM and heading, so absolute pose variation
is redundant. Initial *velocity* jitter is used instead — it tests whether
the policy can arrest unwanted motion it didn't cause.

Multi-process safety: XML is written once per process at __init__ from
identical SYSTEMPARAMETERS; all DR is applied to each process's own
in-memory MjModel with no cross-process file interaction.

Curriculum
----------
Three DR stages, all running with full sensor noise (phase 1.2 is trained
after phase 1.1 has already reached the "full" sensor noise stage).

    Stage 0  "init_vel"   : random starting velocity only (vx ±0.15 m/s,
                            vy ±0.15 m/s, wz ±0.30 rad/s).
                            Teaches the policy to stabilise from
                            arbitrary initial motion before any physics
                            variation is introduced.

    Stage 1  "physical"   : + head body mass ±5%, inertia ±5% per axis,
                            COM ±5 mm; tail hinge stiffness ±15%, damping
                            ±15%.  The fish feels and swims differently
                            each episode.

    Stage 2  "full"       : + fluid-coefficient multipliers ±20% per geom.
                            The hardest stage — drag/lift uncertainty is
                            the dominant source of the sim-to-real gap.

The curriculum runner applies DR flags to env._physics_model and sensor
flags to env._sensor_model at each stage transition.
"""

import numpy as np
from gymnasium import utils

from .env_sofi_phase1_1_noiseAndDelay import FishEnv as _Phase1_1
from ...models.physics_model import PhysicsModel

_FULL_SENSOR = dict(
    enable_imu=True, enable_encoder=True,
    enable_vision=True, enable_vision_noise=True, enable_latency=True,
    latency_min=0.010, latency_max=0.050, frame_drop_prob=0.10,
)


class FishEnv(_Phase1_1, utils.EzPickle):
    """Phase 1.2 — 24-dim obs with full sensor noise and staged physics DR."""

    CURRICULUM_STAGES = [
        dict(name="init_vel",  reward_threshold=400,
             enable_init_vel=True,  enable_inertial=False,
             enable_mechanics=False, enable_fluid=False,
             **_FULL_SENSOR),
        dict(name="physical",  reward_threshold=350,
             enable_init_vel=True,  enable_inertial=True,
             enable_mechanics=True, enable_fluid=False,
             **_FULL_SENSOR),
        dict(name="full",      reward_threshold=300,
             enable_init_vel=True,  enable_inertial=True,
             enable_mechanics=True, enable_fluid=True,
             **_FULL_SENSOR),
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

        super().reset_model()

        init_vel = self._physics_model.get_init_vel()
        if np.any(init_vel != 0.0):
            qvel = self.data.qvel.copy()
            qvel[0] += init_vel[0]   # vx
            qvel[1] += init_vel[1]   # vy
            qvel[5] += init_vel[2]   # wz
            self.set_state(self.data.qpos, qvel)

        self._step_dr_log = self._physics_model.log_sampled_values()
        return self._get_obs()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self._step_dr_log)
        return obs, reward, terminated, truncated, info
