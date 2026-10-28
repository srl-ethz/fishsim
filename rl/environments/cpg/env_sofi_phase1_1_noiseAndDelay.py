"""SOFI phase 1.1 — sensor noise and vision latency.

Inherits BaseFishEnv (24-dim CPG obs) and enables all sensor corruption
channels by default. The curriculum runner can progressively disable
individual channels by passing enable_imu=False etc.

Switches to polar target sampling (1–2 m, ±30° forward arc).

Curriculum
----------
CURRICULUM_STAGES defines the canonical 3-stage noise progression.
The curriculum runner reads this attribute and applies each stage's flags
directly to env._sensor_model at runtime — no env rebuild required.

    Stage 0  "clean+delay"   : vision with 10–50 ms latency only.
                               Teaches the policy to use a frame buffer
                               before any sensor corruption is present.
    Stage 1  "enc+imu+delay" : adds encoder and IMU noise on top of delay.
    Stage 2  "full"          : adds AprilTag position/heading noise and
                               10 % frame-drop probability.

Each stage runs until REWARD_THRESHOLD is met (set in train_rl_curriculum.py)
or nSteps is exhausted.
"""

import os
import numpy as np
from gymnasium import utils

from .env_sofi import BaseFishEnv
from geometry.auto_SOFI_cpg import generate_xml, SYSTEMPARAMETERS


class FishEnv(BaseFishEnv, utils.EzPickle):

    CURRICULUM_STAGES = [
        dict(name="clean+delay",   reward_threshold=400,
             enable_imu=False, enable_encoder=False,
             enable_vision=True, enable_vision_noise=False, enable_latency=True,
             latency_min=0.010, latency_max=0.050, frame_drop_prob=0.00),
        dict(name="enc+imu+delay", reward_threshold=400,
             enable_imu=True,  enable_encoder=True,
             enable_vision=True, enable_vision_noise=False, enable_latency=True,
             latency_min=0.010, latency_max=0.050, frame_drop_prob=0.00),
        dict(name="full",          reward_threshold=400,
             enable_imu=True,  enable_encoder=True,
             enable_vision=True, enable_vision_noise=True,  enable_latency=True,
             latency_min=0.010, latency_max=0.050, frame_drop_prob=0.10),
    ]

    def __init__(
        self,
        enable_imu: bool = True,
        enable_encoder: bool = True,
        enable_vision: bool = True,
        enable_vision_noise: bool = True,
        enable_latency: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            enable_imu=enable_imu,
            enable_encoder=enable_encoder,
            enable_vision=enable_vision,
            enable_vision_noise=enable_vision_noise,
            enable_latency=enable_latency,
            **kwargs,
        )
        model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "geometry", "SOFI_cpg.xml"
        ))
        generate_xml(SYSTEMPARAMETERS, filename=model_path)
        super().__init__(
            system_parameters=SYSTEMPARAMETERS,
            model_path=model_path,
            enable_imu=enable_imu,
            enable_encoder=enable_encoder,
            enable_vision=enable_vision,
            enable_vision_noise=enable_vision_noise,
            enable_latency=enable_latency,
            **kwargs,
        )

    def _sample_target(self, initial_com: np.ndarray, rot: np.ndarray) -> np.ndarray:
        r     = self.np_random.uniform(1.0, 2.0)
        theta = self.np_random.uniform(-np.pi / 6, np.pi / 6)
        return initial_com + rot @ np.array([-r * np.cos(theta), r * np.sin(theta)])
