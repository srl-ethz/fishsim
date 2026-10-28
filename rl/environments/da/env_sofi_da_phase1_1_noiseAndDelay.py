"""SOFI phase 1.1 — direct-action variant — sensor noise and vision latency.

Inherits BaseFishEnvDA (24-dim DA obs). Same 3-stage noise curriculum and
polar target distribution as the CPG phase 1.1 env.
"""

import os
import numpy as np
from gymnasium import utils

from .env_sofi_da import BaseFishEnvDA
from geometry.sofi_motorid import generate_xml, SYSTEMPARAMETERS


class FishEnv(BaseFishEnvDA, utils.EzPickle):

    # Optional per-variant fluid override. None → use the generator default
    # (refitB, from sofi_motorid). A subclass can set the 15-coef vector (e.g. E13a)
    # to train the same motor/curriculum on a different fluid model.
    FLUID_COEF = None

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
        # Apply the optional fluid override (FLUID_COEF class attr). Overridden
        # variants write a PID-namespaced XML so parallel sweeps with different
        # fluids never clobber the shared geometry/SOFI_motorid.xml.
        sp = dict(SYSTEMPARAMETERS)
        _geo = os.path.join(os.path.dirname(__file__), "..", "..", "..", "geometry")
        if type(self).FLUID_COEF is not None:
            sp["fluidCoef"] = [float(x) for x in type(self).FLUID_COEF]
            model_path = os.path.abspath(os.path.join(_geo, f"_sofi_da_{os.getpid()}.xml"))
        else:
            model_path = os.path.abspath(os.path.join(_geo, "SOFI_motorid.xml"))
        generate_xml(sp, filename=model_path)
        super().__init__(
            system_parameters=sp,
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
