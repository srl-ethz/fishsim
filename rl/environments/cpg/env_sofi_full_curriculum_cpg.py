"""SOFI full pipeline curriculum — phases 1.1 → 1.2 → 2 in a single run.

Stage list (10 stages):
    Phase 1.1 — sensor noise curriculum (fixed narrow target)
        0  clean+delay        : vision latency only
        1  enc+imu+delay      : + encoder/IMU noise
        2  full               : + AprilTag noise + frame-drop

    Phase 1.2 — physics DR curriculum (full sensor noise, fixed narrow target)
        3  init_vel           : random starting velocity  [LR reset]
        4  physical           : + inertial/mass/stiffness DR
        5  full_dr            : + fluid-coefficient DR

    Phase 2 — target widening (full DR + full sensor noise)
        6  fwd_wide           : 60°,  0.5–2.5 m           [LR reset]
        7  semi_lateral       : 85°,  0.5–2.5 m
        8  lateral            : 120°, 0.5–2.5 m
        9  full_range         : 180°, 0.5–4 m

Phase 2 starts at fwd_wide (not fwd_narrow) because the env default polar
target (r_min=1.0, r_max=2.0, theta_max_deg=30°) is identical to fwd_narrow,
so that stage would be redundant after phase 1.2 "full_dr".

LR is reset to the initial learning rate at phase boundaries (stages 3 and 6)
instead of being scaled down, because each phase introduces fundamentally new
challenges. The runner recognises the reset_lr key in the stage dict.
"""

import numpy as np
from gymnasium import utils

from .env_sofi_phase2_physicsDR_polarTarget import FishEnv as _Phase2PhysDR

_FULL_SENSOR = dict(
    enable_imu=True, enable_encoder=True,
    enable_vision=True, enable_vision_noise=True, enable_latency=True,
    latency_min=0.010, latency_max=0.050, frame_drop_prob=0.10,
)

_FULL_DR = dict(
    enable_init_vel=True, enable_inertial=True,
    enable_mechanics=True, enable_fluid=True,
)


class FishEnv(_Phase2PhysDR, utils.EzPickle):
    """Full pipeline curriculum: sensor noise → physics DR → target widening."""

    CURRICULUM_STAGES = [
        # ── Phase 1.1: sensor noise curriculum ───────────────────────────────
        dict(name="clean+delay",
             enable_imu=False, enable_encoder=False,
             enable_vision=True, enable_vision_noise=False, enable_latency=True,
             latency_min=0.010, latency_max=0.050, frame_drop_prob=0.00,
             enable_init_vel=False, enable_inertial=False,
             enable_mechanics=False, enable_fluid=False),
        dict(name="enc+imu+delay",
             enable_imu=True,  enable_encoder=True,
             enable_vision=True, enable_vision_noise=False, enable_latency=True,
             latency_min=0.010, latency_max=0.050, frame_drop_prob=0.00,
             enable_init_vel=False, enable_inertial=False,
             enable_mechanics=False, enable_fluid=False),
        dict(name="full_noise",
             **_FULL_SENSOR,
             enable_init_vel=False, enable_inertial=False,
             enable_mechanics=False, enable_fluid=False),

        # ── Phase 1.2: physics DR curriculum ─────────────────────────────────
        dict(name="init_vel",  reset_lr=True,
             enable_init_vel=True,  enable_inertial=False,
             enable_mechanics=False, enable_fluid=False,
             **_FULL_SENSOR),
        dict(name="physical",
             enable_init_vel=True,  enable_inertial=True,
             enable_mechanics=True, enable_fluid=False,
             **_FULL_SENSOR),
        dict(name="full_dr",
             **_FULL_DR, **_FULL_SENSOR),

        # ── Phase 2: target widening ──────────────────────────────────────────
        dict(name="fwd_wide",     r_min=0.5, r_max=2.5, theta_max_deg=60.0,
             reset_lr=True,
             **_FULL_DR, **_FULL_SENSOR),
        dict(name="semi_lateral", r_min=0.5, r_max=2.5, theta_max_deg=85.0,
             **_FULL_DR, **_FULL_SENSOR),
        dict(name="lateral",      r_min=0.5, r_max=2.5, theta_max_deg=120.0,
             **_FULL_DR, **_FULL_SENSOR),
        dict(name="full_range",   r_min=0.5, r_max=4.0, theta_max_deg=180.0,
             **_FULL_DR, **_FULL_SENSOR),
    ]

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        super().__init__(**kwargs)
