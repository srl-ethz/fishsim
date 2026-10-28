"""Phase 1.2 physics domain randomization model.

Four independently togglable perturbation groups, controlled by enable_*
flags so the curriculum runner can progressively activate them:

    enable_init_vel   — randomised starting velocity (vx, vy, wz)
    enable_inertial   — head body mass ±5%, inertia ±5%, COM ±5 mm
    enable_mechanics  — tail hinge stiffness ±15%, damping ±15%
    enable_fluid      — fluid-coefficient multipliers ±20% per geom

The curriculum runner sets flags between episodes (same pattern as
SensorModel.enable_*).  All perturbation values are always sampled on
reset() so no flag change can leave a parameter uninitialised; groups
whose flag is False simply don't write to the model.
"""

import numpy as np
import mujoco


class PhysicsModel:

    def __init__(self, rng=None,
                 enable_init_vel: bool  = True,
                 enable_inertial: bool  = True,
                 enable_mechanics: bool = True,
                 enable_fluid: bool     = True,
                 enable_motor: bool     = True):
        self.rng = rng if rng is not None else np.random.default_rng()

        self.enable_init_vel  = bool(enable_init_vel)
        self.enable_inertial  = bool(enable_inertial)
        self.enable_mechanics = bool(enable_mechanics)
        self.enable_fluid     = bool(enable_fluid)

        # Perturbation magnitudes (σ for Gaussian multipliers / additive)
        self.init_vel_trans_std  = 0.10   # m/s  — vx, vy
        self.init_vel_rot_std    = 0.30   # rad/s — wz
        self.mass_std            = 0.05   # fractional
        self.inertia_std         = 0.05   # fractional, per principal axis
        self.com_jitter_std      = 0.005  # m, per axis
        self.stiffness_std       = 0.15   # fractional
        self.damping_std         = 0.15   # fractional
        self.fluid_std           = 0.20   # fractional, per geom

        self.enable_motor = bool(enable_motor)
        # Asymmetric: hardware cannot exceed stall ceiling (2.3 Nm) but voltage sag /
        # heat can reduce it by up to 40%.  Range is [0.6×2.3, 2.3] = [1.38, 2.3] Nm.
        self.motor_forcerange_range = (1.38, 2.3)
        # Uniform ranges from module-06 2-D (Kp, Kv) period-averaged calibration
        self.motor_kp_range         = (0.7, 10.4)   # [0.7, 10.4] Nm/rad (per-stage override)
        self.motor_kv_range         = (0.0, 0.6)    # [0.0, 0.6] Nm·s/rad (per-stage override)
        # Motor transport delay in physics steps (1 ms each). Per-stage override; nominal
        # identified ~28 ms (23_motor_id.py 2026-06-13). NOT a model param — the env reads
        # this and rebuilds its setpoint ring buffer each episode.
        self.motor_deadtime_range   = (28, 28)
        # Reflected rotor inertia (motor bandwidth). Small DR: fits span 0.0098–0.0137.
        # Affects how fast the motor reaches the velocity cap, not the cap itself.
        self.motor_armature_range   = (0.0113, 0.0113)   # per-stage override
        self.motor_zero_offset_std  = 0.05
        self.motor_forcerange       = 2.3
        self.motor_kp               = 4.09   # identified RL-primary (23_motor_id.py 2026-06-13)
        self.motor_kv               = 0.340
        self.motor_armature         = 0.0113
        self.motor_deadtime_steps   = 28
        self.motor_zero_offset      = 0.0
        self._motor_actuator_idx     = -1
        self._motor_dof              = -1
        self._original_forcerange    = None
        self._original_gainprm       = None
        self._original_biasprm       = None
        self._original_dof_armature  = None

        # Sampled values (set by reset())
        self.init_vel             = np.zeros(3)   # [vx, vy, wz]
        self.mass_multiplier      = 1.0
        self.inertia_multipliers  = np.ones(3)
        self.com_jitter           = np.zeros(3)
        self.stiffness_multiplier = 1.0
        self.damping_multiplier   = 1.0
        self.fluid_multipliers    = np.ones(1)

        # Stored originals (populated by store_originals())
        self._original_body_mass     = None
        self._original_body_inertia  = None
        self._original_body_ipos     = None
        self._original_jnt_stiffness = None
        self._original_dof_damping   = None
        self._original_geom_fluid    = None
        self._fluid_geom_ids         = None
        self._fish_body_id           = None

    def store_originals(self, model):
        self._original_body_mass    = model.body_mass.copy()
        self._original_body_inertia = model.body_inertia.copy()
        self._original_body_ipos    = model.body_ipos.copy()
        self._fish_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fish_0")

        if hasattr(model, "jnt_stiffness"):
            self._original_jnt_stiffness = model.jnt_stiffness.copy()
        if hasattr(model, "dof_damping"):
            self._original_dof_damping = model.dof_damping.copy()
        if hasattr(model, "geom_fluid"):
            self._original_geom_fluid = model.geom_fluid.copy()
            self._fluid_geom_ids = [
                gid for gid in range(model.ngeom)
                if np.any(model.geom_fluid[gid] != 0)
            ]
        self._motor_actuator_idx = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_0"
        )
        if self._motor_actuator_idx >= 0:
            idx = self._motor_actuator_idx
            self._original_forcerange = model.actuator_forcerange[idx].copy()
            self._original_gainprm    = model.actuator_gainprm[idx].copy()
            self._original_biasprm    = model.actuator_biasprm[idx].copy()
        mjid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "motor_0")
        if mjid >= 0 and hasattr(model, "dof_armature"):
            self._motor_dof = int(model.jnt_dofadr[mjid])
            self._original_dof_armature = model.dof_armature.copy()

    def restore_originals(self, model):
        if self._original_body_mass is not None:
            model.body_mass[:]     = self._original_body_mass
        if self._original_body_inertia is not None:
            model.body_inertia[:]  = self._original_body_inertia
        if self._original_body_ipos is not None:
            model.body_ipos[:]     = self._original_body_ipos
        if self._original_jnt_stiffness is not None:
            model.jnt_stiffness[:] = self._original_jnt_stiffness
        if self._original_dof_damping is not None:
            model.dof_damping[:]   = self._original_dof_damping
        if self._original_geom_fluid is not None:
            model.geom_fluid[:]    = self._original_geom_fluid
        if self._motor_actuator_idx >= 0 and self._original_forcerange is not None:
            idx = self._motor_actuator_idx
            model.actuator_forcerange[idx] = self._original_forcerange
            model.actuator_gainprm[idx]    = self._original_gainprm
            model.actuator_biasprm[idx]    = self._original_biasprm
        if self._original_dof_armature is not None:
            model.dof_armature[:] = self._original_dof_armature

    def reset(self):
        """Sample all perturbation values for the coming episode.

        Values are always sampled regardless of enable_* flags so no flag
        change can leave a parameter uninitialised.  Disabled groups have
        their values set to neutral (zero velocity, unit multipliers).
        """
        if self.enable_init_vel:
            self.init_vel = np.array([
                self.rng.normal(0.0, self.init_vel_trans_std),
                self.rng.normal(0.0, self.init_vel_trans_std),
                self.rng.normal(0.0, self.init_vel_rot_std),
            ])
        else:
            self.init_vel = np.zeros(3)

        if self.enable_inertial:
            self.mass_multiplier     = float(self.rng.normal(1.0, self.mass_std))
            self.inertia_multipliers = self.rng.normal(1.0, self.inertia_std, size=3)
            self.com_jitter          = self.rng.normal(0.0, self.com_jitter_std, size=3)
        else:
            self.mass_multiplier     = 1.0
            self.inertia_multipliers = np.ones(3)
            self.com_jitter          = np.zeros(3)

        if self.enable_mechanics:
            self.stiffness_multiplier = float(self.rng.normal(1.0, self.stiffness_std))
            self.damping_multiplier   = float(self.rng.normal(1.0, self.damping_std))
        else:
            self.stiffness_multiplier = 1.0
            self.damping_multiplier   = 1.0

        n = len(self._fluid_geom_ids) if self._fluid_geom_ids else 3
        if self.enable_fluid:
            self.fluid_multipliers = self.rng.normal(1.0, self.fluid_std, size=n)
        else:
            self.fluid_multipliers = np.ones(n)

        if self.enable_motor:
            self.motor_forcerange  = float(self.rng.uniform(*self.motor_forcerange_range))
            self.motor_kp         = float(self.rng.uniform(*self.motor_kp_range))
            self.motor_kv         = float(self.rng.uniform(*self.motor_kv_range))
            self.motor_armature   = float(self.rng.uniform(*self.motor_armature_range))
            self.motor_deadtime_steps = int(round(self.rng.uniform(*self.motor_deadtime_range)))
            self.motor_zero_offset = float(self.rng.uniform(
                -self.motor_zero_offset_std, self.motor_zero_offset_std
            ))
        else:
            # Nominal (motor DR off) — XML values bind via apply_to_model skip; kept consistent.
            self.motor_forcerange  = 2.3
            self.motor_kp          = 5.41
            self.motor_kv          = 0.0
            self.motor_armature    = 0.0113
            self.motor_deadtime_steps = 14
            self.motor_zero_offset = 0.0

    def apply_to_model(self, model):
        """Write sampled multipliers into the in-memory MjModel."""
        joint_names = ["joint0_0", "joint1_0", "joint2_0", "joint3_0", "joint4_0"]

        if self.enable_inertial and self._fish_body_id is not None and self._fish_body_id >= 0:
            fid = self._fish_body_id
            model.body_mass[fid]       *= self.mass_multiplier
            model.body_inertia[fid, :] *= self.inertia_multipliers
            model.body_ipos[fid, :]    += self.com_jitter

        if self.enable_mechanics and hasattr(model, "jnt_stiffness"):
            for name in joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    model.jnt_stiffness[jid] *= self.stiffness_multiplier

        if self.enable_mechanics and hasattr(model, "dof_damping") and hasattr(model, "jnt_dofadr"):
            for name in joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    dof = int(model.jnt_dofadr[jid])
                    if 0 <= dof < len(model.dof_damping):
                        model.dof_damping[dof] *= self.damping_multiplier

        if self.enable_fluid and self._fluid_geom_ids is not None and hasattr(model, "geom_fluid"):
            for idx, gid in enumerate(self._fluid_geom_ids):
                if idx < len(self.fluid_multipliers):
                    model.geom_fluid[gid] *= self.fluid_multipliers[idx]

        if self.enable_motor and self._motor_actuator_idx >= 0:
            idx = self._motor_actuator_idx
            fr = self.motor_forcerange
            kp = self.motor_kp
            kv = self.motor_kv
            model.actuator_forcerange[idx]    = [-fr, fr]
            model.actuator_gainprm[idx, 0]    = kp
            model.actuator_biasprm[idx, 1]    = -kp
            model.actuator_biasprm[idx, 2]    = -kv
            if self._motor_dof >= 0:
                model.dof_armature[self._motor_dof] = self.motor_armature

    def get_init_vel(self) -> np.ndarray:
        """Return [vx, vy, wz] velocity perturbation for the initial qvel."""
        return self.init_vel.copy()

    def log_sampled_values(self) -> dict:
        return {
            "dr/init_vel_norm":     float(np.linalg.norm(self.init_vel)),
            "dr/mass_mult":         self.mass_multiplier,
            "dr/inertia_mult_mean": float(np.mean(self.inertia_multipliers)),
            "dr/com_jitter_norm":   float(np.linalg.norm(self.com_jitter)),
            "dr/stiffness_mult":    self.stiffness_multiplier,
            "dr/damping_mult":      self.damping_multiplier,
            "dr/fluid_mult_mean":   float(np.mean(self.fluid_multipliers)),
            "dr/motor_forcerange":   self.motor_forcerange,
            "dr/motor_kp":           self.motor_kp,
            "dr/motor_kv":           self.motor_kv,
            "dr/motor_deadtime_steps": self.motor_deadtime_steps,
            "dr/motor_zero_offset":  self.motor_zero_offset,
        }
