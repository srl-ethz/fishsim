"""Base SOFI Gymnasium environment — 24-dim CPG observation.

Observation (24 values):
    Motor encoder : cos/sin position + velocity      (3)
    IMU gyroscope : angular velocity wz (yaw rate)   (1)
    CPG phase     : cos/sin of swimming phase         (2)
    Frame buffer  : 3 × (dist, vecX, vecY, age_s)   (12)
    Action hist.  : 3 × (omega, theta_center)        (6)

Action (6 values, CPG parameter vector [-1, 1]^6):
    [k1c, k1s, k2c, k2s, omega, theta_center]

Subclasses override _sample_target to change the target distribution and
pass enable_* flags to configure sensor corruption in SensorModel.
"""

import os
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
import mujoco

from .cpg_oscillator import CPGOscillator, K1_MAX, K2_MAX, OMEGA_MAX, THETA_CENTER_MAX, FILTER_TAU
from ...models.sensor_model import SensorModel, FRAME_DROP_PROB

DEFAULT_ASSETS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "geometry", "meshes")

_FRAME_BUFFER_LEN   = 3
_AGE_SENTINEL_S     = 0.200
_ACTION_HISTORY_LEN = 3


class BaseFishEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        system_parameters,
        model_path,
        frame_skip=20,
        render_mode=None,
        seed=None,
        noterminate=False,
        width=800,
        height=800,
        enable_imu=False,
        enable_encoder=False,
        enable_vision=True,
        enable_vision_noise=False,
        enable_latency=False,
        motorVelRange=None,
        actionMultiplier=None,
        rewardFailure=None,
        **kwargs,
    ):
        model_timestep = system_parameters.get("dt", 0.001)
        self.metadata = self.metadata.copy()
        self.metadata["render_fps"] = int(np.round(1.0 / (model_timestep * frame_skip)))

        self.model_filename = model_path
        self.noterminate    = noterminate

        # 24-dim obs: encoder(3) + imu_wz(1) + cpg(2) + frame_buffer(12) + action_hist(6)
        total_dim         = 4 + 2 + _FRAME_BUFFER_LEN * 4 + _ACTION_HISTORY_LEN * 2
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        self.target        = np.array([0.0, 0.0])
        self.prevAction    = np.zeros((1,), dtype=np.float32)
        self.prevVecTarget = np.array([0.0, 0.0])
        self._last_fish_vx = 0.0   # fish-frame forward vel (>0 = backward); set in _get_obs

        # Indices into the 24-dim output obs
        self.observationIndices = {
            "cos(motorEncoderPos)": range(0,  1),
            "sin(motorEncoderPos)": range(1,  2),
            "motorEncoderVel":      range(2,  3),
            "imuAngularVel":        range(3,  4),
            "cpg_phase":            range(4,  6),
            "frame_buffer_0":       range(6,  10),
            "frame_buffer_1":       range(10, 14),
            "frame_buffer_2":       range(14, 18),
            "action_history_0":     range(18, 20),
            "action_history_1":     range(20, 22),
            "action_history_2":     range(22, 24),
        }
        self.observationIndices["distanceTarget"] = range(14, 15)
        self.observationIndices["vecTarget"]      = range(15, 17)
        self.observationIndices["motorJointVel"]  = self.observationIndices["motorEncoderVel"]

        # Indices into the internal 10-dim base obs consumed by SensorModel
        _base = {
            "cos(motorEncoderPos)": 1,
            "sin(motorEncoderPos)": 1,
            "motorEncoderVel":      1,
            "imuAngularVel":        1,
            "distanceTarget":       1,
            "vecTarget":            2,
            "velVecTarget":         2,
            "prevAction":           1,
        }
        self._base_obs_indices: dict = {}
        idx = 0
        for key, size in _base.items():
            self._base_obs_indices[key] = range(idx, idx + size)
            idx += size

        self._scales = np.array(
            [K1_MAX, K1_MAX, K2_MAX, K2_MAX, OMEGA_MAX, THETA_CENTER_MAX],
            dtype=np.float64,
        )

        self._sensor_model = SensorModel(
            dt=model_timestep * frame_skip,
            camera_fps=30.0,
            enable_imu=enable_imu,
            enable_encoder=enable_encoder,
            enable_vision=enable_vision,
            enable_vision_noise=enable_vision_noise,
            enable_latency=enable_latency,
            frame_drop_prob=FRAME_DROP_PROB,
        )
        self._action_history: list = []
        self._step_noise_log: dict = {}

        utils.EzPickle.__init__(
            self,
            system_parameters=system_parameters,
            model_path=model_path,
            frame_skip=frame_skip,
            render_mode=render_mode,
            noterminate=noterminate,
            enable_imu=enable_imu,
            enable_encoder=enable_encoder,
            enable_vision=enable_vision,
            enable_vision_noise=enable_vision_noise,
            enable_latency=enable_latency,
            motorVelRange=motorVelRange,
            actionMultiplier=actionMultiplier,
            rewardFailure=rewardFailure,
            **kwargs,
        )
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name="tracking_0",
            **kwargs,
        )

        # MujocoEnv.__init__ sets action_space from model.actuator_ctrlrange (1D);
        # override it here with the full 6D CPG parameter vector.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # CPG requires self.model — must be initialised after MujocoEnv.__init__
        self._cpg = CPGOscillator(
            dt_phys=self.model.opt.timestep,
            tau=FILTER_TAU,
            motor_limit=system_parameters["motor_ctrlrange"],
        )

        if seed is not None:
            self.action_space.seed(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        a      = np.asarray(action, dtype=np.float64)
        scaled = a * self._scales
        scaled[4] = 0.5 * (a[4] + 1.0) * OMEGA_MAX
        return scaled

    def _build_frame_buffer_obs(self) -> np.ndarray:
        current_time = self._sensor_model._sim_time
        buf = list(self._sensor_model._frame_buffer)
        while len(buf) < _FRAME_BUFFER_LEN:
            buf.insert(0, (0.0, 0.0, 0.0, current_time - _AGE_SENTINEL_S))
        result = []
        for dist, vx, vy, t in buf[-_FRAME_BUFFER_LEN:]:
            result.extend([dist, vx, vy, float(np.clip(current_time - t, 0.0, _AGE_SENTINEL_S))])
        return np.array(result, dtype=np.float32)

    def _build_action_history_obs(self) -> np.ndarray:
        hist = list(self._action_history)
        while len(hist) < _ACTION_HISTORY_LEN:
            hist.insert(0, np.zeros(2, dtype=np.float32))
        result = []
        for entry in hist[-_ACTION_HISTORY_LEN:]:
            result.extend(entry)
        return np.array(result, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        motor_pos = self.data.joint("motor_0").qpos
        motor_vel = self.data.joint("motor_0").qvel
        imu_wz    = self.data.qvel[5:6]

        com_pos    = self.data.site("COM_0").xpos[:2]
        head_angle = self.data.joint("headRollZ_0").qpos[0]

        vec_target_world     = self.target - com_pos
        vel_vec_target_world = (vec_target_world - self.prevVecTarget) / self.dt
        self.prevVecTarget   = vec_target_world.copy()

        rot = np.array([
            [ np.cos(head_angle), np.sin(head_angle)],
            [-np.sin(head_angle), np.cos(head_angle)],
        ])
        vec_target_local     = rot @ vec_target_world
        vel_vec_target_local = rot @ vel_vec_target_world
        distance_to_target   = np.linalg.norm(vec_target_local)
        # Forward velocity in body frame (same computation as env_sofi_da): >0 = backward.
        self._last_fish_vx   = float(-vel_vec_target_local[0])

        base_obs = np.concatenate([
            [np.cos(motor_pos[0]), np.sin(motor_pos[0])],
            motor_vel,
            imu_wz,
            [distance_to_target],
            vec_target_local,
            vel_vec_target_local,
            self.prevAction,
        ]).astype(np.float32)

        obs_noisy, self._step_noise_log = self._sensor_model.corrupt_obs(
            base_obs, self._base_obs_indices
        )

        cpg_phase = self._cpg.phase % (2 * np.pi)
        return np.concatenate([
            obs_noisy[:4],
            [np.cos(cpg_phase), np.sin(cpg_phase)],
            self._build_frame_buffer_obs(),
            self._build_action_history_obs(),
        ]).astype(np.float32)

    def _get_info(self) -> dict:
        com_pos = self.data.site("COM_0").xpos[:2]
        return {"distance_to_target": float(np.linalg.norm(self.target - com_pos))}

    def _sample_target(self, initial_com: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """Override in subclasses to change the target distribution."""
        return initial_com + rot @ np.array([
            self.np_random.uniform(-2.0, -1.0),
            self.np_random.uniform(-0.5,  0.5),
        ])

    def step(self, action):
        self._action_history.append(
            np.array([float(action[4]), float(action[5])], dtype=np.float32)
        )
        if len(self._action_history) > _ACTION_HISTORY_LEN:
            self._action_history.pop(0)

        desired = self._scale_action(action)
        for _ in range(self.frame_skip):
            self.data.ctrl[0] = self._cpg.step(desired)
            mujoco.mj_step(self.model, self.data)

        obs      = self._get_obs()
        info     = self._get_info()
        distance = info["distance_to_target"]

        terminated = (not self.noterminate) and bool(distance < 0.05)
        reward = float(
            -distance
            - 0.1 * np.linalg.norm(action)
            + (1000.0 if terminated else 0.0)
        )
        info.update(self._step_noise_log)

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def reset_model(self):
        self._sensor_model.reset()
        self._cpg.reset()
        self._action_history = []

        self.set_state(self.init_qpos, self.init_qvel)

        initial_com       = self.data.site("COM_0").xpos[:2]
        initial_head_angle = self.data.joint("headRollZ_0").qpos[0]
        rot = np.array([
            [ np.cos(initial_head_angle), -np.sin(initial_head_angle)],
            [ np.sin(initial_head_angle),  np.cos(initial_head_angle)],
        ])

        self.target = self._sample_target(initial_com, rot)

        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.model.site(target_site_id).pos[:2] = self.target
        self.data.site(target_site_id).xpos[:2] = self.target

        self.prevAction    = np.zeros((1,), dtype=np.float32)
        self.prevVecTarget = self.target - initial_com

        return self._get_obs()
