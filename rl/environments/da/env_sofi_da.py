"""Base SOFI Gymnasium environment — configurable-dim direct-action observation.

Base observation layout (FRAME_BUFFER_LEN=3, TRACK_ACTUAL_MOTOR=False → 24 dims):
    Motor encoder : cos/sin position + velocity              (3)
    IMU gyroscope : angular velocity wz (yaw rate)           (1)
    Frame buffer  : 3 × (dist, vecX, vecY, age_s)           (12)
    Motor history : 8 × filtered_setpoint ã_t (norm. [-1,1]) (8)

Extended layout (FRAME_BUFFER_LEN=5, TRACK_ACTUAL_MOTOR=True → 40 dims):
    Motor encoder : cos/sin position + velocity              (3)
    IMU gyroscope : angular velocity wz (yaw rate)           (1)
    Frame buffer  : 5 × (dist, vecX, vecY, age_s)           (20)
    Motor history : 8 × [filtered_setpoint ã_t, actual]     (16)

    Extra frame slots (indices 0–1 in the 5-slot window, i.e. the oldest) and
    motor actuals are zero-padded during curriculum stages where they carry no
    informative signal, controlled by two instance flags:
      _extra_frames_active  (default False) — set True when fluid DR is active
      _motor_dr_active      (default False) — set True when motor DR is active

    Zero-padding exploits the zero-gradient property: ∂L/∂W_i = (∂L/∂h)·0 = 0,
    so inactive weights are never updated and do not corrupt the learning signal.
    This is standard practice in curriculum RL (RAPID, PLR, Dreamer).

Motor model (re-identified 2026-06-13, fluidID/scripts/23_motor_id.py):
    The action LPF and gear-compliance EMA were REMOVED — both double-counted the
    velocity saturation the MuJoCo position actuator already models. Bandwidth is now
    set purely by the identified actuator (kp=4.09, kv=0.340, armature=0.0044,
    forcerange ±2.3 in geometry/auto_SOFI_da.py), and the only software addition is a
    pure transport delay (motor_deadtime_steps ≈28 ms; cmd→meas lag ~73 ms RL / ~60 ms
    field, delay-vs-pole ambiguous → DR'd 20–40 ms), applied as a substep-level ring
    buffer on the setpoint. Motor history stores the
    raw policy action a_t; the encoder channel shows where the motor physically is.

Action (1 value — direct motor position setpoint):
    a ∈ [-1, 1]  →  θ_motor = a × motor_ctrlrange  (then delayed by the dead time)

Reward (use_distance_reward=False, default):
    r = (d_{t-1} − d_t)                          progress (potential-based)
        + α_fwd · max(0, −vx_fish)               forward speed bonus
        − α_bw  · max(0,  vx_fish)               backward speed penalty (linear)
        − α_h   · EMA|heading_error|             heading alignment
        − α_jerk·|ã_t − 3ã_{t-1} + 3ã_{t-2} − ã_{t-3}|  smoothness (on filtered)
        − α_dist · d_t                           distance (default 0)
        + rewardSuccess·𝟙[reached]

    heading_error = atan2(vecY_local, -vecX_local), EMA α=0.1.
    Fish nose is in the -x body direction, so negating x maps nose-to-target → error=0,
    tail-to-target → error=π (maximum penalty). Sign was inverted before S28.
    α_bw is LINEAR (not quadratic) — α_bw ≥ 0.30 closes the fast-backward exploit.

Reward (use_distance_reward=True, SDA default):
    r = −d_t + rewardSuccess·𝟙[reached] − α_jerk·|...|

    Classic CPG-style reward.  With lpf_cutoff_hz > 0 and α_jerk=0 this is the
    minimal two-term reward: distance penalty + success bonus.
"""

from collections import deque

import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
import mujoco

from ...models.sensor_model import SensorModel, FRAME_DROP_PROB

_FRAME_BUFFER_LEN  = 3
_AGE_SENTINEL_S    = 0.200
_MOTOR_HISTORY_LEN = 8


class BaseFishEnvDA(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    # Subclasses can override these to change observation layout without
    # touching this file.  All methods read from type(self) so inheritance
    # picks up the override automatically.
    FRAME_BUFFER_LEN   = _FRAME_BUFFER_LEN   # number of vision frames in obs
    MOTOR_HISTORY_LEN  = _MOTOR_HISTORY_LEN  # number of history steps
    TRACK_ACTUAL_MOTOR = False               # if True, store (setpoint, actual) pairs

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
        alpha_jerk=0.01,
        alpha_heading=0.0,
        alpha_dist=0.0,
        alpha_fwd=0.0,
        alpha_backward=0.0,
        backward_vel_limit=0.0,
        success_radius=0.1,
        use_distance_reward=False,
        lpf_cutoff_hz=0.0,
        motor_deadtime_steps=0,
        motorVelRange=None,
        actionMultiplier=None,
        rewardFailure=None,
        rewardSuccess=1000.0,
        camera_fps=30.0,
        **kwargs,
    ):
        model_timestep = system_parameters.get("dt", 0.001)
        self.metadata = self.metadata.copy()
        self.metadata["render_fps"] = int(np.round(1.0 / (model_timestep * frame_skip)))

        self.model_filename       = model_path
        self.noterminate          = noterminate
        self.alpha_jerk            = float(alpha_jerk)
        self.alpha_heading         = float(alpha_heading)
        self.alpha_dist            = float(alpha_dist)
        self.alpha_fwd             = float(alpha_fwd)
        self.alpha_backward        = float(alpha_backward)
        self.backward_vel_limit    = float(backward_vel_limit)
        self._success_radius       = float(success_radius)
        self.use_distance_reward  = bool(use_distance_reward)
        self._last_fish_vx: float = 0.0
        self._ep_step_count: int  = 0
        self._reward_success      = float(rewardSuccess)
        self._reward_failure      = float(rewardFailure) if rewardFailure is not None else -1.0
        self._motor_ctrlrange     = float(system_parameters.get("motor_ctrlrange", 1.0))

        # Motor transport delay (identified ≈28 ms; DR'd 20–40 ms per episode by the
        # physics_model when motor DR is active; fluidID/scripts/23_motor_id.py).
        # Applied at the substep level (exact at any frame_skip):
        # the setpoint reaching data.ctrl is delayed by motor_deadtime_steps physics
        # steps. MuJoCo has no native input delay, so it lives here AND in the replay/
        # fluid-ID driver — NOT in the XML.
        self._deadtime_steps: int = int(motor_deadtime_steps)
        self._ctrl_buffer: deque = deque([0.0] * (self._deadtime_steps + 1),
                                         maxlen=self._deadtime_steps + 1)

        # First-order low-pass filter on the motor SETPOINT (the "DA-FOF" policy of the paper).
        # It removes the high-frequency command the policy would otherwise use to exploit the
        # velocity²-scaled fluid coefficients (see docs/sim2real_motor_fluid_lpf_findings.md) and
        # band-limits the command to the ≤4 Hz regime the fluid model was identified on.
        #   ã = (1−α)·ã_{-1} + α·a ,  α = ωc·T/(1+ωc·T);  lpf_cutoff_hz=0 → pass-through (plain DA).
        _fc = float(lpf_cutoff_hz)
        _T  = float(system_parameters.get("dt", 0.001)) * frame_skip   # control-step period
        self._lpf_cutoff_hz: float = _fc
        _wc = 2.0 * np.pi * _fc
        self._lpf_alpha: float = (_wc * _T / (1.0 + _wc * _T)) if _fc > 0.0 else 1.0
        self._filtered_action: float = 0.0          # filter state ã_{-1}

        # EMA state for absolute heading penalty (reset each episode)
        self._last_heading_error: float = 0.0
        self._heading_ema:        float = 0.0

        _fb_len  = type(self).FRAME_BUFFER_LEN
        _mh_len  = type(self).MOTOR_HISTORY_LEN
        _mh_dims = _mh_len * (2 if type(self).TRACK_ACTUAL_MOTOR else 1)
        # obs layout: encoder(3) + imu(1) + frame_buffer(_fb_len*4) + motor_history(_mh_dims)
        total_dim         = 3 + 1 + _fb_len * 4 + _mh_dims
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        self.target        = np.array([0.0, 0.0])
        self.prevVecTarget = np.array([0.0, 0.0])

        # Indices into the 10-dim base obs consumed by SensorModel
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

        self._sensor_model = SensorModel(
            dt=model_timestep * frame_skip,
            camera_fps=float(camera_fps),
            enable_imu=enable_imu,
            enable_encoder=enable_encoder,
            enable_vision=enable_vision,
            enable_vision_noise=enable_vision_noise,
            enable_latency=enable_latency,
            frame_drop_prob=FRAME_DROP_PROB,
        )
        # Deque pre-filled with zeros — always full length, jerk always computable
        self._motor_history: deque = deque([0.0] * _mh_len, maxlen=_mh_len)
        if type(self).TRACK_ACTUAL_MOTOR:
            self._motor_actual_history: deque = deque([0.0] * _mh_len, maxlen=_mh_len)
        # Set to True by the curriculum trainer when enable_motor=True.  When False,
        # actual motor positions are zeroed in the observation — the actuals only carry
        # useful per-episode Kp identification signal when motor DR is active.
        self._motor_dr_active: bool = False
        # Set to True by the curriculum trainer when enable_extra_frames=True (fluid DR
        # active). When False, frame buffer slots beyond the base _FRAME_BUFFER_LEN are
        # zero-padded — they only carry drag-identification signal under fluid DR.
        self._extra_frames_active: bool = False
        self._step_noise_log: dict = {}

        # Named index map — computed dynamically so subclasses that change
        # FRAME_BUFFER_LEN or TRACK_ACTUAL_MOTOR get the correct layout.
        _fb_start = 4
        _mh_start = _fb_start + _fb_len * 4
        self.observationIndices = {
            "cos(motorEncoderPos)": range(0, 1),
            "sin(motorEncoderPos)": range(1, 2),
            "motorEncoderVel":      range(2, 3),
            "imuAngularVel":        range(3, 4),
            **{f"frame_buffer_{i}": range(_fb_start + i*4, _fb_start + (i+1)*4)
               for i in range(_fb_len)},
            "motor_history": range(_mh_start, _mh_start + _mh_dims),
            # distanceTarget: dist component of the most recent frame buffer slot
            "distanceTarget": range(_fb_start + (_fb_len-1)*4, _fb_start + (_fb_len-1)*4 + 1),
        }
        self.observationIndices["motorJointVel"] = self.observationIndices["motorEncoderVel"]

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
            alpha_jerk=alpha_jerk,
            alpha_heading=alpha_heading,
            alpha_dist=alpha_dist,
            alpha_fwd=alpha_fwd,
            alpha_backward=alpha_backward,
            backward_vel_limit=backward_vel_limit,
            success_radius=success_radius,
            use_distance_reward=use_distance_reward,
            lpf_cutoff_hz=lpf_cutoff_hz,
            motor_deadtime_steps=motor_deadtime_steps,
            rewardSuccess=rewardSuccess,
            camera_fps=camera_fps,
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

        # Override MujocoEnv's 1-D actuator action_space with our explicit [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        if seed is not None:
            self.action_space.seed(seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_frame_buffer_obs(self) -> np.ndarray:
        fb_len = type(self).FRAME_BUFFER_LEN
        current_time = self._sensor_model._sim_time
        buf = list(self._sensor_model._frame_buffer)
        while len(buf) < fb_len:
            buf.insert(0, (0.0, 0.0, 0.0, current_time - _AGE_SENTINEL_S))
        # Extra slots beyond the base _FRAME_BUFFER_LEN (=3) are the oldest frames
        # in the window. Zero-pad them when inactive using the same sentinel as empty
        # slots: (dist=0, vec=0, age=AGE_SENTINEL_S). The zero-gradient property of
        # padded inputs ensures these weights are never updated during inactive stages.
        n_extra = fb_len - _FRAME_BUFFER_LEN
        result = []
        for i, (dist, vx, vy, t) in enumerate(buf[-fb_len:]):
            if not self._extra_frames_active and i < n_extra:
                result.extend([0.0, 0.0, 0.0, _AGE_SENTINEL_S])
            else:
                result.extend([dist, vx, vy, float(np.clip(current_time - t, 0.0, _AGE_SENTINEL_S))])
        return np.array(result, dtype=np.float32)

    def _build_motor_history_obs(self) -> np.ndarray:
        """Returns motor history oldest→newest.

        TRACK_ACTUAL_MOTOR=False: shape (MOTOR_HISTORY_LEN,), setpoints only.
        TRACK_ACTUAL_MOTOR=True:  shape (MOTOR_HISTORY_LEN*2,), interleaved
            [setpoint_0, actual_0, setpoint_1, actual_1, ...] oldest→newest.
        """
        setpoints = list(self._motor_history)
        if not type(self).TRACK_ACTUAL_MOTOR:
            return np.array(setpoints, dtype=np.float32)
        # Zero actuals when motor DR is inactive: Kp is fixed at the nominal value
        # across all episodes, so actuals carry no per-episode identification signal
        # and only add gradient noise during the navigation bootstrap stages.
        if not self._motor_dr_active:
            actuals = [0.0] * len(setpoints)
        else:
            actuals = list(self._motor_actual_history)
        interleaved = [v for pair in zip(setpoints, actuals) for v in pair]
        return np.array(interleaved, dtype=np.float32)

    def _compute_jerk(self) -> float:
        """3rd-order finite difference: |a_t - 3a_{t-1} + 3a_{t-2} - a_{t-3}|.

        Uses the four most recent entries of _motor_history (already updated
        with a_t before this is called). Returns 0 only if all four are zero
        (episode start), which is numerically correct.
        """
        h = list(self._motor_history)
        return float(abs(h[-1] - 3.0 * h[-2] + 3.0 * h[-3] - h[-4]))

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

        # Negate x so error=0 when nose (-x body) faces target, π when tail faces target.
        # Fish nose is in -x body direction; atan2(y, -x) maps nose-to-target → 0.
        self._last_heading_error = float(np.arctan2(vec_target_local[1], -vec_target_local[0]))
        # Forward velocity in body frame: vel_vec_target_local[0] = -v_fish_forward
        self._last_fish_vx = float(-vel_vec_target_local[0])

        prev_action = np.array([list(self._motor_history)[-1]], dtype=np.float32)

        base_obs = np.concatenate([
            [np.cos(motor_pos[0]), np.sin(motor_pos[0])],
            motor_vel,
            imu_wz,
            [distance_to_target],
            vec_target_local,
            vel_vec_target_local,
            prev_action,
        ]).astype(np.float32)

        obs_noisy, self._step_noise_log = self._sensor_model.corrupt_obs(
            base_obs, self._base_obs_indices
        )

        return np.concatenate([
            obs_noisy[:4],                    # encoder(3) + imu(1)
            self._build_frame_buffer_obs(),   # frame buffer(12)
            self._build_motor_history_obs(),  # motor history(8)
        ]).astype(np.float32)

    def _get_info(self) -> dict:
        com_pos = self.data.site("COM_0").xpos[:2]
        return {"distance_to_target": float(np.linalg.norm(self.target - com_pos))}

    def _sample_target(self, initial_com: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """Default target — override in subclasses."""
        return initial_com + rot @ np.array([
            self.np_random.uniform(-2.0, -1.0),
            self.np_random.uniform(-0.5,  0.5),
        ])

    def _lpf(self, a: float) -> float:
        """Filter one raw action sample → motor setpoint (first-order; α=1 → pass-through)."""
        self._filtered_action = ((1.0 - self._lpf_alpha) * self._filtered_action
                                 + self._lpf_alpha * a)
        return self._filtered_action

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))

        # Setpoint filter. History stores the FILTERED setpoint so
        # the policy observes the actual command, not the raw output it chose.
        filt  = self._lpf(a)
        self._motor_history.append(filt)
        jerk  = self._compute_jerk()
        theta = filt * self._motor_ctrlrange

        # Transport delay: push the constant per-step setpoint through a substep-level
        # ring buffer, so the value reaching data.ctrl is delayed by the identified
        # dead time (motor_deadtime_steps physics steps). Shifts the command staircase;
        # exact at any dt. With motor_deadtime_steps=0 the buffer is pass-through.
        for _ in range(self.frame_skip):
            self._ctrl_buffer.append(theta)
            self.data.ctrl[0] = self._ctrl_buffer[0]
            mujoco.mj_step(self.model, self.data)

        if type(self).TRACK_ACTUAL_MOTOR:
            actual_norm = float(np.clip(
                self.data.joint("motor_0").qpos[0] / self._motor_ctrlrange, -1.0, 1.0
            ))
            self._motor_actual_history.append(actual_norm)

        obs      = self._get_obs()
        info     = self._get_info()
        distance = info["distance_to_target"]

        terminated = (not self.noterminate) and bool(distance < self._success_radius)

        if self.use_distance_reward:
            reward = float(
                -distance
                + (self._reward_success if terminated else 0.0)
                - self.alpha_jerk * jerk
            )
        else:
            progress = self._prev_distance - distance
            self._prev_distance = distance

            # Absolute heading penalty: EMA of |heading_error| in [0, π].
            # Penalises being misaligned at every step — discourages detours and
            # sideways motion regardless of whether heading is improving or not.
            # Delta-based angular_progress only rewarded turning; this also penalises
            # NOT turning. EMA (τ=0.1) removes tail-beat oscillation noise.
            self._heading_ema = 0.1 * abs(self._last_heading_error) + 0.9 * self._heading_ema

            reward = float(
                progress
                - self.alpha_heading * self._heading_ema
                + (self._reward_success if terminated else 0.0)
                - self.alpha_jerk * jerk
                - self.alpha_dist * distance
            )
        # Per-step locomotion terms — applied every step, no grace period.
        # α_bw is LINEAR: α_bw ≥ 0.30 makes backward swimming net-negative even
        # for fast backward swimming (0.09 m/s) toward a close behind target.
        vx_bw  = max(0.0,  self._last_fish_vx)   # positive when swimming backward
        vx_fwd = max(0.0, -self._last_fish_vx)   # positive when swimming forward
        reward -= self.alpha_backward * vx_bw
        reward += self.alpha_fwd * vx_fwd

        self._ep_step_count += 1
        # Hard backward termination (grace period covers init-vel DR decay).
        # Disabled in S27 (backward_vel_limit=0.0); kept for backward compatibility.
        if self._ep_step_count > 50 and self.backward_vel_limit > 0.0 and vx_bw > self.backward_vel_limit:
            reward += self._reward_failure
            info.update(self._step_noise_log)
            info["jerk"] = jerk
            info["is_success"] = False
            return obs, reward, True, False, info

        info.update(self._step_noise_log)
        info["jerk"] = jerk
        info["is_success"] = terminated

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def reset_model(self):
        self._ep_step_count           = 0
        self._filtered_action         = 0.0   # reset filter state each episode
        self._ctrl_buffer = deque([0.0] * (self._deadtime_steps + 1),
                                  maxlen=self._deadtime_steps + 1)
        mh_len = type(self).MOTOR_HISTORY_LEN
        self._sensor_model.reset()
        self._motor_history = deque([0.0] * mh_len, maxlen=mh_len)
        if type(self).TRACK_ACTUAL_MOTOR:
            self._motor_actual_history = deque([0.0] * mh_len, maxlen=mh_len)

        self.set_state(self.init_qpos, self.init_qvel)

        initial_com        = self.data.site("COM_0").xpos[:2]
        initial_head_angle = self.data.joint("headRollZ_0").qpos[0]
        rot = np.array([
            [ np.cos(initial_head_angle), -np.sin(initial_head_angle)],
            [ np.sin(initial_head_angle),  np.cos(initial_head_angle)],
        ])

        self.target = self._sample_target(initial_com, rot)
        self._prev_distance = float(np.linalg.norm(self.target - initial_com))

        target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.model.site(target_site_id).pos[:2]  = self.target
        self.data.site(target_site_id).xpos[:2]  = self.target

        self.prevVecTarget = self.target - initial_com

        obs = self._get_obs()
        # Warm-start EMA at |initial heading error| so the first step's penalty
        # reflects actual misalignment rather than the zero-initialised EMA.
        self._heading_ema = abs(self._last_heading_error)
        return obs
