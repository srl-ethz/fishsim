"""Shared sensor corruption model used by all SOFI phase-1+ environments."""

import numpy as np

VISION_LATENCY_MIN_S = 0.010
VISION_LATENCY_MAX_S = 0.050
FRAME_DROP_PROB       = 0.10

_FRAME_BUFFER_LEN = 5


class SensorModel:
    """Phase 1.1 sensor corruption with feature flags and a rolling frame buffer.

    Supports selectively enabling/disabling IMU, encoder, and vision noise and
    optionally modeling vision pipeline latency. A rolling _frame_buffer stores
    the last _FRAME_BUFFER_LEN delivered frames for use by frame-buffer envs.
    """

    def __init__(self, dt, camera_fps=53.18, rng=None, *,
                 enable_imu=True, enable_encoder=True,
                 enable_vision=True, enable_vision_noise=True,
                 enable_fps_dr=True, enable_latency=True,
                 frame_drop_prob=FRAME_DROP_PROB):
        self.dt            = dt
        self.camera_period = 1.0 / camera_fps  # overwritten each episode by reset()
        self.rng           = rng if rng is not None else np.random.default_rng()

        self.enable_imu          = bool(enable_imu)
        self.enable_encoder      = bool(enable_encoder)
        self.enable_vision       = bool(enable_vision)
        self.enable_vision_noise = bool(enable_vision_noise)
        # enable_fps_dr: per-episode FPS draw + within-episode jitter (DR stages only).
        # When False, camera runs at fixed camera_fps with no jitter — use in clean stages.
        self.enable_fps_dr       = bool(enable_fps_dr)
        self.enable_latency      = bool(enable_latency)
        self.frame_drop_prob     = float(frame_drop_prob)
        # May-20 deployment measurements (module-05): mean=53.18 Hz, range=[39.4, 66.9]
        self.camera_fps_range      = (39.4, 66.9)
        self.camera_fps_jitter_std = 0.01247
        self._fps_episode          = camera_fps
        self.latency_min         = VISION_LATENCY_MIN_S
        self.latency_max         = VISION_LATENCY_MAX_S

        self.imu_noise_std         = 0.006
        self.imu_bias_std          = 0.005
        self.imu_bias              = None

        self.encoder_pos_noise_std = 0.001
        self.encoder_vel_noise_std = 0.01

        self.tag_pos_noise_std     = 0.002
        self.tag_vel_noise_std     = 0.02
        self.tag_heading_noise_std = np.deg2rad(1.5)

        self._camera_acc     = 0.0
        self._sim_time       = 0.0
        self._pending_frames = []
        self._last_delivered = None
        self._frame_buffer   = []  # list of (dist, vecX, vecY, delivery_sim_time)

    def reset(self):
        if self.enable_imu:
            self.imu_bias = self.rng.normal(0.0, self.imu_bias_std, size=1)
        else:
            self.imu_bias = np.zeros(1, dtype=np.float64)

        self._sim_time       = 0.0
        self._pending_frames = []
        self._last_delivered = None
        self._frame_buffer   = []
        if self.enable_vision:
            if self.enable_fps_dr:
                self._fps_episode = float(self.rng.uniform(*self.camera_fps_range))
                self.camera_period = 1.0 / self._fps_episode
            # randomise initial phase even at fixed FPS so episodes don't all start in-sync
            self._camera_acc = self.rng.uniform(0.0, self.camera_period)
        else:
            self._fps_episode  = self.camera_fps_range[0]  # vision off: pin to range floor
            self.camera_period = 1.0 / self._fps_episode
            self._camera_acc   = 0.0

    def corrupt_obs(self, obs_clean, obs_indices):
        obs_noisy = obs_clean.copy()
        sampled_values = {
            "imu_noise_std":         0.0,
            "imu_bias":              0.0,
            "encoder_pos_noise":     0.0,
            "encoder_vel_noise":     0.0,
            "tag_pos_noise":         0.0,
            "tag_heading_noise":     0.0,
            "vision_latency_frames": 0,
        }

        # IMU
        imu_idx = obs_indices["imuAngularVel"]
        if self.enable_imu:
            imu_noise = self.rng.normal(0.0, self.imu_noise_std, size=1)
            obs_noisy[imu_idx] = obs_clean[imu_idx] + self.imu_bias + imu_noise
            sampled_values["imu_noise_std"] = float(np.linalg.norm(imu_noise))
            sampled_values["imu_bias"]      = float(np.linalg.norm(self.imu_bias))
        else:
            obs_noisy[imu_idx] = obs_clean[imu_idx]

        # Encoder
        cos_idx = obs_indices["cos(motorEncoderPos)"]
        sin_idx = obs_indices["sin(motorEncoderPos)"]
        vel_idx = obs_indices["motorEncoderVel"]
        if self.enable_encoder:
            pos_noise = self.rng.normal(0.0, self.encoder_pos_noise_std)
            noisy_pos = np.arctan2(obs_clean[sin_idx][0], obs_clean[cos_idx][0]) + pos_noise
            obs_noisy[cos_idx] = np.cos(noisy_pos)
            obs_noisy[sin_idx] = np.sin(noisy_pos)
            vel_noise = self.rng.normal(0.0, self.encoder_vel_noise_std)
            obs_noisy[vel_idx] = obs_clean[vel_idx] + vel_noise
            sampled_values["encoder_pos_noise"] = float(np.abs(pos_noise))
            sampled_values["encoder_vel_noise"] = float(np.abs(vel_noise))
        else:
            obs_noisy[cos_idx] = obs_clean[cos_idx]
            obs_noisy[sin_idx] = obs_clean[sin_idx]
            obs_noisy[vel_idx] = obs_clean[vel_idx]

        # AprilTag / Vision pipeline
        dist_idx    = obs_indices["distanceTarget"]
        vec_idx     = obs_indices["vecTarget"]
        vel_vec_idx = obs_indices["velVecTarget"]

        _clean_vision = np.array([
            obs_clean[dist_idx][0],
            obs_clean[vec_idx][0], obs_clean[vec_idx][1],
            obs_clean[vel_vec_idx][0], obs_clean[vel_vec_idx][1],
        ], dtype=np.float32)

        if not self.enable_vision:
            delayed_vision = _clean_vision
            sampled_values["tag_pos_noise"]         = 0.0
            sampled_values["tag_heading_noise"]     = 0.0
            sampled_values["vision_latency_frames"] = 0
        else:
            self._sim_time   += self.dt
            self._camera_acc += self.dt

            if self._camera_acc >= self.camera_period:
                if self.enable_fps_dr:
                    jitter_interval = max(
                        1.0 / 60.0,
                        self.rng.normal(self.camera_period, self.camera_fps_jitter_std)
                    )
                else:
                    jitter_interval = self.camera_period
                self._camera_acc -= jitter_interval

                if self.rng.random() < self.frame_drop_prob:
                    sampled_values["tag_pos_noise"]     = 0.0
                    sampled_values["tag_heading_noise"] = 0.0
                else:
                    if self.enable_vision_noise:
                        tag_pos_noise     = self.rng.normal(0.0, self.tag_pos_noise_std, size=2)
                        tag_heading_noise = self.rng.normal(0.0, self.tag_heading_noise_std)
                        vel_noise         = self.rng.normal(0.0, self.tag_vel_noise_std, size=2)
                    else:
                        tag_pos_noise     = np.zeros(2)
                        tag_heading_noise = 0.0
                        vel_noise         = np.zeros(2)

                    vec_target = np.array([obs_clean[vec_idx][0], obs_clean[vec_idx][1]], dtype=np.float64)
                    vec_target += tag_pos_noise
                    if np.linalg.norm(vec_target) > 1e-6 and tag_heading_noise != 0.0:
                        rot = np.array([
                            [np.cos(tag_heading_noise), -np.sin(tag_heading_noise)],
                            [np.sin(tag_heading_noise),  np.cos(tag_heading_noise)],
                        ])
                        vec_target = rot @ vec_target

                    frame = np.array([
                        np.linalg.norm(vec_target),
                        vec_target[0], vec_target[1],
                        obs_clean[vel_vec_idx][0] + vel_noise[0],
                        obs_clean[vel_vec_idx][1] + vel_noise[1],
                    ], dtype=np.float32)

                    delay = self.rng.uniform(self.latency_min, self.latency_max) if self.enable_latency else 0.0
                    self._pending_frames.append((frame, self._sim_time + delay))
                    sampled_values["tag_pos_noise"]     = float(np.linalg.norm(tag_pos_noise))
                    sampled_values["tag_heading_noise"] = float(tag_heading_noise)
            else:
                sampled_values["tag_pos_noise"]     = 0.0
                sampled_values["tag_heading_noise"] = 0.0

            arrived = [(f, t) for f, t in self._pending_frames if t <= self._sim_time]
            if arrived:
                self._last_delivered, _ = max(arrived, key=lambda x: x[1])
                self._pending_frames    = [(f, t) for f, t in self._pending_frames if t > self._sim_time]
                self._frame_buffer.append((
                    float(self._last_delivered[0]),
                    float(self._last_delivered[1]),
                    float(self._last_delivered[2]),
                    self._sim_time,
                ))
                if len(self._frame_buffer) > _FRAME_BUFFER_LEN:
                    self._frame_buffer.pop(0)

            delayed_vision = self._last_delivered if self._last_delivered is not None else _clean_vision
            sampled_values["vision_latency_frames"] = len(self._pending_frames)

        obs_noisy[dist_idx]    = delayed_vision[0]
        obs_noisy[vec_idx]     = delayed_vision[1:3]
        obs_noisy[vel_vec_idx] = delayed_vision[3:5]

        return obs_noisy, sampled_values
