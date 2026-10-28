"""Left-right mirrored replay buffer for DA SOFI observations (24-dim or 40-dim).

The SOFI fish is bilaterally symmetric about its forward (X) axis: any
trajectory reaching a target at angle +θ has an equally valid mirrored
counterpart at −θ.  Storing both halves of every transition doubles the
effective experience for behind-target scenarios — where real successes
are rare — at zero extra rollout cost.

Background: MAER-Nav (arXiv:2503.23908) showed that replay-buffer
starvation of rare backward-motion experiences is the primary reason
DRL fails to learn non-holonomic U-turn behaviour.  This buffer
addresses that directly by ensuring every behind-target success also
trains the mirrored direction, without requiring any additional episodes.

Mirror transform under a left-right (lateral Y-axis) reflection:
  cos(motorPos)    — unchanged (cos is even)
  sin(motorPos)    — NEGATE   (sin is odd)
  motorVel         — NEGATE
  imuAngularVel wz — NEGATE   (yaw rate reverses)
  dist             — unchanged (Euclidean distance is always positive)
  vecX             — unchanged (forward component does not flip)
  vecY             — NEGATE   (lateral component reverses)
  age              — unchanged (time is invariant)
  motor setpoints  — NEGATE   (past actions flip)
  motor actuals    — NEGATE   (actual positions flip; harmless when zeroed)

Action (motor setpoint) is negated.
Reward and termination flag are invariant to the mirror.

Obs-dim support:
  24-dim (FRAME_BUFFER_LEN=3, TRACK_ACTUAL_MOTOR=False — sweep 9 / angular curriculum):
    [0:4]   encoder+IMU
    [4:16]  3 × frame(dist, vecX, vecY, age)
    [16:24] 8 × motor setpoint

  40-dim (FRAME_BUFFER_LEN=5, TRACK_ACTUAL_MOTOR=True — sweep 14+ / prog curriculum):
    [0:4]   encoder+IMU
    [4:24]  5 × frame(dist, vecX, vecY, age)
    [24:40] 8 × (setpoint, actual) interleaved
              setpoints at even indices [24,26,...,38]
              actuals at  odd  indices [25,27,...,39]  (zeroed when motor DR inactive)
"""

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Union
import torch

# Observation indices that flip sign under a left-right body mirror.
# Only vecY (lateral) and motor history (actions) change sign; dist, vecX, age do not.

# 24-dim layout: motor history is 8 setpoints at [16:24].
_NEGATE_24 = np.array([
    1, 2, 3,                              # sin(motorPos), motorVel, imuAngularVel
    6, 10, 14,                            # frame_buf_{0,1,2} vecY
    16, 17, 18, 19, 20, 21, 22, 23,       # motor history (8 setpoints)
], dtype=np.intp)

# 40-dim layout: motor history is 8×(setpoint,actual) interleaved at [24:40].
# Extra frame slots [4:12] are zero-padded when inactive — negating zeros is harmless
# but including their vecY [6,10] ensures correctness when they become active (stage 7+).
_NEGATE_40 = np.array([
    1, 2, 3,                              # sin(motorPos), motorVel, imuAngularVel
    6, 10, 14, 18, 22,                    # all 5 frame slots vecY
    *range(24, 40),                       # full motor history (setpoints + actuals)
], dtype=np.intp)


class MirroredReplayBuffer(ReplayBuffer):
    """Adds a left-right mirrored copy of every transition.

    Supports 24-dim (sweep 9 angular curriculum) and 40-dim (progressive-obs
    curriculum) observations.  Raises ValueError for any other obs_dim.

    Use with buffer_size = 2 × desired_real_capacity so that half the
    buffer holds real transitions and half holds their mirrors, keeping
    the effective real-experience count equal to a standard buffer.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ) -> None:
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        obs_dim = observation_space.shape[0]
        if obs_dim == 24:
            self._negate = _NEGATE_24
        elif obs_dim == 40:
            self._negate = _NEGATE_40
        else:
            raise ValueError(
                f"MirroredReplayBuffer: unsupported obs_dim={obs_dim}. "
                "Expected 24 (angular curriculum) or 40 (prog-obs curriculum). "
                "Add a _NEGATE array for this layout in mirrored_replay_buffer.py."
            )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)

        obs_m      = obs.copy()
        next_obs_m = next_obs.copy()
        obs_m[..., self._negate]      *= -1
        next_obs_m[..., self._negate] *= -1

        super().add(obs_m, next_obs_m, -action, reward, done, infos)
