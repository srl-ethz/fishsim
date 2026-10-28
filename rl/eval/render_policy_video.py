"""Render an mp4 of the trained policy navigating to a fixed target angle.

Useful for inspecting how the policy handles targets behind (θ≈180°) or to
the side of the fish.  Target angle uses the same body-frame convention as
the curriculum env:  θ=0 → straight ahead,  θ=±180 → directly behind.

Usage:
    python rl/render_policy_video.py <model.zip> [options]

Examples:
    # Target directly behind, stage 6 (soft_dr), 3 episodes
    python rl/render_policy_video.py outputs/checkpoints/yca3hlq9/SAC_curriculum_1800000_steps.zip \\
        --stage 6 --theta_deg 170 --radius 1.5 --n_episodes 3

    # Forward target at stage 4 for comparison
    python rl/render_policy_video.py ... --stage 4 --theta_deg 10 --radius 2.0

    # Random targets in full 180° range (like the real eval)
    python rl/render_policy_video.py ... --stage 6 --random_targets
"""

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rl.env_config_registry import register_environments
import gymnasium as gym


def build_render_env(env_name: str, stage_idx: int,
                     reward_success: float = 20.0,
                     alpha_jerk: float = 0.0,
                     alpha_heading: float = 0.0,
                     alpha_backward: float = 0.0,
                     success_radius: float = 0.1,
                     lpf_cutoff_hz: float = 0.0,
) -> gym.Env:
    register_environments(max_episode_steps=2000)
    extra = {}
    if lpf_cutoff_hz > 0:
        extra["lpf_cutoff_hz"] = lpf_cutoff_hz
    env = gym.make(
        f"gymnasium_env/{env_name}",
        frame_skip=20,
        render_mode="rgb_array",
        motorVelRange=1,
        actionMultiplier=1,
        rewardFailure=-1.0,
        rewardSuccess=reward_success,
        alpha_jerk=alpha_jerk,
        alpha_heading=alpha_heading,
        alpha_backward=alpha_backward,
        success_radius=success_radius,
        **extra,
    )
    raw = env.unwrapped
    stages = raw.CURRICULUM_STAGES
    if stage_idx >= len(stages):
        raise ValueError(f"stage_idx={stage_idx} out of range (env has {len(stages)} stages)")
    stage = stages[stage_idx]
    sm = raw._sensor_model
    sm.enable_imu          = stage["enable_imu"]
    sm.enable_encoder      = stage["enable_encoder"]
    sm.enable_vision       = stage["enable_vision"]
    sm.enable_vision_noise = stage["enable_vision_noise"]
    sm.enable_latency      = stage["enable_latency"]
    if "latency_min" in stage:
        sm.latency_min = stage["latency_min"]
        sm.latency_max = stage["latency_max"]
    if "frame_drop_prob" in stage:
        sm.frame_drop_prob = stage["frame_drop_prob"]
    sm.enable_fps_dr     = stage.get("enable_fps_dr", False)
    raw._motor_dr_active     = stage.get("enable_motor",        False)
    raw._extra_frames_active = stage.get("enable_extra_frames", False)
    raw.r_min     = float(stage["r_min"])
    raw.r_max     = float(stage["r_max"])
    raw.theta_max = float(np.deg2rad(stage.get("theta_max_deg", 180.0)))
    print(f"Stage {stage_idx} ({stage['name']}):  "
          f"θ_max=±{stage.get('theta_max_deg',180):.0f}°  "
          f"r=[{stage['r_min']}, {stage['r_max']}]m  "
          f"motor_dr={raw._motor_dr_active}")
    return env


def make_fixed_target_sampler(theta_deg: float, radius: float):
    theta_rad = np.deg2rad(theta_deg)
    def _sample(initial_com, rot):
        body_vec = np.array([-radius * np.cos(theta_rad), radius * np.sin(theta_rad)])
        return initial_com + rot @ body_vec
    return _sample


def run_episode(env, model, seed: int, fixed_sampler=None):
    raw = env.unwrapped
    if fixed_sampler is not None:
        raw._sample_target = fixed_sampler

    obs, _ = env.reset(seed=seed)
    fish_start = raw.data.site("COM_0").xpos[:2].copy()
    target     = raw.target.copy()
    dist_to_target = float(np.linalg.norm(target - fish_start))
    angle_deg = float(np.degrees(
        np.arctan2(target[1] - fish_start[1], target[0] - fish_start[0])
    ))
    print(f"  seed={seed}  fish={fish_start.round(2)}  target={target.round(2)}"
          f"  dist={dist_to_target:.2f}m")

    frames, ep_reward = [], 0.0
    done = False
    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated

    dist_end = info.get("distance_to_target", float("nan"))
    success  = bool(terminated)
    print(f"    → {len(frames)} steps  reward={ep_reward:.1f}  "
          f"final_dist={dist_end:.3f}m  success={success}")
    return frames, success


def save_video(frames, path: str, fps: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import cv2
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
    except ImportError:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)
    print(f"Saved {len(frames)} frames → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path",      help="Path to .zip checkpoint")
    parser.add_argument("--env",           default="sofi_da_angular_prog_curriculum_v4")
    parser.add_argument("--stage",         type=int,   default=6,
                        help="Curriculum stage index (default: 6 = soft_dr)")
    parser.add_argument("--theta_deg",     type=float, default=170.0,
                        help="Body-frame target angle in degrees (0=fwd, 180=behind). "
                             "Ignored when --random_targets is set.")
    parser.add_argument("--radius",        type=float, default=1.5,
                        help="Target distance in meters (ignored with --random_targets)")
    parser.add_argument("--random_targets", action="store_true",
                        help="Use the stage's normal random target sampling instead of a fixed angle")
    parser.add_argument("--n_episodes",    type=int,   default=3)
    parser.add_argument("--fps",           type=int,   default=50)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--output",        default="outputs/render.mp4")
    parser.add_argument("--reward_success", type=float, default=20.0)
    parser.add_argument("--alpha_jerk",    type=float, default=0.0)
    parser.add_argument("--alpha_heading", type=float, default=0.0)
    parser.add_argument("--alpha_backward", type=float, default=0.0)
    parser.add_argument("--success_radius", type=float, default=0.1)
    parser.add_argument("--lpf_cutoff_hz",  type=float, default=0.0,
                        help="LPF cutoff (Hz). MUST match training run.")
    args = parser.parse_args()

    env = build_render_env(
        args.env, args.stage,
        reward_success=args.reward_success,
        alpha_jerk=args.alpha_jerk,
        alpha_heading=args.alpha_heading,
        alpha_backward=args.alpha_backward,
        success_radius=args.success_radius,
        lpf_cutoff_hz=args.lpf_cutoff_hz,
    )

    model = SAC.load(args.model_path, env=env, device="cpu")
    print(f"Loaded: {args.model_path}")

    sampler = None if args.random_targets else make_fixed_target_sampler(args.theta_deg, args.radius)
    if sampler:
        print(f"Fixed target: θ={args.theta_deg}°, r={args.radius}m")
    else:
        print("Random targets (stage default)")

    all_frames, successes = [], []
    for ep in range(args.n_episodes):
        print(f"\nEpisode {ep + 1}/{args.n_episodes}")
        frames, success = run_episode(env, model, seed=args.seed + ep, fixed_sampler=sampler)
        all_frames.extend(frames)
        successes.append(success)

    env.close()
    print(f"\nSuccess rate: {sum(successes)}/{args.n_episodes}")

    if all_frames:
        save_video(all_frames, args.output, args.fps)


if __name__ == "__main__":
    main()
