"""Reachability evaluation for a trained DA policy.

Runs N episodes with targets uniformly sampled over the full polar space
(r in [r_min, r_max], theta in [-180°, 180°]).  All fish positions are
recorded in the fish body frame (origin = initial COM, x-axis = initial
forward direction).  Three plots are produced:

  1. Trajectory spaghetti  — up to max_traj trajectories, green=success, red=fail
  2. Target scatter         — where targets were placed, coloured by outcome
  3. Polar success heatmap  — (r, theta) bins coloured by fraction of successes

r_min/r_max/theta_max default to the curriculum stage config when not specified,
so the eval automatically matches training conditions.

Usage:
    python rl/eval_reachability.py <model.zip> [options]

Example:
    python rl/eval_reachability.py outputs/run_8thhv272/best_model.zip \\
        --n_episodes 500 --stage 5 \\
        --env sofi_da_angular_prog_curriculum \\
        --output outputs/reachability.png
"""

import argparse
import os
import sys

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rl.env_config_registry import register_environments
import gymnasium as gym


def apply_stage(raw, stage):
    """Apply a curriculum stage's sensor + physics-DR settings to a live env.

    Single source of truth shared by eval_reachability and eval_compare so the
    two evals configure DR/noise identically.
    """
    sm = raw._sensor_model
    sm.enable_imu          = stage["enable_imu"]
    sm.enable_encoder      = stage["enable_encoder"]
    sm.enable_vision       = stage["enable_vision"]
    sm.enable_vision_noise = stage["enable_vision_noise"]
    sm.enable_latency      = stage["enable_latency"]
    if "latency_min" in stage:
        sm.latency_min     = stage["latency_min"]
        sm.latency_max     = stage["latency_max"]
    if "frame_drop_prob" in stage:
        sm.frame_drop_prob = stage["frame_drop_prob"]
    sm.enable_fps_dr = stage.get("enable_fps_dr", False)
    raw._motor_dr_active     = stage.get("enable_motor",        False)
    raw._extra_frames_active = stage.get("enable_extra_frames", False)
    pm = getattr(raw, "_physics_model", None)
    if pm is not None and "enable_fluid" in stage:
        pm.enable_init_vel  = stage.get("enable_init_vel",  False)
        pm.enable_inertial  = stage.get("enable_inertial",  False)
        pm.enable_mechanics = stage.get("enable_mechanics", False)
        pm.enable_fluid     = stage.get("enable_fluid",     False)
        pm.fluid_std        = stage.get("fluid_std",        0.20)
        pm.enable_motor     = stage.get("enable_motor",     False)
        if "motor_kp_range"          in stage: pm.motor_kp_range         = tuple(stage["motor_kp_range"])
        if "motor_kv_range"          in stage: pm.motor_kv_range         = tuple(stage["motor_kv_range"])
        if "motor_forcerange_range"  in stage: pm.motor_forcerange_range = tuple(stage["motor_forcerange_range"])
        if "motor_zero_offset_std"   in stage: pm.motor_zero_offset_std  = float(stage["motor_zero_offset_std"])


def build_env(env_name: str, r_min, r_max, stage_idx: int = 0,
              reward_success: float = 1000.0,
              alpha_jerk: float = 0.0,
              alpha_heading: float = 0.0,
              success_radius: float = 0.1,
              lpf_cutoff_hz: float = 0.0,
              lpf_order: int = 1,
              gear_compliance_hz: float = 0.0,
              gear_compliance_order: int = 1,
              use_distance_reward: bool = False) -> gym.Env:
    register_environments(max_episode_steps=2000)
    extra_kwargs = {}
    if lpf_cutoff_hz > 0.0:
        extra_kwargs["lpf_cutoff_hz"] = lpf_cutoff_hz
    if lpf_order > 1:
        extra_kwargs["lpf_order"] = lpf_order
    if gear_compliance_hz > 0.0:
        extra_kwargs["gear_compliance_hz"] = gear_compliance_hz
    if gear_compliance_order > 1:
        extra_kwargs["gear_compliance_order"] = gear_compliance_order
    if use_distance_reward:
        extra_kwargs["use_distance_reward"] = True
    env = gym.make(
        f"gymnasium_env/{env_name}",
        frame_skip=20,
        render_mode=None,
        motorVelRange=1,
        actionMultiplier=1,
        rewardFailure=-1.0,
        rewardSuccess=reward_success,
        alpha_jerk=alpha_jerk,
        alpha_heading=alpha_heading,
        success_radius=success_radius,
        **extra_kwargs,
    )
    print(f"Env reward params: rewardSuccess={reward_success}, "
          f"alpha_jerk={alpha_jerk}, alpha_heading={alpha_heading}, "
          f"success_radius={success_radius}, lpf_cutoff_hz={lpf_cutoff_hz}, "
          f"gear_compliance_hz={gear_compliance_hz}, gear_compliance_order={gear_compliance_order}, "
          f"use_distance_reward={use_distance_reward}")
    raw = env.unwrapped

    # Apply the sensor + DR settings for the requested curriculum stage so the
    # eval observations match what the checkpoint was actually trained on.
    stages = raw.CURRICULUM_STAGES
    if stage_idx >= len(stages):
        raise ValueError(f"stage_idx={stage_idx} out of range (env has {len(stages)} stages)")
    stage = stages[stage_idx]
    apply_stage(raw, stage)
    sm = raw._sensor_model
    print(f"Stage {stage_idx} ({stage['name']}) sensors: "
          f"imu={sm.enable_imu}, enc={sm.enable_encoder}, "
          f"vis_noise={sm.enable_vision_noise}, latency={sm.enable_latency}, "
          f"fps_dr={sm.enable_fps_dr}")

    # r_min/r_max/theta_max: use stage config as default, CLI args override
    eval_r_min = float(r_min) if r_min is not None else float(stage["r_min"])
    eval_r_max = float(r_max) if r_max is not None else float(stage["r_max"])
    eval_theta = float(np.deg2rad(stage.get("theta_max_deg", 180.0)))
    src = "CLI override" if r_min is not None else "stage config"
    print(f"Eval range: r=[{eval_r_min}, {eval_r_max}] m, "
          f"θ=±{np.degrees(eval_theta):.0f}°  ({src})")

    raw.r_min     = eval_r_min
    raw.r_max     = eval_r_max
    raw.theta_max = eval_theta
    return env


def run_episodes(model, env, n_episodes: int, seed: int,
                 deterministic: bool = True, alpha_backward: float = 1.0):
    """
    Returns a list of dicts with keys:
        target_body   : (2,) target position in body frame
        traj_body     : (T, 2) fish COM trajectory in body frame
        is_backward   : (T,) bool — True at step i if the fish moved backward during step i
        success       : bool
        r_target      : float  polar radius of target
        theta_target  : float  polar angle of target (rad)
        n_steps       : int    steps taken in episode
        start_dist    : float  initial distance to target (m)
        end_dist      : float  final distance to target (m)
        reward_base   : float  total episode reward (no backward penalty)
        reward_new    : float  total reward with FIXED penalty (penalises backward)
        reward_old    : float  total reward with OLD/INVERTED penalty (penalises forward)
    """
    rng = np.random.default_rng(seed)
    results = []
    n_success = 0

    # Disable env-level locomotion terms — we compute both variants manually below.
    raw0 = env.unwrapped
    if hasattr(raw0, "alpha_backward"):
        raw0.alpha_backward = 0.0
    if hasattr(raw0, "alpha_fwd"):
        raw0.alpha_fwd = 0.0
    reward_success = getattr(raw0, "_reward_success", 20.0)

    bar = tqdm(
        total=n_episodes,
        desc="Evaluating",
        unit="ep",
        postfix=dict(success="0.0%"),
    )

    for ep in range(n_episodes):
        obs, _info = env.reset(seed=int(rng.integers(0, 2**31)))

        # Capture pose at episode start
        raw = env.unwrapped
        initial_com = raw.data.site("COM_0").xpos[:2].copy()
        # Fish nose points in -x_world at heading=0, so +π aligns +x_body with the nose.
        head_angle  = raw.data.joint("headRollZ_0").qpos[0] + np.pi
        rot = np.array([
            [ np.cos(head_angle), -np.sin(head_angle)],
            [ np.sin(head_angle),  np.cos(head_angle)],
        ])

        # Target in body frame (rot.T cancels the rot applied in _sample_target)
        target_world = raw.target.copy()
        target_body  = rot.T @ (target_world - initial_com)
        r_t     = float(np.linalg.norm(target_body))
        theta_t = float(np.arctan2(target_body[1], target_body[0]))

        traj = []
        is_backward = []
        ep_reward_base = 0.0
        ep_reward_new  = 0.0   # fixed penalty: penalise _last_fish_vx > 0 (backward)
        ep_reward_old  = 0.0   # old/inverted: penalise _last_fish_vx < 0 (forward)
        done = False
        success = False
        n_steps = 0
        ep_start_dist = r_t    # initial distance = magnitude of target in body frame
        ep_end_dist   = r_t
        last_step_info = {}

        unstable = False
        while not done:
            pos_world = raw.data.site("COM_0").xpos[:2].copy()
            if np.any(~np.isfinite(pos_world)):
                unstable = True
                break
            traj.append(rot.T @ (pos_world - initial_com))

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, base_r, terminated, truncated, last_step_info = env.step(action)
            done = terminated or truncated
            n_steps += 1

            # _last_fish_vx > 0 → moving toward tail (backward). Both CPG and DA
            # envs set this in _get_obs from the same vel_vec_target_local.
            vx = float(getattr(raw, "_last_fish_vx", 0.0))
            is_backward.append(vx > 0.0)

            ep_reward_base += base_r
            if n_steps > 10 and alpha_backward > 0.0:
                ep_reward_new += base_r - alpha_backward * max(0.0,  vx)   # fixed
                ep_reward_old += base_r - alpha_backward * max(0.0, -vx)   # inverted
            else:
                ep_reward_new += base_r
                ep_reward_old += base_r

            if last_step_info.get("is_success", terminated):
                success = True

        if unstable or len(traj) == 0:
            bar.update(1)
            continue  # discard episode with NaN physics

        ep_end_dist = float(last_step_info.get("distance_to_target", ep_start_dist))

        if success:
            n_success += 1

        results.append(dict(
            target_body   = target_body,
            traj_body     = np.array(traj),
            is_backward   = np.array(is_backward, dtype=bool),
            success       = success,
            r_target      = r_t,
            theta_target  = theta_t,
            n_steps       = n_steps,
            start_dist    = ep_start_dist,
            end_dist      = ep_end_dist,
            reward_success= reward_success,
            reward_base   = ep_reward_base,
            reward_new    = ep_reward_new,
            reward_old    = ep_reward_old,
        ))

        bar.set_postfix(success=f"{n_success / (ep + 1):.1%}")
        bar.update(1)

    bar.close()
    return results


def print_angle_bins(results):
    """Print success rate broken down by target direction (front / side / behind)."""
    bins = [
        ("Front  (|θ| ≤  60°)", lambda t: abs(t) <= np.radians(60)),
        ("Side   (60° < |θ| ≤ 120°)", lambda t: np.radians(60) < abs(t) <= np.radians(120)),
        ("Behind (|θ| > 120°)", lambda t: abs(t) > np.radians(120)),
    ]
    print("\nDirection breakdown:")
    for label, mask_fn in bins:
        subset = [r for r in results if mask_fn(r["theta_target"])]
        if not subset:
            print(f"  {label}: no samples")
            continue
        n_ok = sum(r["success"] for r in subset)
        steps_ok = [r["n_steps"] for r in subset if r["success"]]
        mean_steps = f"  avg {np.mean(steps_ok):.0f} steps to success" if steps_ok else ""
        total_steps = sum(r["n_steps"] for r in subset)
        bw_steps = sum(int(r.get("is_backward", np.array([])).sum()) for r in subset)
        bw_pct = f"  {bw_steps/total_steps:.1%} backward" if total_steps > 0 else ""
        print(f"  {label}: {n_ok:3d}/{len(subset):3d} = {n_ok/len(subset):.1%}{mean_steps}{bw_pct}")


def print_reward_comparison(results, alpha_backward: float):
    """Compare total episode reward under three penalty regimes."""
    def _stats(key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.median(vals), np.min(vals), np.max(vals)

    bw_fracs = [r["is_backward"].mean() for r in results]
    print(f"\nBackward-swimming: {np.mean(bw_fracs):.1%} of steps (mean across episodes)")

    print(f"\nEpisode reward comparison  (alpha_backward={alpha_backward}):")
    print(f"  {'Regime':<38s}  {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}")
    print(f"  {'-'*38}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}")
    for label, key in [
        ("No penalty (base env reward)",     "reward_base"),
        ("NEW penalty — penalise backward",  "reward_new"),
        ("OLD penalty — penalise forward",   "reward_old"),
    ]:
        mean, med, lo, hi = _stats(key)
        print(f"  {label:<38s}  {mean:>8.2f}  {med:>8.2f}  {lo:>8.2f}  {hi:>8.2f}")

    print()
    mean_new, *_ = _stats("reward_new")
    mean_old, *_ = _stats("reward_old")
    if mean_old > mean_new:
        print("  → OLD > NEW: model optimised the INVERTED penalty (exploited backward swimming)")
    elif mean_new > mean_old:
        print("  → NEW > OLD: model is penalised less under correct penalty (swims forward)")
    else:
        print("  → NEW ≈ OLD: backward penalty has little effect on this policy")


def print_episode_table(results, alpha_backward: float = 1.0):
    """Per-episode reward breakdown with theoretical comparison."""
    rs = results[0].get("reward_success", 20.0) if results else 20.0
    print(f"\nPer-episode breakdown  (rewardSuccess={rs}, alpha_backward={alpha_backward}):")
    print(f"  {'Ep':>3}  {'θ(°)':>7}  {'r(m)':>5}  {'OK':>3}  "
          f"{'Steps':>5}  {'BW%':>4}  "
          f"{'d_start':>7}  {'d_end':>6}  "
          f"{'R_base':>7}  {'theor':>7}  "
          f"{'bw_pen':>7}  {'R_new':>7}")
    print("  " + "-" * 90)
    for i, r in enumerate(results):
        theta_deg  = np.degrees(r["theta_target"])
        bw_pct     = 100.0 * r["is_backward"].mean()
        theor      = r["start_dist"] + rs   # straight-shot: progress=r + success bonus
        bw_penalty = r["reward_new"] - r["reward_base"]   # negative value
        print(f"  {i+1:>3}  {theta_deg:>+7.1f}  {r['r_target']:>5.2f}  "
              f"{'✓' if r['success'] else '✗':>3}  "
              f"{r['n_steps']:>5}  {bw_pct:>3.0f}%  "
              f"{r['start_dist']:>7.3f}  {r['end_dist']:>6.3f}  "
              f"{r['reward_base']:>+7.2f}  {theor:>+7.2f}  "
              f"{bw_penalty:>+7.2f}  {r['reward_new']:>+7.2f}")
    print()
    print(f"  Theoretical = straight-shot progress (r_start) + rewardSuccess={rs}")
    print(f"  bw_pen = reward with FIXED penalty − base reward (negative = backward steps penalised)")


def plot_results(results, output_path: str, max_traj: int = 300,
                 policy_label: str = "", stage_idx: int = None,
                 mode: str = ""):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(20, 7))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")

    # ── Panel 1: Trajectory spaghetti ──────────────────────────────────────
    n_show = min(max_traj, len(results))
    # Show balanced mix of success/failure
    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]
    n_s = min(n_show // 2, len(successes))
    n_f = min(n_show - n_s, len(failures))
    subset = (
        list(np.random.default_rng(0).choice(successes, n_s, replace=False))
        + list(np.random.default_rng(0).choice(failures, n_f, replace=False))
        if successes and failures
        else results[:n_show]
    )

    bw_segments_all = []
    for r in subset:
        color = "#22aa44" if r["success"] else "#cc2222"
        ax1.plot(r["traj_body"][:, 0], r["traj_body"][:, 1],
                 alpha=0.15, linewidth=0.6, color=color, rasterized=True)
        # Collect backward segments for orange overlay
        traj_arr = r["traj_body"]
        bw = r.get("is_backward", np.zeros(len(traj_arr), dtype=bool))
        for i in range(min(len(bw), len(traj_arr) - 1)):
            if bw[i]:
                bw_segments_all.append(traj_arr[i : i + 2])

    # Overlay backward-motion segments in orange (drawn last so they're visible)
    if bw_segments_all:
        lc = LineCollection(bw_segments_all, color="#ff8800", alpha=0.45,
                            linewidth=0.9, rasterized=True, zorder=2)
        ax1.add_collection(lc)

    for r in subset:
        color = "#22aa44" if r["success"] else "#cc2222"
        ax1.scatter(*r["target_body"], s=8, color=color, alpha=0.5, zorder=3)

    # Fish start indicator
    ax1.annotate("", xy=(0.4, 0.0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="steelblue", lw=2))
    ax1.scatter(0, 0, s=60, color="steelblue", zorder=5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x body (m)")
    ax1.set_ylabel("y body (m)")
    ax1.set_title(f"Trajectories (n={n_show})\ngreen=success  red=fail  orange=backward")
    ax1.axhline(0, color="k", lw=0.4, ls="--")
    ax1.axvline(0, color="k", lw=0.4, ls="--")

    # ── Panel 2: All-positions heatmap ────────────────────────────────────
    all_pos = np.vstack([r["traj_body"] for r in results])
    lim = max(np.abs(all_pos).max() * 1.05, 1.0)
    h = ax2.hist2d(
        all_pos[:, 0], all_pos[:, 1],
        bins=120, range=[[-lim, lim], [-lim, lim]],
        norm=mcolors.LogNorm(), cmap="hot",
    )
    plt.colorbar(h[3], ax=ax2, label="visit count (log scale)")
    ax2.scatter(0, 0, s=60, color="steelblue", zorder=5)
    ax2.annotate("", xy=(0.4, 0.0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="steelblue", lw=2))
    ax2.set_aspect("equal")
    ax2.set_xlabel("x body (m)")
    ax2.set_ylabel("y body (m)")
    ax2.set_title("Visited-position density")

    # ── Panel 3: Polar success-rate map ──────────────────────────────────
    thetas = np.array([r["theta_target"] for r in results])
    rs     = np.array([r["r_target"]     for r in results])
    wins   = np.array([float(r["success"]) for r in results])

    n_r_bins     = 12
    n_theta_bins = 24
    r_edges     = np.linspace(rs.min(), rs.max(), n_r_bins + 1)
    theta_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1)

    rate_grid = np.full((n_r_bins, n_theta_bins), np.nan)
    for i in range(n_r_bins):
        for j in range(n_theta_bins):
            mask = (
                (rs >= r_edges[i]) & (rs < r_edges[i + 1]) &
                (thetas >= theta_edges[j]) & (thetas < theta_edges[j + 1])
            )
            if mask.sum() >= 2:
                rate_grid[i, j] = wins[mask].mean()

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    r_centers     = 0.5 * (r_edges[:-1] + r_edges[1:])
    T, R = np.meshgrid(theta_centers, r_centers)

    pc = ax3.pcolormesh(
        theta_edges, r_edges, rate_grid,
        cmap="RdYlGn", vmin=0.0, vmax=1.0, shading="flat",
    )
    plt.colorbar(pc, ax=ax3, label="success rate", pad=0.1)
    ax3.set_title("Success rate by target (r, θ)")
    ax3.set_theta_zero_location("N")   # forward = top
    ax3.set_theta_direction(-1)         # clockwise = right turn

    # Summary stats
    n_total   = len(results)
    n_success = sum(r["success"] for r in results)
    # Identify which policy/stage this eval came from so the saved figure is
    # self-documenting (we always know which checkpoint produced it).
    id_bits = []
    if policy_label:
        id_bits.append(policy_label)
    if stage_idx is not None:
        id_bits.append(f"stage {stage_idx}")
    if mode:
        id_bits.append(mode)
    id_line = ("\n" + "   ·   ".join(id_bits)) if id_bits else ""
    fig.suptitle(
        f"Reachability evaluation  —  {n_success}/{n_total} episodes succeeded "
        f"({n_success/n_total:.1%}){id_line}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reachability evaluation for a trained DA policy."
    )
    parser.add_argument("model_path", help="Path to model .zip file")
    parser.add_argument("--env",        default="sofi_da_radial_curriculum",
                        help="Environment registry name")
    parser.add_argument("--n_episodes", type=int,   default=500)
    parser.add_argument("--r_min",      type=float, default=None,
                        help="Min target radius (m). Defaults to stage config value.")
    parser.add_argument("--r_max",      type=float, default=None,
                        help="Max target radius (m). Defaults to stage config value.")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--stochastic",  action="store_false", dest="deterministic",
                        help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--stage",      type=int,   default=0,
                        help="Curriculum stage index whose sensor settings to apply (default: 0)")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--max_traj",      type=int,   default=300,
                        help="Max trajectories drawn in spaghetti plot")
    parser.add_argument("--alpha_backward", type=float, default=1.0,
                        help="Backward-penalty weight for reward comparison table (default: 1.0)")
    parser.add_argument("--reward_success", type=float, default=1000.0,
                        help="rewardSuccess passed to env; MUST match training value (default: 1000)")
    parser.add_argument("--alpha_jerk",    type=float, default=0.0,
                        help="alpha_jerk passed to env (default: 0)")
    parser.add_argument("--alpha_heading",  type=float, default=0.0,
                        help="alpha_heading passed to env (default: 0); set to match training run")
    parser.add_argument("--success_radius", type=float, default=0.1,
                        help="Success distance threshold in meters (default: 0.1)")
    parser.add_argument("--lpf_cutoff_hz", type=float, default=0.0,
                        help="LPF cutoff frequency (Hz); 0 = no filter. MUST match training run.")
    parser.add_argument("--lpf_order", type=int, default=1,
                        help="Setpoint LPF order: 1=EMA, 2=Butterworth. MUST match training run.")
    parser.add_argument("--gear_compliance_hz", type=float, default=0.0,
                        help="Gear compliance bandwidth (Hz); 0 = disabled. MUST match training run. "
                             "SDA-S03+ uses 3.8 Hz (physics-derived).")
    parser.add_argument("--gear_compliance_order", type=int, default=1,
                        help="Gear compliance filter order (1=1st-order EMA, 2=cascaded). SDA-S04+ uses 2.")
    parser.add_argument("--use_distance_reward", action="store_true", default=False,
                        help="Use CPG-style distance reward (r=-dist+success). MUST match training.")
    parser.add_argument("--verbose",       action="store_true",
                        help="Print per-episode reward breakdown table")
    parser.add_argument("--label",         default=None,
                        help="Policy/run name to stamp on the figure + filename "
                             "(default: parent dir name of the model, e.g. the run id)")
    parser.add_argument("--output",         default=None,
                        help="Output PNG path. If omitted, auto-named as "
                             "outputs/reachability_<label>_stage<N>.png")
    args = parser.parse_args()

    # Policy label defaults to the run-id directory holding the checkpoint
    # (e.g. outputs/os4ok2sy/best_model_stage9.zip → "os4ok2sy").
    if args.label is None:
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.model_path)))
        args.label = parent or os.path.splitext(os.path.basename(args.model_path))[0]
    if args.output is None:
        safe = args.label.replace("/", "_")
        args.output = f"outputs/reachability_{safe}_stage{args.stage}.png"

    print(f"Loading model: {args.model_path}")
    env = build_env(args.env, args.r_min, args.r_max, stage_idx=args.stage,
                    reward_success=args.reward_success,
                    alpha_jerk=args.alpha_jerk,
                    alpha_heading=args.alpha_heading,
                    success_radius=args.success_radius,
                    lpf_cutoff_hz=args.lpf_cutoff_hz,
                    lpf_order=args.lpf_order,
                    gear_compliance_hz=args.gear_compliance_hz,
                    gear_compliance_order=args.gear_compliance_order,
                    use_distance_reward=args.use_distance_reward)
    model = SAC.load(args.model_path, env=env)

    mode = "stochastic" if not args.deterministic else "deterministic"
    print(f"Running {args.n_episodes} episodes ({mode})...")
    results = run_episodes(model, env, args.n_episodes, args.seed,
                           deterministic=args.deterministic,
                           alpha_backward=args.alpha_backward)

    n_ok = sum(r["success"] for r in results)
    n_total = len(results)
    print(f"\nOverall success: {n_ok}/{n_total} = {n_ok/n_total:.1%}")
    print_angle_bins(results)
    print_reward_comparison(results, alpha_backward=args.alpha_backward)
    if args.verbose:
        print_episode_table(results, alpha_backward=args.alpha_backward)
    print("\nPlotting...")
    plot_results(results, args.output, max_traj=args.max_traj,
                 policy_label=args.label, stage_idx=args.stage, mode=mode)


if __name__ == "__main__":
    main()
