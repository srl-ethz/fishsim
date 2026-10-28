"""Side-by-side reachability comparison of the two canonical policies.

Runs the CPG policy and the DA policy over the full polar task (targets at
r in [r_min, r_max], theta in [-180, 180]) and plots their trajectory
spaghetti next to each other:

    LEFT  = CPG  (policies/sofi_cpg_rl_policy.zip, success radius 0.05 m)
    RIGHT = DA   (policies/sofi_da_rl_policy.zip,  success radius 0.10 m)

green = reached target, red = failed, orange = backward-swimming segment
(both CPG and DA set _last_fish_vx in _get_obs from the same vel_vec_target_local).

Physics mode (applies to BOTH panels):
    default            nominal / mean physics, clean sensors  (no DR)
    --dr               full training-stage domain randomization
                         CPG: fluid + inertial + mechanics DR, sensor noise
                              (no motor DR — the CPG model has none)
                         DA : the above PLUS motor DR (kp 5-7, kv 0-0.12,
                              forcerange 1.6-2.3) + dead-time

Usage:
    python rl/eval/eval_compare.py
    python rl/eval/eval_compare.py --dr --n_episodes 500
    python rl/eval/eval_compare.py --deterministic --output outputs/cmp.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from rl.env_config_registry import register_environments
from rl.eval.eval_reachability import run_episodes, apply_stage
import gymnasium as gym

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_env(env_name, success_radius, dr, pass_reward_kwargs=False,
              theta_max_deg=None, r_min=None, r_max=None, lpf_cutoff_hz=0.0,
              dr_stage=None):
    """Make the env over its full-task range; apply (or disable) domain randomization.

    dr=False → nominal/mean physics + clean sensors (vision only, no noise).
    dr=True  → the final curriculum stage's full DR + sensor settings.

    pass_reward_kwargs: DA envs accept success_radius/rewardSuccess; the CPG
    BaseFishEnv does not (it hardcodes success<0.05 + reward 1000 and forwards
    unknown kwargs to MujocoEnv), so they are omitted for CPG.
    """
    register_environments(max_episode_steps=2000)
    make_kwargs = dict(
        frame_skip=20, render_mode=None,
        motorVelRange=1, actionMultiplier=1, rewardFailure=-1.0,
    )
    if pass_reward_kwargs:
        make_kwargs.update(rewardSuccess=1000.0, success_radius=success_radius)
    if lpf_cutoff_hz > 0.0:                       # DA setpoint LPF (must match training)
        make_kwargs["lpf_cutoff_hz"] = float(lpf_cutoff_hz)
    env = gym.make(f"gymnasium_env/{env_name}", **make_kwargs)
    raw = env.unwrapped
    stages = raw.CURRICULUM_STAGES
    final = stages[-1]                       # full-task stage (theta=180)

    sm = raw._sensor_model
    pm = getattr(raw, "_physics_model", None)

    if dr:
        # Apply a curriculum stage's full DR + sensor settings (shared apply_stage).
        # dr_stage=None → the env's FINAL stage; else the given stage index (clamped),
        # so a policy can be evaluated at the DR level it actually trained to.
        stage_cfg = final if dr_stage is None else stages[max(0, min(dr_stage, len(stages) - 1))]
        apply_stage(raw, stage_cfg)
    else:
        # Nominal / mean physics, clean observations.
        sm.enable_imu = sm.enable_encoder = False
        sm.enable_vision = True
        sm.enable_vision_noise = sm.enable_latency = sm.enable_fps_dr = False
        sm.frame_drop_prob = 0.0
        raw._motor_dr_active = False
        if pm is not None:
            pm.enable_init_vel = pm.enable_inertial = pm.enable_mechanics = False
            pm.enable_fluid = pm.enable_motor = False

    # Eval range: CLI override (common range = fair comparison) else env final stage.
    raw.r_min     = float(r_min) if r_min is not None else float(final["r_min"])
    raw.r_max     = float(r_max) if r_max is not None else float(final["r_max"])
    raw.theta_max = float(np.deg2rad(theta_max_deg if theta_max_deg is not None
                                     else final.get("theta_max_deg", 180.0)))
    return env


def _plot_panel(ax, results, title, max_traj):
    """Trajectory spaghetti (mirrors eval_reachability panel 1)."""
    n_show = min(max_traj, len(results))
    successes = [r for r in results if r["success"]]
    failures  = [r for r in results if not r["success"]]
    n_s = min(n_show // 2, len(successes))
    n_f = min(n_show - n_s, len(failures))
    rng = np.random.default_rng(0)
    subset = (list(rng.choice(successes, n_s, replace=False)) +
              list(rng.choice(failures, n_f, replace=False))) if (successes and failures) else results[:n_show]

    bw_segments = []
    for r in subset:
        color = "#22aa44" if r["success"] else "#cc2222"
        ax.plot(r["traj_body"][:, 0], r["traj_body"][:, 1],
                alpha=0.15, linewidth=0.6, color=color, rasterized=True)
        traj = r["traj_body"]; bw = r.get("is_backward", np.zeros(len(traj), dtype=bool))
        for i in range(min(len(bw), len(traj) - 1)):
            if bw[i]:
                bw_segments.append(traj[i:i + 2])
    if bw_segments:
        ax.add_collection(LineCollection(bw_segments, color="#ff8800", alpha=0.45,
                                         linewidth=0.9, rasterized=True, zorder=2))
    for r in subset:
        ax.scatter(*r["target_body"], s=8,
                   color="#22aa44" if r["success"] else "#cc2222", alpha=0.5, zorder=3)

    ax.annotate("", xy=(0.4, 0.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=2))
    ax.scatter(0, 0, s=60, color="steelblue", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x body (m)"); ax.set_ylabel("y body (m)")
    ax.axhline(0, color="k", lw=0.4, ls="--"); ax.axvline(0, color="k", lw=0.4, ls="--")
    ax.set_title(title)


def main():
    p = argparse.ArgumentParser(description="Side-by-side CPG vs DA reachability comparison.")
    p.add_argument("--cpg_model", default=os.path.join(_REPO, "policies/sofi_cpg_rl_policy.zip"))
    p.add_argument("--da_model",  default=os.path.join(_REPO, "policies/sofi_da_rl_policy.zip"))
    p.add_argument("--cpg_env",   default="sofi_full_curriculum_cpg")
    p.add_argument("--da_env",    default="sofi_full_curriculum_da")
    p.add_argument("--cpg_success_radius", type=float, default=0.05,
                   help="CPG success radius (note: the CPG env hardcodes 0.05).")
    p.add_argument("--da_success_radius",  type=float, default=0.10)
    p.add_argument("--n_episodes", type=int, default=300)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--dr", action="store_true",
                   help="Apply full training-stage domain randomization (default: nominal/mean physics).")
    p.add_argument("--deterministic", action="store_true",
                   help="Deterministic policy (default: stochastic — correct for ±180° navigation).")
    p.add_argument("--max_traj", type=int, default=300)
    # Common eval range applied to BOTH panels (fair comparison). Default None =
    # each env's own final-stage range. NOTE: the deployed CPG policy is the
    # stage-8 'lateral' checkpoint (±120°, r≤2.5); pass --theta_max_deg 120
    # --r_max 2.5 to compare both within the CPG's actual training envelope.
    p.add_argument("--theta_max_deg", type=float, default=None)
    p.add_argument("--r_min", type=float, default=None)
    p.add_argument("--r_max", type=float, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--lpf_cutoff_hz", type=float, default=0.0,
                   help="Setpoint LPF cutoff for the DA env (Hz). MUST match the DA "
                        "policy's training value (e.g. 4.5 for the LPF sweeps); 0 = off.")
    p.add_argument("--dr_stage", type=int, default=None,
                   help="With --dr, evaluate at this curriculum stage index instead of the "
                        "final stage — applied to BOTH envs (e.g. 6 = the highest DR stage "
                        "both LPF DA policies reached → fair robustness comparison).")
    args = p.parse_args()

    mode = "DR (training randomization)" if args.dr else "nominal physics (no DR)"
    inf  = "deterministic" if args.deterministic else "stochastic"
    if args.output is None:
        args.output = f"outputs/compare_cpg_da_{'dr' if args.dr else 'nominal'}.png"

    panels = []
    for tag, model_path, env_name, sr in [
        ("CPG", args.cpg_model, args.cpg_env, args.cpg_success_radius),
        ("DA",  args.da_model,  args.da_env,  args.da_success_radius),
    ]:
        print(f"\n=== {tag}: {os.path.basename(model_path)}  ({mode}, {inf}) ===")
        env = build_env(env_name, sr, args.dr, pass_reward_kwargs=(tag == "DA"),
                        theta_max_deg=args.theta_max_deg, r_min=args.r_min, r_max=args.r_max,
                        lpf_cutoff_hz=(args.lpf_cutoff_hz if tag == "DA" else 0.0),
                        dr_stage=args.dr_stage)
        model = SAC.load(model_path, env=env)
        results = run_episodes(model, env, args.n_episodes, args.seed,
                               deterministic=args.deterministic)
        n_ok = sum(r["success"] for r in results)
        print(f"{tag} success: {n_ok}/{len(results)} = {n_ok/len(results):.1%}")
        panels.append((tag, sr, results))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, (tag, sr, results) in zip(axes, panels):
        n_ok = sum(r["success"] for r in results); n = len(results)
        _plot_panel(ax, results, f"{tag}  —  {n_ok}/{n} reached ({n_ok/n:.1%})\nsuccess radius {sr:.2f} m",
                    args.max_traj)
    fig.suptitle(f"CPG vs DA reachability  —  {mode}, {inf}  "
                 f"(n={args.n_episodes}/policy)\ngreen=success  red=fail  orange=backward",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
