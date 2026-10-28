#!/usr/bin/env python3
"""Roll out the CPG and direct-action policies in sim, recording commanded tail motion.

Reuses rl/eval/eval_compare.py's build_env(), which already applies the correct
ASYMMETRIC domain randomisation: the CPG model has no motor DR, the DA model does
(kp, kv, forcerange, dead-time). Getting that asymmetry wrong would invalidate the
comparison, so we do not reimplement it.

The deployed policy is policies/sofi_cpg_rl_policy.zip -- confirmed by model_hash
7bbe08b8 recorded in all 50 run_config.json files of the 2026-05-20 session. (The
file is called SAC_curriculum_stage0_fwd_wide.zip on the Pi; the hash is what
identifies it. SAC_curriculum_stage7_semi_lateral.zip is a DIFFERENT policy and was
never deployed.)

Writes per-step commanded motor angle + dt so the same spectral metric can be run
on sim and hardware alike.
"""
import os, sys, argparse
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # fishsim root
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "rl", "eval"))
os.chdir(REPO)

from stable_baselines3 import SAC
from eval_compare import build_env

OUT = "/home/sramc/Robotic_Fish/06_Publication/paper/Images/scripts/_sim_rollouts.npz"


def rollout(model_path, env_name, n_ep, dr, seed, pass_rw, lpf=0.0, radius=0.05):
    env = build_env(env_name, radius, dr, pass_reward_kwargs=pass_rw,
                    lpf_cutoff_hz=lpf)
    model = SAC.load(model_path, device="cpu")
    raw = env.unwrapped
    dt = float(raw.model.opt.timestep * raw.frame_skip)
    mq = raw.model.jnt_qposadr[
        __import__("mujoco").mj_name2id(raw.model,
                                        __import__("mujoco").mjtObj.mjOBJ_JOINT, "motor_0")]
    eps = []
    for i in range(n_ep):
        obs, _ = env.reset(seed=seed + i)
        cmd, meas = [], []
        done = trunc = False
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(a)
            cmd.append(float(raw.data.ctrl[0]))
            meas.append(float(raw.data.qpos[mq]))
        if len(cmd) > 40:
            eps.append((np.array(cmd), np.array(meas)))
    env.close()
    return eps, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dr", action="store_true", default=True)
    a = p.parse_args()

    out = {}
    for tag, mp, env, pr, lpf, rad in (
        ("cpg", "policies/sofi_cpg_rl_policy.zip", "sofi_full_curriculum_cpg", False, 0.0, 0.05),
        ("da",  "policies/sofi_da_rl_policy.zip",  "sofi_full_curriculum_da",  True,  0.0, 0.10),
    ):
        print(f"[{tag}] rolling out {a.n_episodes} episodes (dr={a.dr}) ...")
        eps, dt = rollout(os.path.join(REPO, mp), env, a.n_episodes, a.dr,
                          a.seed, pr, lpf, rad)
        print(f"  {len(eps)} usable episodes, dt={dt:.4f}s ({1/dt:.1f} Hz), "
              f"mean len {np.mean([len(c) for c, _ in eps]):.0f} steps")
        out[f"{tag}_dt"] = dt
        out[f"{tag}_cmd"] = np.array([c for c, _ in eps], dtype=object)
        out[f"{tag}_meas"] = np.array([m for _, m in eps], dtype=object)
    np.savez(OUT, **out, allow_pickle=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
