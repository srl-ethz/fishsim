# Policies

Stable-Baselines3 SAC checkpoints (`SAC.load(path)`, SB3 2.6.0, torch 2.6.0), trained with
`rl/train_rl_curriculum.py` through the sweeps in `rl/sweepconfigs/`.

| File | md5 | Action space | Observation | Environment, W&B run, stage | Status |
|---|---|---|---|---|---|
| `sofi_cpg_rl_policy.zip` | `7bbe08b8e0999fb4324b073d324d8c78` | 6 CPG parameters per step: k1c, k1s, k2c, k2s, omega, theta_c | 24-D: relative target (cos, sin, distance) with vision latency, encoder, gyro, action history | `sofi_full_curriculum_cpg`, run `e06g760j`, curriculum stage 8 "lateral" (targets to ±120°), seed 42 | **Deployed.** All 43 hardware runs of 2026-05-20 (in fishdeploy) record its hash `7bbe08b8` |
| `sofi_da_rl_policy.zip` | `ba052debc0043ce7203699031578d318` | 1 motor setpoint per step (direct action, "DA") | 40-D with motor history | `sofi_full_curriculum_da`, run `9k7k8k3f`, stage 9 "full_range" | Simulation only. The policy exploits high-frequency tail flicks the stateless fluid model rewards; Fig. 9, Fig. 10 and Table III compare it with the TG policy |
| `sofi_da_fof_rl_policy.zip` | `c56d55259cd8a335f43e5bcfb65228de` | direct action behind a first-order 4.5 Hz setpoint filter ("DA-FOF") | 40-D with motor history | `sofi_full_curriculum_da` with `--lpf_cutoff_hz 4.5`, sweep `sweep_da_fof.yml`, run `o4whyrwa`, stage 8 | Simulation only. The filter removes the exploit; its gait is the third trace of Fig. 10 |

The paper calls the CPG policy the trajectory generator (TG). It was trained with the E13a fluid set and the
kp 800 motor model (`geometry/sofi.py`); the two direct-action policies with the refitB fluid set and
the identified motor (`geometry/sofi_motorid.py`).
Retraining: `docs/TRAINING.md`. On the fish both run with stochastic inference.
