# Cached inputs

Two things here are caches: derived once from data too large to ship, then frozen so the
figures do not need it. Both can be regenerated with the tools named below.

| file | what | regenerate with |
|---|---|---|
| `sim_rollouts.npz` | commanded and measured motor traces, 25 episodes each for the deployed CPG policy and the unconstrained direct-action policy | `tools/probe_sim_rollouts.py`, using the checkpoints in `policies/` and the fishCREATE training environment |
| `gait_traces.npz` | commanded and actual motor angle, one 6 s episode each for TG, DA and DA-LPF1, as drawn in panel (b) of the 13 July gait figure | `tools/extract_gait_traces.py` on `10_gaits_combined.pdf`; the original rollouts are not in this repo |
| `policies/sofi_cpg_rl_policy.zip` | **the policy that was deployed on the fish**, SAC via Stable-Baselines3 | not derived; this is the trained artifact |
| `policies/sofi_da_rl_policy.zip` | the unconstrained direct-action policy it is compared against | as above |

The third cache lives with the data it belongs to:
`upstream/sofi_control/rl_deploy/analysis_20260520/cache_heading/initial_heading.json`,
the thirty AprilTag-derived initial headings, regenerated with `tools/make_heading_cache.py`.
It stands in for 1.5 GB of raw deployment video.
