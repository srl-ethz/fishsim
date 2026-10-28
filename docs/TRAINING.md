# Training the swimming policy

Everything runs from the repository root with the `fishsim` environment (`environment.yml`).

## The stack, in four layers

1. **Model.** `geometry/auto_SOFI_cpg.py` holds the identified physics in `SYSTEMPARAMETERS`: the E13a fluid
   coefficients (15 numbers, three ellipsoid groups: head, tail body, fin), the per-hinge stiffness and damping
   from the tail benchmark, the mass and inertia from CAD, and the motor as a position servo (kp 800, kv 20,
   force limit 1.0). `python geometry/auto_SOFI_cpg.py` writes `geometry/SOFI_cpg.xml`; the environments
   regenerate it on start, so the XML is derived, never edited.
2. **Environment.** `rl/environments/cpg/env_sofi.py` is the Gymnasium environment. The action is the six
   parameters of a central pattern generator (`cpg_oscillator.py`): two first-harmonic and two second-harmonic
   amplitudes, the frequency, and a filtered turn bias; the CPG runs at the simulation rate and the policy at
   50 Hz. The observation (24-D) is the target in the fish frame with simulated vision latency and drops,
   motor encoder and gyro with noise (`rl/models/sensor_model.py`), and the previous actions. On top of it
   sit `phase1_1` (sensor noise and delay), `phase1_2` (physics domain randomisation,
   `rl/models/physics_model.py`) and `phase2` (polar target curriculum), chained into
   `env_sofi_full_curriculum_cpg.py`, whose `CURRICULUM_STAGES` table is the curriculum.
   `rl/env_config_registry.py` registers the ids.
3. **Sweep.** `rl/sweepconfigs/sweep_cpg.yml` fixes the environment and the hyperparameter ranges of the
   deployed policy's sweep (W&B project `fishsim`).
4. **Trainer.** `rl/train_rl_curriculum.py`: SAC (Stable-Baselines3), seed 42, a mirrored replay buffer
   (left/right symmetry), promotion to the next stage at 90 % success over the last 100 episodes, learning
   rate scaled by `lr_scale` at each stage and reset at phase boundaries.

## Reproduce the deployed policy

```bash
# one run with the sweep's best hyperparameters
python rl/train_rl_curriculum.py --env sofi_full_curriculum_cpg --nSteps 3000000 --maxEpisodeSteps 800 \
    --hiddenDim 256 --numLayers 3 --learningRate 3e-4 --batchSize 512 --gamma 0.995 --tau 0.005 --seed 42

# or the full W&B sweep that produced it (entity and project in the yml)
wandb sweep rl/sweepconfigs/sweep_cpg.yml
wandb agent <sweep-id>
```

Checkpoints land in `outputs/<run>/SAC_curriculum_stage<N>_<name>.zip`, one per stage reached. The deployed
policy is the stage-8 "lateral" checkpoint of run `e06g760j` (`policies/README.md`). A full run takes about a
day on a workstation CPU; stage 8 is reached around 2 M steps.

## Evaluate

```bash
python rl/eval/eval_reachability.py policies/sofi_cpg_rl_policy.zip --env sofi_full_curriculum_cpg \
    --stage 8 --n_episodes 500 --stochastic
python rl/eval/render_policy_video.py policies/sofi_cpg_rl_policy.zip --env sofi_full_curriculum_cpg --stage 8
```

Evaluate with stochastic actions: SAC's mean action is near zero for targets behind the fish (turn left or
right are both right), and a deterministic evaluation underestimates the policy by two to three times.

## The direct-action comparison

`rl/environments/da/` is the same curriculum with the motor setpoint as the action (1-D, 40-D observation
with motor history), trained with `rl/sweepconfigs/sweep_da.yml` against `geometry/auto_SOFI_da.py` (refitB
fluid, identified motor). The resulting policy (`policies/sofi_da_rl_policy.zip`) swims in simulation by
exploiting high-frequency tail flicks that the stateless fluid model rewards (paper Fig. 9 and Fig. 10).
`sweep_da_fof.yml` is the same recipe with a first-order 4.5 Hz low-pass filter on the setpoint
(`--lpf_cutoff_hz 4.5`), the "DA-FOF" policy whose gait is the third trace of Fig. 10
(`policies/sofi_da_fof_rl_policy.zip`); `docs/sim2real_motor_fluid_lpf_findings.md` has the analysis. `docs/CURRICULUM_DESIGN.md` and
`docs/RL_QUICKREF.md` are the working notes behind the curriculum.
