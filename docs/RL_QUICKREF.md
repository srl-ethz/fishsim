# RL Quick Reference

> **Status (2026-06-02):** **NEW REGIME — SDA (Smooth Direct Action).** SDA-S01 READY (`sweep_sda_s01.yml`). DA sweeps (S01–S28) archived in `SWEEP_HISTORY.md`.
>
> **SDA regime:** Low-pass filter on motor setpoints (4–8 Hz, swept HP) + CPG-style reward (`r = −dist + 1000·success`). LPF prevents backward swimming physically; no heading/jerk/backward reward terms. Motor history stores filtered setpoints `ã_t`, not raw actions. Env: `sofi_sda_curriculum_v1`.
> **SDA-S01:** `8aaqutuu` — launch with `bash batching/sweep_launch.sh --sweepId srl_ethz/fishsim/8aaqutuu`
>
> **S28 (DA):** registered as `h86h9ud4`, not launched. All 3 DA bugs fixed (heading sign, alpha_fwd wire, rewardSuccess 20→100). Superseded by SDA-S01 which addresses the fundamental backward-exploit issue at the physics level.

Single file for Claude (and humans) to load for full context without reading every script.
Covers: sweep workflow, current env, locked HPs, sweep history highlights, known failure modes.

---

## 1. How to launch sweeps

### Register a new sweep (local machine or Euler)
```bash
cd ~/Robotic_Fish/fishCREATE   # or ~/sofiRL on Euler
wandb sweep rl/sweepconfigs/sweep_<name>.yml
# → prints sweep ID, e.g. srl_ethz/fishsim/zoi2qkk2
```
Always add the real sweep ID to `SWEEP_HISTORY.md` immediately after registration.

### Launch — cancel ALL existing jobs first (normal workflow)
From your local machine (SSHes to Euler + lab machines automatically):
```bash
bash batching/sweep_launch.sh --sweepId srl_ethz/fishsim/<ID>
```
**Warning:** this runs `scancel -u $(whoami)` on Euler, killing everything. Do not use when preserving other runs.

### Launch — keep existing jobs (e.g. S19 survivors while starting S20)
Run sbatch directly on Euler (already logged in):
```bash
cd ~/sofiRL
sbatch --export=ALL,SWEEP_ID='srl_ethz/fishsim/<ID>',WANDB_TAGS='euler' \
       batching/run_euler_hyperrl_cpu_array.sh
```
Or use the `--no-cancel` flag (see §1a below).

### Euler concurrency
- Array: `0-128%12` — submits 129 tasks, 12 run simultaneously.
- WandB `run_cap` in the YAML stops dispatching after N runs (even if more tasks pending).
- The launcher script is `rl/opt_hyperRL.py` (default) or `rl/opt_hyperRL_finetune.py` (finetuning sweeps via `--launcher`).

### 1a. `--no-cancel` flag (added to sweep_launch.sh)
Skips the `scancel` step, letting existing jobs continue:
```bash
bash batching/sweep_launch.sh --sweepId srl_ethz/fishsim/<ID> --no-cancel
```

### Identifying which SLURM task = which WandB run
When 4 jobs start at the same time on the same node, find their PIDs from WandB debug logs,
then map PID → SLURM task via the compute node's `/proc/<pid>/environ`:
```bash
# Step 1: get PID from wandb debug log
for rid in <run_id_1> <run_id_2>; do
  logdir=$(ls -d ~/sofiRL/wandb/run-*-${rid}/logs 2>/dev/null)
  head -3 $logdir/debug.log
done
# → shows "Configure stats pid to <PID>"

# Step 2: match PID to SLURM task on the compute node (e.g. eu-a2p-076)
ssh eu-a2p-076 "for pid in <pid1> <pid2>; do \
  echo \"PID \$pid: \$(tr '\0' '\n' < /proc/\$pid/environ | grep SLURM_ARRAY_TASK_ID)\"; done"
```

---

## 2. Current environment: `sofi_da_angular_prog_curriculum`

**File:** `rl/environments/env_sofi_da_angular_prog_curriculum.py`

### Observation layout (40 dims, always 40 throughout)
```
[0:3]   encoder cos/sin/vel        — always active
[3:4]   IMU wz                     — always active
[4:12]  frame slots 0–1 (oldest 2) — ZEROED until stage 10 (fluid_dr_10pct_frs)
[12:24] frame slots 2–4 (newest 3) — always active
[24:40] motor history interleaved [setpoint_i, actual_i]:
          setpoints at [24,26,28,30,32,34,36,38]  — always active
          actuals   at [25,27,29,31,33,35,37,39]  — ZEROED until stage 7 (motor_dr_soft)
```

### Active dims per stage group
| Stages | Active | Notes |
|--------|--------|-------|
| 0–6    | 24     | motor actuals zeroed, extra frames zeroed |
| 7–9    | 32     | motor actuals unlocked (stage 7), frames still zeroed |
| 10–11  | 40     | extra frame slots unlocked (stage 10) |

### 12-stage curriculum (S20)
| # | Name | θ | Sensors | DR | Obs | LR reset |
|---|------|---|---------|-----|-----|----------|
| 0 | fwd60 | ±60° | clean | none | 24 | — |
| 1 | semi90 | ±90° | clean | none | 24 | — |
| 2 | lateral120 | ±120° | clean | none | 24 | ✓ |
| 3 | wide150 | ±150° | clean | none | 24 | — |
| 4 | full180 | ±180° | clean | none | 24 | ✓ |
| 5 | full180_noise | ±180° | full+fps_dr | none | 24 | ✓ |
| 6 | soft_dr | ±180° | full+fps_dr | init+inertia+mech | 24 | ✓ |
| 7 | motor_dr_soft | ±180° | full+fps_dr | +motor Kp[3,7] | 32 | ✓ (actuals unlock) |
| 8 | motor_dr_full | ±180° | full+fps_dr | +motor Kp[1,8] | 32 | ✓ |
| 9 | fluid_dr_10pct | ±180° | full+fps_dr | +fluid 10% | 32 | ✓ |
| 10 | fluid_dr_10pct_frs | ±180° | full+fps_dr | +fluid 10% | 40 | ✓ (frames unlock) |
| 11 | fluid_dr_20pct | ±180° | full+fps_dr | +fluid 20% | 40 | ✓ |

`r_min=0.50 m, r_max=3.00 m` throughout. All stages ≥3 have `success_window=150`.

### Zero-init at obs unlock (in `train_rl_curriculum.py`)
- Stage 7 entry: `W[:, 25::2] = 0` (motor actuals columns)
- Stage 10 entry: `W[:, 4:12] = 0` (extra frame columns)
Eliminates Kaiming-init Q-perturbation when new dims activate.

### Motor DR range override (in `train_rl_curriculum.py`)
Stage dicts with `motor_kp_range` / `motor_kv_range` etc. override `physics_model` attributes.
Stages 9–11 have no override key → retain Kp=[1,8] from stage 8 automatically.

---

## 3. Locked hyperparameters (never need to sweep again)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| `batchSize` | 256 | S6: 512/1024 always fail |
| `numLayers` | 3 | Consistent across all sweeps |
| `train_freq` | 4 | Consistent |
| `lr_scale` | 0.60 | S9 validated |
| `mirror_replay_buffer` | true | S9 validated; bimodal left/right |
| `alpha_dist` | 0.0 | Dead parameter in every sweep |
| `maxEpisodeSteps` | 2000 | S17+: needed for kp=5 at r=3m, θ=180° |
| `motor_ctrlrange` | 1.2 | S17+: kp=5 needs wider swing |
| `camera_fps` | 53.18 | Hardware measured module-05, May 2026 |
| `rewardSuccess` | 30.0 | S19: Q_ratio ≈ 16×, within SAC safe zone |
| `nSteps` | 6 000 000 | S20: 12 stages × ~500k each |
| `eval_freq` | 50 000 | Check interval |
| `eval_freq_early` | 200 000 | Stages 0–3 (saves wall-clock) |
| `n_eval_episodes` | 100 | |

## 4. Active sweep space (S20 priors)

| Parameter | Distribution | Range | Evidence |
|-----------|-------------|-------|----------|
| `learningRate` | log_uniform | [1.2e-4, 1.5e-4] | S16c floor; S9 ceiling |
| `hiddenDim` | int_uniform | [200, 350] | S9 validated |
| `gamma` | uniform | [0.994, 0.997] | S6 constraint (γ>0.994) was a rewardSuccess=1000 artefact; with RS=20, need γ≥0.994 for 600-step U-turn signals |
| `tau` | uniform | [0.003, 0.007] | S9 validated |
| `learningStarts` | int_uniform | [3700, 5000] | S19+S24: all successful runs LS≥3695; fn2rzt89 (LS=3098) failed despite correct gamma |
| `alpha_jerk` | uniform | [0.0, 2e-4] | kp=5 needs overshoot; 0 must be reachable |
| `alpha_heading` | log_uniform | [0.002, 0.008] | S25+: now ABSOLUTE heading penalty (reward -= α·EMA(|θ_err|)), not delta; scale is ~20× smaller than old delta form |

---

## 5. Sweep history — key findings only

Full detail: `rl/sweepconfigs/SWEEP_HISTORY.md`. Column `L#` = line number of that sweep's header.

**S1–S8 (L9–L584): curriculum search, locked HPs established.**
All lessons are already baked into §§3–4 above. Read only to re-derive a specific locked HP.
Landmark: S8 (L523) confirmed ±60° still trains → angular expansion is the right path.

| Sweep | WandB ID | Problem → root cause → fix | L# |
|-------|----------|-----------------------------|-----|
| **S9** | a1wzz2bk | **Angular expansion validated (240k/stage clean). Hit 85% DR ceiling.** Root cause: motor DR silently OFF (`enable_motor=False` default); buffer 100% clean at DR entry. Bimodal policy — **always eval stochastic.** Fix: enable motor DR + progressive obs unlock. | L586 |
| S10 | aoxhiani | Noisy stage 0 → 50% entropy-collapse ceiling. Root cause: full sensor noise prevents bootstrap. Fix: clean stage 0. | L752 |
| S11 | 27v0bf4s | 40-dim bootstrap → 60% ceiling. Root cause: Kaiming init under-scales Q (only 24/40 active); motor actuals carry zero per-episode variance when DR is off. Fix: zero actuals at non-DR stages. | L853 |
| S12 | 0xzfd31d | Still 60% ceiling. Root cause: extra frame dims kept active → 32 effective dims, same saturation. Fix: zero extra frames too at bootstrap. | L1022 |
| S13 | qazhno0k | 24 effective dims restored but 2× slower bootstrap. Root cause: Kaiming scales for 40 dims but 24 non-zero → 0.77× pre-activation signal. Fix: rescale active column weights at init. | L1092 |
| S14 | 3isjoob3 | **FAILED.** Two bugs: mirror buffer negated wrong dims for 40-dim obs; init rescaling amplified corrupted gradients. Fix: both patched for S15. | L1251 |
| **S15** | 65o9fuu2 | Bugs fixed. **First run reached fluid DR.** SR 1.0→0.24 dip, then recovered to 0.71. Root cause: buffer contamination (dominant) + Kaiming unlock + premature promotion — three shocks at once. **kp=800 discovered here** (all S9–S15 used near-perfect actuators). Fix: kp=5, ADR split, zero-init at unlock, longer success window. | L1298 |
| S16c | ochrc82j | All runs stuck at stages 1–3. Root cause: kp=5 + mES=1500 only 14% margin → timeout ceiling 72–75%; alpha_jerk>0 penalises the overshoot kp=5 needs; rewardSuccess=1000 → Q-ratio≈670×. Fix: mES=2000, alpha_jerk∈[0,2e-4], rewardSuccess=200. | L1449 |
| S17 | qjbekg9k | Reached stage 7 at 7–24% SR. Root cause: rewardSuccess=200 → Q-ratio≈160× → entropy collapse after each promotion. Fix: rewardSuccess=30 (ratio≈16×, SAC safe zone). | L1566 |
| **S19** | swjyhc14 | **rewardSuccess=30 confirmed. 3 runs at stage 7 (29–43% SR, not collapse).** Stage 7 plateau: buffer 100% soft_dr at entry + 15× Kp range in one step. HP signals locked: AJ<5e-5, AH=0.10–0.14, LS≥3695. Fix: ADR Kp[3,7]→[1,8]. | L1661 |
| **S20** | zoi2qkk2 | **Cancelled.** Motor ADR split: stage 7 Kp=[3,7], stage 8 Kp=[1,8]. 7/10 runs reached stage 7 but all stalled (simultaneous obs-unlock + buffer shock). | L1738 |
| **S21** | w4piml6x | **Complete.** Tiny-DR-always-on: Kp=[4,6]+5% from stage 0, no unlock shocks. Top: 1× stage 9, 2× stage 8. Stage 9 ceiling = fluid SysID limit (5 frames insufficient for per-episode drag ID). AJ<8e-05 is the dominant HP predictor. | L1786 |
| **S22** | v0s9u9f5 | **Complete. 1 GRADUATED (vl2dx3ze, SR=91%, 2.63M steps).** Real DR from stage 0 proved. Stage-3 bottleneck: AH>0.15 correlated with ±150° stalls. min_stage_steps working as designed. | L1856 |
| **S23** | p2y6jl8m | **INVALID — backward penalty inverted + never passed to env.** All runs tainted (trained with alpha_backward=0). | L2012 |
| **S24** | e4qzkbj1 | **RUNNING (2026-06-01).** S23 bugs fixed. BUT: gamma range [0.993,0.997] too low (Q-calc confirms U-turns Q-negative at γ<0.996); backward α=0.1 linear lets close-target sprints profit; heading still delta. 1 run reached stage 6 (gamma=0.993, LS=4860, hDim=314). | L2081 |
| **S25** | `4ei7n1vp` | **Registered 2026-06-01.** Absolute heading penalty, quadratic backward α=3.0, gamma [0.996,0.997], α_jerk [8e-5,3e-4], LS [3700,5000], hDim [280,350]. Full Q-landscape verified numerically. | L2120 |
| **S26** | `76dmychx` | **Running 2026-06-01.** Episode termination for backward swimming (limit=0.03 m/s, grace=50 steps). init_vel moved to stage 5; init_vel_trans_std=0.10 m/s. Q-ratio 7×. All other HPs identical to S25. | L2160 |

### Known failure modes (hard-won)
- **Noisy stage 0**: entropy collapse at 50–55% (forward-only saturation). Always start with clean.
- **40-dim bootstrap**: if actuals active during non-DR stages, Kaiming noise kills learning. Zero until informative.
- **alpha_jerk > 2e-4**: kp=5 motor needs overshoot for thrust. Penalising jerk kills propulsion.
- **high alpha_heading at DR stages**: over-constrains navigation, conflicts with motor adaptation. Use 0.10–0.14 at DR.
- **rewardSuccess > 30**: Q_ratio >16× → SAC entropy collapse after stage promotions.
- **MirroredReplayBuffer**: always eval `deterministic=False`. Bimodal policy; deterministic mean ≈ no turn → fails lateral targets.
- **Buffer contamination at DR entry**: 2M buffer full of pre-DR experience → 1M flush steps. Split with ADR.
- **Kaiming perturbation at obs unlock**: zero-init new columns before activating.
- **maxEpisodeSteps=1500 with kp=5**: only 14% margin at ±150°; episodes timeout → 72–75% ceiling.

---

## 6. Key file locations

| What | Where |
|------|-------|
| Current env | `rl/environments/env_sofi_da_angular_prog_curriculum.py` |
| Training script | `rl/train_rl_curriculum.py` |
| Sweep launcher | `batching/sweep_launch.sh` (local) |
| SLURM script | `batching/run_euler_hyperrl_cpu_array.sh` |
| Machine config | `batching/computers.sh` (Euler concurrency=12) |
| Sweep YAMLs | `rl/sweepconfigs/sweep_s*.yml` |
| Full sweep history | `rl/sweepconfigs/SWEEP_HISTORY.md` |
| Mirror buffer | `rl/mirrored_replay_buffer.py` |
| Motor SysID | `../sofi_control/sim2real/results/06_motor_calibration.json` |
| Eval script | `rl/eval/eval_reachability.py` |
| Euler output dir | `/cluster/scratch/simonra/sofiRL/outputs/<run_id>/` |
| WandB project | `srl_ethz/fishsim` |

---

## 7. Physics model quick facts

- **Nominal motor**: `kp=5.0 Nm/rad`, `kv=0.30`, `ctrlrange=1.2 rad`
- **Motor DR (S20 stage 7)**: Kp ∈ [3.0, 7.0], stage 8: [1.0, 8.0]
- **SysID range** (hardware, module-06): Kp=1.0 @ 4Hz, 8.0 @ 1Hz
- **Fluid DR**: ±10% at stages 9–10, ±20% at stage 11
- **Camera**: 53.18 Hz nominal; fps_dr draws from [39.4, 66.9] Hz at stages 5+
- **Frame skip**: 20 sim steps per env step (≈20 ms/step)
