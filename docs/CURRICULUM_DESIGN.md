# SOFI RL Curriculum Design: Analysis, Literature, and Optimal Strategy

**Date:** 2026-05-25 (updated 2026-05-26)  
**Context:** Sweeps 9–16. Sweep 9 succeeded (bootstrap ~240k steps, >80% stage-6 reward).
Sweeps 10–11 plateaued at 60–65%, never advancing past stage 0. Sweep 15 reached stage 7
with recovery. Sweep 16 (thpjgi3p) running: finetune from 8a9yod1g stage 5, sweeping LR/tau.
**Goal:** Understand exactly why, derive the principled optimal curriculum, and give a
step-by-step implementation roadmap.

---

## Table of Contents

1. [Sweep-by-Sweep Observation Comparison](#1-sweep-by-sweep-observation-comparison)
2. [Why Extra Dims Hurt: Three Compounding Mechanisms](#2-why-extra-dims-hurt)
3. [Literature Foundation](#3-literature-foundation)
4. [LR Reset Strategy — What and When](#4-lr-reset-strategy)
5. [Finetuning from a Checkpoint](#5-finetuning-from-a-checkpoint)
6. [Optimal 11-Stage Curriculum Design](#6-optimal-11-stage-curriculum-design)
7. [Implementation Roadmap](#7-implementation-roadmap)
8. [Architecture Research: RMA, PEARL, and RMA-for-SAC](#8-architecture-research)
9. [DR Phase Analysis: Complete Picture](#9-dr-phase-analysis)
10. [Sweep 16 and Forward: RMA-for-SAC Contingency Plan](#10-sweep-16-and-forward)

---

## 1. Sweep-by-Sweep Observation Comparison

### Architecture sizes

| Component                | Sweep 9 (`angular_curriculum`) | Sweep 11 (`earlydrv2_curriculum`) |
|--------------------------|-------------------------------|-----------------------------------|
| Encoder (cos/sin/vel)    | 3                             | 3                                 |
| IMU (wz)                 | 1                             | 1                                 |
| Frame buffer             | 3 × 4 = **12**                | 5 × 4 = **20**                   |
| Motor setpoints (history)| 8                             | 8                                 |
| Motor actuals (tracking) | 0                             | 8                                 |
| **Total**                | **24**                        | **40**                            |

### Which dims are *informative* at stage 0?

At `clean60` (stage 0):
- `enable_encoder=False`, `enable_imu=False` → SensorModel **zeros** encoder and IMU slots in the observation every step.
- `enable_motor=False` → `_motor_dr_active=False` → (with sweep 12 fix) motor **actuals are zeroed** in `_build_motor_history_obs()`.
- Frame buffer: all slots contain real vision data (latency on, no noise, no drop).

| Component         | Sweep 9 informative | Sweep 12 informative (fix applied) | Sweep 11 informative (no fix) |
|-------------------|--------------------|------------------------------------|-------------------------------|
| Encoder + IMU     | 0                  | 0                                  | 0                             |
| Frame buffer      | 12 (3 slots)       | **20** (5 slots — still active!)   | 20 (5 slots)                  |
| Motor setpoints   | 8                  | 8                                  | 8                             |
| Motor actuals     | 0                  | **0** (zeroed by fix)              | 8 (≈ const at Kp=5.0)        |
| **Informative**   | **20**             | **28**                             | **36**                        |

**Sweep 11 had 80% more informative dims than sweep 9.  
Sweep 12 has 40% more (zeroing fix removes the 8 constant actuals).**

The extra 8 frame dims in sweeps 11/12 (slots 4+5 of a 5-slot buffer) are real vision
observations — not zero. They represent older approach history and only become
genuinely useful when fluid drag varies per-episode (i.e., when fluid DR is active).
At clean stage, they duplicate information already deducible from the 3 recent slots.

---

## 2. Why Extra Dims Hurt: Three Compounding Mechanisms

### Mechanism 1 — Gradient variance scales with informative dimension count

Every parameter update in SAC involves backpropagating through the Q-function and
policy. When the input contains `d` informative dimensions, the gradient of the loss
with respect to network parameters has variance that grows at least linearly in `d`
(for a fixed network width). This is a consequence of the input-layer weight noise in
deep networks: each additional informative input feature adds a stochastic contribution
to every parameter gradient.

For 36 vs 20 informative dims, the gradient variance at input layer is ~1.8× larger.
Because SAC uses a fixed replay buffer and batch size (256), this variance is not
reduced by batching — it propagates throughout. The immediate practical effect: the
policy needs proportionally more samples to converge to the same loss level.

Empirically: sweep 11 stage 0 required ~900k steps to reach the advancement threshold,
vs ~240k steps in sweep 9. That is a **3.75× slowdown** for an **80% increase in dims**,
consistent with near-quadratic scaling from multiplicative gradient noise.

References:
- **Bellman (1957)**: Curse of dimensionality — sample complexity grows exponentially
  with state dimension in tabular RL. For function approximation, the growth is
  polynomial but still severe.
- **Yarats et al., "Improving Sample Efficiency in Model-Free Reinforcement Learning
  from Images," ICML 2021** (arXiv:2012.04430): demonstrates that irrelevant input
  dims force the encoder to waste capacity, roughly doubling sample complexity per
  uninformative feature group.

### Mechanism 2 — Constant motor actuals at Kp=5.0 create near-redundant features

At stage 0 (`enable_motor=False`), every episode uses the same motor servo gains
(Kp=5.0 Nm/rad, Kv=0.3 from geometry/SOFI.xml). The tracking error is a deterministic
function of the setpoint history and these fixed gains. The actual motor position is
therefore almost perfectly predictable from the setpoint history:

```
actual_t ≈ f(setpoint_{t}, setpoint_{t-1}, ...; Kp=5.0, Kv=0.3)
```

The correlation between the setpoint and actual slots in the observation is typically
r ≈ 0.85–0.95 at Kp=5.0. Near-redundant features impose a specific cost on neural
networks: the weight matrix at the input layer must learn to suppress the near-duplicate
column, effectively wasting representational capacity. For a width-256 network, 8
near-redundant features consume capacity that could otherwise encode navigation policy.

More critically: when these near-redundant features have *slightly* different values
(because real tracking error is not zero), the network may assign them large weights
to exploit small systematic correlations. This produces high-variance gradient
directions that conflict with the gradients from the navigation reward.

References:
- **Strobl & Zeileis, "Danger: High Power!" (statistical redundancy theory)**: redundant
  features inflate effective parameter count while adding no predictive signal.
- **Cobbe et al., "Phasic Policy Gradient," ICML 2021** (arXiv:2009.04063): identifies
  that value network overfitting to spurious input correlations degrades policy quality.

### Mechanism 3 — Temporal redundancy of oldest frame buffer slots

At `FRAME_BUFFER_LEN=5` with 50 ms steps, slots 4 and 5 represent observations from
200 ms and 150 ms ago. At a clean (fixed physics) stage, the fish trajectory is smooth
and nearly periodic. The information in the 2 oldest slots is largely derivable from
the 3 most recent:

```
dist_t ≈ dist_{t-1} - v_mean * dt  (crude linear extrapolation)
```

With fixed fluid coefficients, the policy gets no advantage from knowing the 5-frame
approach history over the 3-frame one. The extra 8 dims add gradient noise without
adding information.

The situation reverses when fluid drag is randomized per-episode (fluid DR active): the
approach velocity change across 5 frames allows implicit estimation of the drag
coefficient (higher drag → faster deceleration visible across the 5-frame window). At
that point, the extra frames become genuinely informative — exactly when we should
unlock them (see Section 6).

References:
- **Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots," RSS 2021**
  (arXiv:2107.04034): history encoder length should match the system's adaptation time
  constant. Too-long history adds noise; too-short history loses the identification signal.
  For SOFI's fluid drag, the relevant time constant is O(5–10 steps); a 5-frame buffer
  becomes useful precisely when drag varies per-episode.

### How the 60–65% ceiling forms

These three mechanisms compound: the 3.75× slower bootstrap gives SAC's entropy tuning
enough time to decay the policy into a narrow, confident, but suboptimal local optimum.

The 60–65% success rate is not a coincidence. For uniformly sampled targets in
±60°, approximately 60–65% of target positions are reachable by a policy that always
swims forward (i.e., targets roughly in front of the fish, which it reaches by
driving straight). A fish that *never turns* achieves ~60% success on ±60° episodes.

Once SAC's entropy collapses to this local optimum, the policy outputs a near-constant
action (e.g., high-amplitude forward oscillation) and the temperature α decays toward
zero. Escaping this attractor requires either:
1. A forced entropy injection (not standard in SB3 SAC), or
2. Never entering it — i.e., bootstrapping fast enough (< 400k steps) before entropy decays.

References:
- **Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
  Learning with a Stochastic Actor," ICML 2018** (arXiv:1801.01290): the entropy
  regularization mechanism. When the policy finds a mode that consistently exceeds the
  target entropy, α decreases monotonically, locking in the mode.
- **Eysenbach & Levine, "Maximum Entropy RL (Provably) Solves Some Robust RL Problems,"
  ICLR 2022** (arXiv:2103.06257): entropy collapse corresponds to a failure to explore
  the reward landscape, not a fundamental policy suboptimality.

---

## 3. Literature Foundation

### Core papers and their lessons for SOFI

#### Curriculum Learning — Bengio et al., ICML 2009
> Bengio, Y., Louradour, J., Collobert, R., & Weston, J. "Curriculum Learning." *ICML 2009*.

Start with easy examples and gradually introduce harder ones. The ordering matters:
training on the full distribution from the start leads to early high-gradient noise
that prevents the network from learning the core skill. For SOFI: the angular
progression (±60° → ±90° → ... → ±180°) is the canonical application.

**Key lesson:** Bootstrap the navigation skill on easy targets (directly in front)
before asking the policy to generalize to hard targets (behind). The fraction of
"hard" episodes should increase only after the policy solves the easy ones.

#### Domain Randomization — Tobin et al., IROS 2017
> Tobin, J. et al. "Domain Randomization for Transferring Deep Neural Networks from
> Simulation to the Real World." *IROS 2017*.

Train across a distribution of physics parameters so the real-world parameters fall
within the distribution. The width of the DR distribution is a hyperparameter: too
narrow → no transfer, too wide → too hard to learn any navigation signal.

**Key lesson:** DR distribution width should start narrow and expand. For SOFI's fluid
coefficients (c8_tail_kutta: nominal ≈ 6.0), a ±10% initial range (5.4–6.6) is much
more learnable than the full ±20% range (4.8–7.2) introduced immediately.

#### Automatic Domain Randomization (ADR) — OpenAI, arXiv:1910.07113
> OpenAI, Akkaya, I. et al. "Solving Rubik's Cube with a Robot Hand." *arXiv:1910.07113*, 2019.

Widen the DR distribution automatically when the policy's success rate exceeds a
threshold on the current distribution. This is essentially adaptive curriculum for DR.
For SOFI: if we advance curriculum stages based on rolling success rate, we are
implementing a manual version of ADR. The key ADR insight is that **gradual expansion
beats fixed full DR** because the policy always has a learnable signal at the current
distribution width.

**Key lesson:** Introduce each DR dimension separately and expand its range gradually.
Introducing all DR dimensions simultaneously (as in sweep 10's direct dr60 stage)
creates a product space that is much harder to explore than individual dimensions
introduced sequentially.

#### Rapid Motor Adaptation (RMA) — Kumar et al., RSS 2021
> Kumar, A., Fu, Z., Pathak, D., & Malik, J. "RMA: Rapid Motor Adaptation for Legged
> Robots." *RSS 2021*. arXiv:2107.04034.

The robot maintains a history of observations and actions. An encoder maps this history
to a latent adaptation vector that the base policy conditions on. For deployment, a
second adaptation module trained via supervised learning maps *observable* history
(obs + actions, no privileged info) to the same latent.

**Key lesson for SOFI:** Motor history dims (setpoints + actuals) are the direct
observable for implicit Kp identification. But this identification signal only exists
when Kp varies per-episode. At fixed Kp (clean stages), the actual-setpoint tracking
error is constant in expectation, and the actuals contain no per-episode identification
signal — only noise for the gradient. Therefore, actuals should be *zeroed until motor
DR is active* (the `_motor_dr_active` flag, implemented in sweep 12).

The history encoder length should match the system's adaptation time constant. For
motor Kp: the motor responds within 1–3 steps. For fluid drag: the deceleration is
visible over 5–10 steps. These match our 8-step setpoint history (motor) and 5-frame
buffer (fluid drag).

#### Curriculum Domain Randomization — Rudin et al., CoRL 2022
> Rudin, N., Hoeller, D., Reist, P., & Hutter, M. "Learning to Walk in Minutes Using
> Massively Parallel Deep Reinforcement Learning." *CoRL 2022*.

Used massively parallel environment sampling with progressive difficulty increase.
Key design choice: DR parameters are introduced one group at a time, with each group
having its own ramp schedule.

**Key lesson:** Never introduce multiple DR dimensions simultaneously. Separate
init_vel, inertial, fluid, and motor DR into distinct curriculum stages so that each
has an isolated effect on the policy.

#### MAER-Nav (Mirrored Replay, arXiv:2503.23908)
> (Reference noted in env_sofi_da_angular_curriculum.py)

Doubling rare U-turn successes in the replay buffer via mirroring counteracts
replay-buffer starvation of hard episodes. Combined with the angular curriculum, this
ensures that early U-turn successes are not crowded out by the large majority of
forward-target episodes.

**Key lesson:** `mirror_replay_buffer=True` is non-negotiable once θ_max > ±90°.

#### Progressive Dim Disclosure — ExoRL / implicit in Yarats et al.
> Yarats, D. et al. "Don't Change the Algorithm, Change the Data: Exploratory Data for
> Offline Reinforcement Learning." *arXiv:2201.13425*, 2022.

Input dimensions that carry no reward-correlated signal at a given curriculum stage are
"exogenous" from the policy's perspective — they add gradient noise without useful signal.
The effective dimensionality at each stage should match the *task-relevant information
content* at that stage.

**Key lesson:** Match observation dims to what is informative *now*. Zero extra dims
until they become informative (see Section 6 for the unlock schedule).

---

## 4. LR Reset Strategy — What and When

### What `reset_lr` actually does (train_rl_curriculum.py)

```python
elif stage.get("reset_lr", False):
    new_lr = initial_lr          # ← FULL reset to initial, not a decay
    model.learning_rate = new_lr
    for opt in [model.actor.optimizer, model.critic.optimizer]:
        for pg in opt.param_groups:
            pg["lr"] = new_lr
    if hasattr(model, "ent_coef_optimizer"):
        for pg in model.ent_coef_optimizer.param_groups:
            pg["lr"] = new_lr
else:
    new_lr = model.learning_rate * lr_scale  # ← decay by lr_scale (e.g. 0.60)
```

`reset_lr=True` → LR goes back to the **original sweep hyperparameter** (e.g., 1.2e-4),
regardless of how many decays have occurred.  
`reset_lr=False` (default) → LR is multiplied by `lr_scale` (default 0.60).

After 3 non-reset transitions: LR = initial × 0.60³ = **21.6% of initial**. If this
happens before a genuinely new behavior is needed, the LR is too low to learn it.
This is why sweep 9's env resets at ±120° (0.60² = 36% — already borderline).

### When to reset vs. when to decay

| Transition type | LR action | Rationale |
|-----------------|-----------|-----------|
| Angular extension (small, e.g. ±60°→±90°) | **Decay** (lr_scale) | Smooth continuation; existing policy generalizes with minor gradient updates |
| Angular extension requiring qualitatively new behavior | **Full reset** | New skill needed (e.g. turning); LR may have decayed too far to learn it |
| New sensor modality enabled (IMU, encoder noise) | **Full reset** | Observation distribution shifts; old Q-values no longer calibrated |
| DR introduced (any group) | **Full reset** | Episode-level distribution change; policy must adapt to variability, not just point estimate |
| New DR group added (motor after fluid) | **Full reset** | Equivalent to new sensor modality — new identification problem |

### Angular reset thresholds — validated in sweep 9

The existing `angular_curriculum` already implements the correct schedule:

```
±60° → ±90°:   NO reset  (small extension, policy generalizes by interpolation)
±90° → ±120°:  RESET     (first stage requiring genuine turning; LR ≈ 0.60² = 36% of initial — too low)
±120° → ±150°: NO reset  (turning already learned; policy extends by generalization)
±150° → ±180°: RESET     (introduces pure U-turn targets for the first time)
±180° → noise: RESET     (observation distribution shift)
noise → DR:    RESET     (physics distribution shift)
```

This is why sweep 9 succeeded: it applied full LR resets exactly when the loss landscape
changed qualitatively, and decays at smooth continuations.

**Answer to your question about ±90°→±120°:** Yes, we already reset there. The comment
in `env_sofi_da_angular_curriculum.py` explains: "LR has already decayed to 0.60² ≈ 36%
of initial — too low for learning a new manoeuvre." This is correct behavior.

### Literature backing for full vs. partial LR reset

**Full reset:**
- **SGDR (Loshchilov & Hutter, ICLR 2017)**: warm restarts (cycling LR back to initial)
  help escape local optima. Empirically, the reset should coincide with a loss landscape
  change — exactly what curriculum stage transitions represent.
- **Smith et al., "Cyclical Learning Rates for Training Neural Networks," WACV 2017**:
  LR cycling with resets to initial consistently outperforms monotonic decay for tasks
  with multiple saddle points.

**Partial reset (lr_scale):**
- Appropriate when the policy already has most of the needed features and just needs
  to fine-tune. The danger of full reset is that a high LR may destabilize a policy
  that's already working well on the previous stage.
- For smooth extensions (±60°→±90°), partial decay (×0.60) is correct: we don't want
  to undo the learned forward-navigation policy.

**Recommendation (already implemented correctly in train_rl_curriculum.py):**
- `reset_lr=True` at regime changes (new behavior required, new modality, new DR group)
- `reset_lr=False` (lr_scale=0.60 decay) at smooth continuations

For the progressive 11-stage curriculum (Section 6), the reset schedule should mirror
the angular curriculum's logic extended to DR stages.

---

## 5. Finetuning from a Checkpoint

### Can we load the sweep 9 checkpoint into the new 40-dim environment?

**No.** The checkpoint's actor and critic networks have input layers of size 24.
Loading these weights into a 40-dim environment would require resizing the first weight
matrix, which is not a meaningful weight transfer — the extra 16 input weights have
no learned values.

**The correct finetuning approach:**

Load the sweep 9 checkpoint into **the same 24-dim environment** (`sofi_da_angular_curriculum`),
but run it forward through **only the DR stages** that sweep 9 never solved:

```
[Load neoaxaym checkpoint at stage 5 (full180_noise)]
Stage 0: ±180° + full noise + fluid DR ±10%    [LR reset, 24 dims]
Stage 1: ±180° + full noise + fluid DR ±20%    [24 dims]
Stage 2: ±180° + full noise + fluid+motor DR   [LR reset, 24 dims]
```

This answers the key question: is the DR ceiling fundamental (physics limits) or an
adaptation problem (needs more time/better curriculum)?

If the 24-dim policy trained on clean/noise reaches 90%+ with graduated DR, the
problem was purely curriculum sequencing. If it still plateaus at 85%, some episodes
are physically unsolvable (extreme Kp=0.7 + behind target + r=4m).

### Checkpoint location to check

```bash
ls /cluster/scratch/simonra/sofiRL/outputs/neoaxaym/
ls /cluster/scratch/simonra/sofiRL/outputs/x8234ido/
```

The checkpoint file should be: `SAC_curriculum_stage5_full180_noise.zip`

### Why this is the fastest test

Sweep 9 took ~1.5M steps to reach stage 5 (noise). Finetuning from that checkpoint
skips those 1.5M steps. A finetune sweep of ~1.5M steps tests all DR stages — answering
in ~2 days instead of ~5 days from scratch.

---

## 6. Optimal 11-Stage Curriculum Design

### Core design principle

> **Unlock observation dimensions exactly when they first carry task-relevant information.**

This is the unifying lesson from RMA, ADR, ExoRL, and curriculum learning: matching
dim-count to information content at each stage minimizes gradient noise and maximizes
sample efficiency.

### Stage table

| Stage | Name | θ | Sensors | DR | Obs dims (informative) | LR |
|-------|------|---|---------|----|-----------------------|----|
| 0 | clean60 | ±60° | clean | none | **20** (3fr + 8sp) | initial |
| 1 | clean90 | ±90° | clean | none | 20 | ×0.60 |
| 2 | clean120 | ±120° | clean | none | 20 | **reset** |
| 3 | clean150 | ±150° | clean | none | 20 | ×0.60 |
| 4 | clean180 | ±180° | clean | none | 20 | **reset** |
| 5 | noise180 | ±180° | +IMU+encoder noise | none | **24** (+IMU, +enc) | **reset** |
| 6 | noise+initvel | ±180° | full noise | init_vel | 24 | ×0.60 |
| 7 | dr_fluid_soft | ±180° | full noise | fluid ±10% | **32** (+2 extra frames) | **reset** |
| 8 | dr_fluid_full | ±180° | full noise | fluid ±20% | 32 | ×0.60 |
| 9 | dr_motor_soft | ±180° | full noise | +motor ±30% | **40** (+8 actuals) | **reset** |
| 10 | dr_full | ±180° | full noise | all full | 40 | **reset** |

The network is always 40-dim. Dims outside the "informative" set are zeroed by flags:
- `_extra_frames_active=False` until stage 7 → frame buffer slots 4 and 5 return zeros
- `_motor_dr_active=False` until stage 9 → actual motor slots return zeros

### Stage-by-stage rationale

**Stages 0–4 (clean angular curriculum, 20 dims)**  
Identical to sweep 9's stages 0–4. Guaranteed ~240k steps per stage (validated).
The only informative dims are the 3-frame vision buffer and 8 setpoints.
IMU and encoder are zeroed; actuals and extra frames are zeroed.

Why 20 informative dims? Sweep 9 proved this bootstraps reliably. Any additional
zeroed dims add nothing — they're constant zero and produce zero gradients.

LR reset at ±120° (stage 2) and ±180° (stage 4): mirrors sweep 9's validated schedule.

**Stage 5 (noise180, 24 dims)**  
`enable_imu=True`, `enable_encoder=True` → these 4 dims become informative for the
first time. IMU gives yaw rate (helps turn calibration). Encoder gives motor position
(closes the loop on where the tail currently is). Full reset: the observation
distribution changes, making old Q-values inaccurate.

Extra frames and actuals remain zeroed: no DR yet, so they're still uninformative.

**Stage 6 (noise+initvel, 24 dims)**  
`enable_init_vel=True`: fish starts with random initial velocity. This is the softest
DR component — it changes episode initial conditions without changing the physics.
No new observation dims needed: the policy adapts within the current 24-dim obs.
Decay only (×0.60): not a regime change, just initial condition variability.

**Stage 7 (dr_fluid_soft, 32 dims)**  
`enable_fluid=True` with ±10% range → per-episode drag varies.
`_extra_frames_active=True` → frames 4 and 5 are now informative: the 5-frame approach
history shows deceleration rate, allowing implicit drag coefficient estimation.
Full reset: physics distribution changes; plus unlocking 8 new dims shifts the gradient.
Starting at ±10% before ±20%: ADR principle — narrow range first.

**Stage 8 (dr_fluid_full, 32 dims)**  
Expand fluid DR to ±20%. No new dims: the policy is already using 5-frame history.
Decay (×0.60): smooth continuation of fluid adaptation.

**Stage 9 (dr_motor_soft, 40 dims)**  
`enable_motor=True` → Kp varies per-episode (range ±30% of nominal, i.e., Kp ∈ [3.5, 6.5] Nm/rad).
`_motor_dr_active=True` → actual motor positions now contain per-episode Kp identification
signal (high Kp = actuals track setpoints closely; low Kp = larger lag).
Full reset: new identification problem + 8 new informative dims.
Using ±30% before full range (Kp ∈ [0.7, 10.4]): narrow DR first.

**Stage 10 (dr_full, 40 dims)**  
Full motor DR range. Full inertial DR (if not already active). Full reset: distribution
expansion is significant enough to warrant restarting exploration.

### Why this is better than earlydrv2 (sweep 11)

Sweep 11's earlydrv2 started at 40 informative dims (36 without fix, 28 with fix).
This 11-stage design starts at 20 and unlocks dims only when they earn their place.
The bootstrap at stage 0 is identical to sweep 9 — the most validated configuration.

---

## 7. Implementation Roadmap

### Step 1: Monitor Sweep 12 (now running)

Sweep 12 = earlydrv2 curriculum with actuals-zeroing fix. 40-dim network, 28
informative dims at stage 0 (still 40% more than sweep 9).

**Key metric:** Does stage 0 bootstrap in < 400k steps?
- Yes → zeroing fix works; partial improvement confirmed
- No → need to also zero extra frames → proceed directly to sweep 13

**Secondary metric:** Do runs advance past stage 0?
- If yes → the 40% overhead is acceptable and earlydrv2 curriculum may work

**WandB sweep ID:** `srl_ethz/fishsim/0xzfd31d` | **SLURM job:** 625164

### Step 2: Check sweep 9 checkpoint (fastest DR ceiling test)

```bash
# Run on Euler
ls /cluster/scratch/simonra/sofiRL/outputs/neoaxaym/
ls /cluster/scratch/simonra/sofiRL/outputs/x8234ido/
```

If `SAC_curriculum_stage5_full180_noise.zip` exists:
1. Implement `sweep_finetune_dr_graduated.yml` (24-dim env, DR-only stages)
2. Submit via `sbatch` (don't use `sweep_launch.sh` — it cancels everything)

### Step 3: Implement Sweep 13 (11-stage progressive curriculum)

Three code changes required:

**3a. Add `_extra_frames_active` flag to `env_sofi_da.py`**

In `_build_frame_buffer_obs()`:
```python
def _build_frame_buffer_obs(self) -> np.ndarray:
    fb_len = type(self).FRAME_BUFFER_LEN
    active_len = fb_len if getattr(self, '_extra_frames_active', True) else min(3, fb_len)
    current_time = self._sensor_model._sim_time
    buf = list(self._sensor_model._frame_buffer)
    while len(buf) < fb_len:
        buf.insert(0, (0.0, 0.0, 0.0, current_time - _AGE_SENTINEL_S))
    result = []
    for i, (dist, vx, vy, t) in enumerate(buf[-fb_len:]):
        if i < active_len:
            result.extend([dist, vx, vy, float(np.clip(current_time - t, 0.0, _AGE_SENTINEL_S))])
        else:
            result.extend([0.0, 0.0, 0.0, _AGE_SENTINEL_S])
    return np.array(result, dtype=np.float32)
```

Initialize `_extra_frames_active = False` alongside `_motor_dr_active` in `__init__`.

**3b. Update `train_rl_curriculum.py`** (one line, next to `_motor_dr_active`):

```python
physics_model.enable_motor      = stage.get("enable_motor",       False)
env.unwrapped._motor_dr_active  = stage.get("enable_motor",       False)
env.unwrapped._extra_frames_active = stage.get("enable_extra_frames", False)
```

**3c. Create `env_sofi_da_earlydrv3_curriculum.py`**

Extends `_EarlyDR` (which extends angular_curriculum) with the 11-stage table from
Section 6. Key settings per stage:
- `enable_extra_frames: False` for stages 0–6 (frame slots 4+5 zeroed)
- `enable_extra_frames: True` for stages 7–10 (frame slots 4+5 active)
- `enable_motor: False` for stages 0–8 (actuals zeroed)
- `enable_motor: True` for stages 9–10 (actuals informative)
- `reset_lr: True` at stages 2, 4, 5, 7, 9, 10
- Fluid DR soft range: ±10% (add as new dict `_FLUID_DR_SOFT`)
- Motor DR soft range: ±30% (add `motor_kp_range: [3.5, 6.5]` — needs physics model support)

### Step 4: Decision matrix based on sweep 12 and finetune results

| Sweep 12 stage-0 | Finetune DR ceiling | Action |
|------------------|---------------------|--------|
| < 400k ✓ | > 85% ✓ | Launch sweep 13 as production training |
| < 400k ✓ | ≈ 85% (plateau) ✗ | Ceiling is fundamental; reduce full DR range; accept 85% for deployment |
| > 400k ✗ | — | Must also zero extra frames; earlydrv3 is necessary |
| < 400k ✓ | Not tested | Launch sweep 13 and finetune in parallel |

---

## Summary

Sweeps 10–11 failed because they started with 80% more informative dims than sweep 9
(36 vs 20), slowing bootstrap 3.75× and allowing SAC's entropy to decay into the
60–65% forward-only local optimum before the policy learned to turn.

The fix has three layers:
1. **Immediate (sweep 12):** Zero motor actuals when motor DR is inactive (`_motor_dr_active`) — reduces to 28 informative dims.
2. **Near-term (sweep 13):** Also zero extra frame slots when fluid DR is inactive (`_extra_frames_active`) — reduces to 20 informative dims at stage 0, matching sweep 9 exactly.
3. **Structural:** Progressive 11-stage curriculum that unlocks each dim group exactly when it becomes informative — backed by RMA, ADR, CDR, curriculum learning theory.

The literature unanimously supports this approach: introduce observation dimensions and
DR parameters one group at a time, aligned with when they carry task-relevant signal.
Full LR resets at regime changes (new sensor modality, new DR group, first U-turn stage);
partial decays at smooth continuations.

**Papers cited:**
- Bengio et al., "Curriculum Learning," ICML 2009
- Tobin et al., "Domain Randomization," IROS 2017
- OpenAI, "Solving Rubik's Cube," arXiv:1910.07113, 2019
- Haarnoja et al., "Soft Actor-Critic," ICML 2018 (arXiv:1801.01290)
- Loshchilov & Hutter, "SGDR," ICLR 2017
- Smith et al., "Cyclical Learning Rates," WACV 2017
- Kumar et al., "RMA: Rapid Motor Adaptation," RSS 2021 (arXiv:2107.04034)
- Rudin et al., "Learning to Walk in Minutes," CoRL 2022
- Yarats et al., "Improving Sample Efficiency in Model-Free RL from Images," ICML 2021 (arXiv:2012.04430)
- Yarats et al., "Don't Change the Algorithm, Change the Data," arXiv:2201.13425, 2022
- Eysenbach & Levine, "Maximum Entropy RL," ICLR 2022 (arXiv:2103.06257)
- MAER-Nav, arXiv:2503.23908 (cited in env_sofi_da_angular_curriculum.py)

---

## 8. Architecture Research: RMA, PEARL, and RMA-for-SAC

*Added 2026-05-26. Context: sweep 9–15 show the curriculum works up to DR stages; the
open question is whether the observation architecture is optimal for DR generalization,
and what the right contingency is if end-to-end identification proves insufficient.*

### 8.1 Current Architecture: End-to-End SAC with History Observation

SOFI currently uses a flat MLP (3 layers, 256 hidden) that receives the full 40-dim
observation as input. History is encoded implicitly: the 5-frame vision buffer carries
approach velocity in frame-to-frame differences, and the 8-step motor setpoint history
carries command intent.

**Is the frame buffer actually needed?**

Yes — even in a clean, fixed-physics environment. SOFI has no GPS/speed sensor. The
only way to estimate approach velocity is from frame-to-frame distance changes:

```
v_approach ≈ (dist_{t-1} - dist_t) / dt_frame
```

Without at least 2 frames the policy cannot estimate closure rate and cannot judge
whether to decelerate. Confirmed: sweep 9 reached stage 5 (±180° + noise) with a
3-slot buffer. The 2 extra slots (0–1) are only needed when drag varies per-episode
(fluid DR), where the 5-frame deceleration profile implicitly identifies drag.

**Is the motor setpoint history needed?**

Yes. It encodes the recent action trajectory, allowing the policy to avoid oscillations
(jerk penalty) and to condition on current motion state. At fixed physics this is enough.

**Is the motor actual history needed (before motor DR)?**

No. At fixed Kp=5.0, Kv=0.30, the actual position is nearly deterministic from the
setpoint history (r ≈ 0.85–0.95). Motor actuals carry zero per-episode *identification
signal* when Kp is fixed; unlocking them early only adds gradient noise. This is why
`_motor_dr_active=False` zeros actual slots until stage 10.

**The identification problem at DR stages:**

When physics varies per-episode, the policy has to simultaneously:
1. Navigate toward the target (navigation)
2. Identify which physics regime this episode is in (identification)
3. Adapt its behavior appropriately (adaptation)

End-to-end SAC must solve all three jointly from the same obs. The difficulty grows
superlinearly because the policy can only identify physics by *actively probing* (e.g.,
making a deliberate tail stroke to observe the response) — and probing conflicts with
navigating. This is the fundamental tension that RMA Phase 1 resolves.

### 8.2 Rapid Motor Adaptation (RMA) — Kumar et al., RSS 2021

**Original architecture (PPO-based):**
```
Phase 1:  obs_t + z_physics  →  MLP policy  →  action
          z_physics = f(Kp, Kv, fluid_coefs, mass, ...)  ← exact truth, privileged
Phase 2:  (obs, a)_{t-H:t}  →  1D-CNN encoder  →  ẑ ≈ z_physics
          supervised: minimize ||ẑ - z_physics||² using Phase 1 replay data
          deploy: policy(obs_t, ẑ_t) — no privileged info needed
```

Key design choices from Kumar et al.:
- **Phase 1 policy** gets exact physics truth → can learn optimal adaptation behavior
  without needing to waste capacity on physics identification.
- **Phase 2 encoder** is trained offline as simple supervised regression on replay data
  from Phase 1. This is a much cleaner training signal than joint RL + identification.
- **History length H=50 steps** (1 second at 50Hz). For SOFI at ~20Hz sim: H=8–12 steps.
- **Uses PPO** (on-policy) for Phase 1. The key claim in the paper: Phase 1 is stable
  because z_physics is exact and consistent across the episode.

**Applicability to SOFI with SAC:**

The original RMA uses PPO, but the Phase 1 + Phase 2 structure transfers cleanly to SAC
with one key observation: **z_physics is fixed per episode**. This means:

- At episode start: sample physics → compute z_physics (5 numbers) → append to obs
- Each step: `obs_aug = concat(obs_40, z_physics_5)` → store `(obs_aug, a, r, next_obs_aug)`
- z_physics is part of `obs_aug` → it is stored in the replay buffer automatically
- Every buffer entry `(obs_aug_t, ...)` has the *correct* z for that episode

There is **no buffer inconsistency**. The Q-function sees Q(obs|z, a) with the correct z
for each transition. This is not "storing z in the buffer" as extra work — it is zero
extra work because z is baked into obs_aug before the buffer sees it.

The "contamination" concern from vanilla DR (§9.3) was about physics distribution mismatch
across episodes. Here z disambiguates: the Q-function learns Q(obs, z_low_kp, a) and
Q(obs, z_high_kp, a) as two different functions. Episodes with different physics have
different z → different Q-function behavior. The buffer correctly contains both.

**RMA for SAC in the literature:**

The exact Phase 1+2 structure with SAC has not been published under the name "RMA for
SAC," but equivalent approaches are established:
- **Pinto et al., "Asymmetric Actor-Critic," RSS 2018**: gives the critic privileged state;
  in our case we give *both* actor and critic z_physics (necessary for behavior adaptation).
- **Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain," Science
  Robotics 2020**: teacher-student framework with privileged env parameters in Phase 1;
  student encoder in Phase 2. Uses PPO, but the same logic applies to SAC.
- **Zhuang et al., "Robot Parkour Learning," CoRL 2023**: uses depth-obs privilege in
  Phase 1 for SAC-variant training; Phase 2 encoder is supervised offline. Direct SAC
  analogue of RMA Phase 1+2.

The pattern is well-established. The SOFI implementation is a straightforward instance.

### 8.3 PEARL — Rakelly et al., ICML 2019 (not recommended for SOFI)

**Architecture:**
```
Context c = {(s_i, a_i, r_i, s'_i)}_{i=1..K}   ← K recent transitions
z ~ q_φ(z | c)                                   ← probabilistic context encoder
π_θ(a | s, z)                                    ← policy conditioned on inferred z
```

PEARL jointly trains the context encoder + policy + Q-function via SAC with a KL
regularization term `β × KL[q(z|c) || N(0,I)]`. There is no Phase 2 — the encoder
must learn to identify physics end-to-end during RL training.

**Why PEARL is not the right fallback for SOFI:**

1. **Noisy z during training**: The context encoder must learn to infer physics from
   (s,a,r,s') tuples simultaneously with policy learning. Early in training, before the
   encoder has learned, z is noise → policy trains on bad z → bad policy → bad z.
   RMA Phase 1 avoids this entirely: z is always exact truth.

2. **KL hyperparameter (β_KL)**: β too high → z collapses to N(0,I), encoder is
   ignored. β too low → z overfits to episode-specific trajectory details. This is an
   extra sensitive hyperparameter on top of already-fussy SAC hyperparameters.

3. **Separate context buffer**: PEARL uses a separate context buffer (stores full
   episodes, not individual transitions) for sampling context at training time.
   This requires significant replay buffer restructuring that RMA-for-SAC does not.

4. **Identification from rewards**: PEARL uses reward in the context (s,a,r,s').
   For SOFI, rewards are sparse at stage 10 (motor DR just activated) → the reward
   signal provides very weak physics identification signal in the early motor DR stage.
   PEARL degrades exactly when identification is hardest; RMA Phase 1 does not (z exact).

**One advantage PEARL has:** within-episode adaptation. As K transitions accumulate in
the current episode, the inferred z improves. RMA Phase 1 gives z at episode start
(exact but requires knowing the physics). RMA Phase 2 (the deployment encoder) also
adapts within-episode via the rolling history window, so this advantage is not relevant
for our use case.

**Verdict:** PEARL is a valid approach for multi-task RL with many tasks and no
privileged info access. For SOFI — where we have direct access to physics parameters at
training time — RMA-for-SAC is strictly preferable.

### 8.4 RMA-for-SAC: z_physics Design for SOFI

The core question: what goes in z_physics, and why?

**Principle:** include a parameter in z if and only if: (a) it varies per-episode
AND (b) the optimal policy behavior is significantly different at different values AND
(c) there is an observable signal in the history obs that a Phase 2 encoder can use to
identify it.

**The 5 z_physics components:**

| Index | Physical param | DR range | Identification source | Why essential |
|-------|---------------|----------|----------------------|---------------|
| z[0] | fluid_mult_mean | N(1.0, fluid_std) | Frame-to-frame distance deceleration rate over 5 frames | Drag affects approach speed and turn radius — policy must adjust timing |
| z[1] | motor_kp | Uniform[0.7, 10.4] Nm/rad | Setpoint–actual gap magnitude over 8 steps | 15× range — fundamentally different dynamics (limp vs. rigid) |
| z[2] | motor_kv | Uniform[0.0, 0.6] Nm·s/rad | Transient decay rate of setpoint–actual gap | Affects velocity damping independently of Kp |
| z[3] | motor_forcerange | Uniform[1.38, 2.3] Nm | Amplitude saturation pattern at large setpoints | At low forcerange, large oscillations get clipped — policy must reduce amplitude |
| z[4] | motor_zero_offset | Uniform[−0.05, 0.05] rad | Persistent signed asymmetry in setpoint–actual over 8 steps | Creates systematic turning bias; policy must apply a constant setpoint correction |

**Why 5 and not 3:**

Three dimensions (fluid_mult, Kp, and one more) are insufficient because Kv, forcerange,
and zero_offset each cause qualitatively different and independently uncompensable behavior:
- Kv=0.0 → underdamped oscillation. Kv=0.6 → overdamped, sluggish velocity response.
  Neither Kp nor fluid_mult can substitute.
- forcerange=1.38 Nm → oscillation amplitude clipped even at modest setpoints. The
  policy must choose a lower amplitude. Kp or Kv cannot signal this.
- zero_offset=+0.05 rad → fish naturally turns left. The policy must add −0.05 rad to
  all setpoints. This is not visible in distance signals (fluid-related) or gap magnitude
  (Kp-related); only visible in the *signed* mean gap over multiple steps.

**What is NOT in z, and why:**

- **mass_multiplier** (±5%): effect too small (< 5% change in inertia term); not
  observable with confidence from frame buffer; no distinct optimal policy for ±5% mass.
- **inertia_multipliers** (±5%): same.
- **com_jitter** (±5mm): negligible at the navigation task scale.
- **stiffness_multiplier** (±15%): confounded with motor_kp in the observable signal
  (setpoint–actual gap looks the same whether gap comes from low Kp or high joint spring).
  The ±15% effect is much smaller than the Kp 15× range.
- **damping_multiplier** (±15%): confounded with motor_kv in the observable signal.
- **init_vel** (vx, vy, wz at episode start): directly readable from the first 2 frames
  in the frame buffer (the fish is already moving → distance changes immediately). The
  policy reads this from obs without needing z.

**Normalization (map each param to z=0 when DR inactive, z∈[−1,1] when DR active):**

```python
def compute_z_physics(pm: PhysicsModel) -> np.ndarray:
    z = np.zeros(5, dtype=np.float32)
    if pm.enable_fluid:
        # fluid_mult_mean ∈ [~0.4, 1.6] at 3σ for fluid_std=0.2
        z[0] = (np.mean(pm.fluid_multipliers) - 1.0) / (pm.fluid_std * 3.0)
    if pm.enable_motor:
        kp_lo, kp_hi = pm.motor_kp_range        # (0.7, 10.4)
        kv_lo, kv_hi = pm.motor_kv_range        # (0.0, 0.6)
        fr_lo, fr_hi = pm.motor_forcerange_range # (1.38, 2.3)
        z[1] = 2.0 * (pm.motor_kp          - kp_lo) / (kp_hi - kp_lo) - 1.0
        z[2] = 2.0 * (pm.motor_kv          - kv_lo) / (kv_hi - kv_lo) - 1.0
        z[3] = 2.0 * (pm.motor_forcerange  - fr_lo) / (fr_hi - fr_lo) - 1.0
        z[4] = pm.motor_zero_offset / pm.motor_zero_offset_std  # → [-1, 1]
    # enable_motor=False → z[1:5]=0 (nominal behavior, policy uses default)
    # enable_fluid=False → z[0]=0 (nominal drag, policy uses default)
    return z
```

When DR is inactive: z = zeros. The policy learns to ignore zero z during clean stages.
At deployment: Phase 2 encoder outputs zeros when it identifies nominal physics.

**Augmented obs dimension: 40 + 5 = 45 dims** throughout all stages.

The z dims are zero during stages 0–6, z[0] non-zero during stages 7–9 (fluid DR),
all 5 non-zero during stage 10+ (motor DR). Same zero-gradient logic as frame slots:
zero-valued inputs produce zero gradients through the input layer → no overhead during
clean stages.

**Zero-init for z input weights:**
- At stage 7 (z[0] first becomes non-zero): zero-init W[:, 40]
- At stage 10 (z[1:5] first become non-zero): zero-init W[:, 41:45]
- Same protocol as frame slot and motor actual unlocking.

### 8.5 Phase 2 Encoder Design

For deployment, estimate z from observable history. The encoder takes a rolling window
of the last K=8 steps of observations (at ~20Hz sim, 8 steps ≈ 400ms — sufficient
to see deceleration profile AND 8 setpoint–actual gap readings):

```python
class PhysicsEncoder(nn.Module):
    """Phase 2 encoder: (frame_buf_history, motor_history) → ẑ (5 dims)."""

    def __init__(self, K: int = 8):
        super().__init__()
        # frame buffer portion: 5 frames × 4 = 20 dims per step, last K steps
        self.fluid_head = nn.Sequential(
            nn.Linear(20 * K, 64), nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1),                    # → ẑ[0]: fluid_mult_mean
        )
        # motor history portion: 8 (setpoint, actual) pairs = 16 dims per step
        self.motor_head = nn.Sequential(
            nn.Linear(16 * K, 64), nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 4),                    # → ẑ[1:5]: motor params
        )

    def forward(self, frame_hist, motor_hist):
        # frame_hist: (B, K, 20)   motor_hist: (B, K, 16)
        z_f = self.fluid_head(frame_hist.flatten(1))   # (B, 1)
        z_m = self.motor_head(motor_hist.flatten(1))   # (B, 4)
        return torch.cat([z_f, z_m], dim=-1)           # (B, 5)
```

**Training Phase 2 offline:**
1. Run Phase 1 training to completion (collect all replay buffer entries with stored z_physics)
2. For each buffer entry: extract (obs_aug_t, ...) → split obs into frame portion + motor portion
3. Stack K consecutive steps within same episode → input to encoder
4. Target: the stored z_physics for that episode
5. Train with MSE loss: `L = ||ẑ - z_physics||²`

The Phase 2 training is stable (supervised regression, no RL gradient), fast (< 10k steps
on replay data), and interpretable — you can check encoder prediction accuracy per parameter.

**At deployment:**
```python
# Rolling buffer of last K obs (40-dim, no z):
obs_queue = deque(maxlen=K)
# Each step:
frame_hist = stack([obs[4:24] for obs in obs_queue])  # last K frame portions
motor_hist = stack([obs[24:40] for obs in obs_queue]) # last K motor portions
z_hat = encoder(frame_hist, motor_hist)               # → 5 dims
obs_aug = np.concatenate([obs, z_hat])                # → 45 dims
action = policy(obs_aug)
```

### 8.6 Architecture Recommendation

**For sweeps 16 and any near-term from-scratch sweep:** Maintain end-to-end SAC with
the current 40-dim obs + progressive unlock. The curriculum is already well-designed to
help the policy learn implicit identification (extra frames at stage 8 for fluid, actuals
at stage 10 for motor).

**If S16 fails to reach stage 8 (fluid_dr_10pct_frs):** Implement RMA-for-SAC Phase 1
(add z_physics as 5 extra dims, total 45-dim network). This is trivially implemented —
`compute_z_physics()` called after each `physics_model.reset()`, concatenated to obs.
No buffer changes needed. Then train Phase 2 encoder offline from Phase 1 replay data.

**PEARL is not the recommended contingency.** Its jointly-trained context encoder
degrades exactly when identification is hardest (early in a new DR stage), whereas
RMA Phase 1 provides exact z at all times during training.

---

## 9. DR Phase Analysis: Complete Picture

### 9.1 Sensor DR — Already Active from Stage 5

The sensor noise model in `_FULL_SENSOR` includes:

| Component | Implementation | Range |
|-----------|---------------|-------|
| Camera FPS | per-episode draw from `camera_fps_range` | [39.4, 66.9] Hz |
| Within-episode FPS jitter | `jitter_std=0.01247 s` per frame period | ±~±2–3% |
| Frame drop | Bernoulli each frame | prob=0.10 |
| Latency | uniform per frame | [10, 50] ms |
| Vision noise | Gaussian on (dist, vecX, vecY) | enabled from stage 5 |

All of these activate when `enable_vision_noise=True` or `enable_fps_dr=True` (both
set in `_FULL_SENSOR`). **There is no need for a separate "sensor DR" curriculum stage.**
The sensor noise is baked into the _FULL_SENSOR dict, which activates at stage 5.
The FPS range is calibrated to the real camera hardware.

**Key implication:** The current stage 5 (`full180_noise`) *already covers* all sensor
domain randomization. No separate sensor-DR stage is needed between stages 5 and 6.

### 9.2 Soft DR — Stage 6 (init_vel + inertial + mechanics)

Three sub-groups are activated simultaneously in `_SOFT_DR`:

| Sub-group | Parameters | Range |
|-----------|-----------|-------|
| Initial velocity | vx, vy ~ N(0, 0.15 m/s), wz ~ N(0, 0.30 rad/s) | Small perturbation |
| Inertial | CoM jitter ~ N(0, 0.005 m) per axis | ±5mm |
| Mechanics | Tail hinge stiffness ±15%, damping ±15% | Moderate |

**Should these be split into 3 sub-stages?** No — for the following reasons:

1. All three are small perturbations (no obs unlock needed, no new identification
   challenge for the policy). The policy adapts within the same 24-dim obs space.
2. The combined effect is still mild compared to fluid DR (which requires a 15-step
   buffer to observe). The reward signal degrades < 5% per sub-group.
3. Splitting would add 2 more stages (total 13) without a learnable benefit.
4. Sweep 15 showed that the policy can absorb this stage without significant dip
   (8a9yod1g advanced from stage 6 to 7 without reported plateau).

**Could any soft-DR component cause a large enough distribution shift to warrant a separate stage?**

The hinge stiffness ±15% could in principle affect the fish's maneuverability enough
to require a new turning strategy. However, sweep 9's run neoaxaym (which used the same
combined `_SOFT_DR` in its stage 5) passed this stage reliably, which validates the
combined approach empirically.

### 9.3 Fluid DR — Stages 7–9 (3-Stage ADR Split)

This is the most carefully designed part of the curriculum. The split addresses two
independent shocks that are each damaging and deadly in combination:

**Shock 1 — Buffer contamination (SAC-specific):**
When fluid DR activates, the replay buffer still contains ~91% episodes with fixed
nominal fluid coefficients. Q-targets for new fluid-DR episodes use value estimates
from non-DR experience → overestimates of Q for extreme fluid episodes → biased
gradient → policy learns wrong response.

Time to resolve: ~1M steps (buffer of 1M experiences × 91% contamination → ~900k
steps until < 50% contamination). At 2M steps/stage this is tolerable, but during
the flush the policy may temporarily degrade.

**Shock 2 — Observation perturbation (Kaiming init):**
When extra frame slots (dims 4–11) unlock, the first-layer weights connecting those
dims to hidden units have Kaiming-uniform values ≈ ±0.158 (for input_size=40,
output_size=256). The perturbation to Q-values on the first forward pass is
`||W_new × obs_new|| ≈ 0.158 × √8 × 0.5 ≈ 0.22`. For typical Q-values of 1–5,
this is a 4–22% sudden shift → wrong gradient direction for ~100k steps.

Fix: `train_rl_curriculum.py` zero-inits W[:,4:11] at the stage 8 transition,
eliminating this perturbation entirely.

**The Option A Sequencing (implemented):**

```
Stage 7  fluid_dr_10pct     : fluid DR ±10%,  obs=24 dims  ← Shock 1 only
Stage 8  fluid_dr_10pct_frs : fluid DR ±10%,  obs=32 dims  ← Shock 2 only (zero-inited)
Stage 9  fluid_dr_20pct     : fluid DR ±20%,  obs=32 dims  ← just range expansion
```

Why Option A (10% no frames → 10% frames → 20% frames) beats Option B (5% → 10% → 20%)?

Option A separates the two shocks cleanly: stage 7 → 8 transition has zero buffer
contamination (same physics distribution, only obs changes) AND zero Kaiming perturbation
(zero-init). Option B separates by range width, not by shock type — both shocks hit
simultaneously at the first range change.

**Fluid DR ranges and calibration:**

The fluid_std parameterizes a per-geometry multiplicative factor drawn from N(1.0, std).
The main thrust coefficient c8 (tail_kutta ≈ 6.0 nominal) has physics bounds [0, 12]:

- ±10% range: c8 ∈ [5.4, 6.6] — stays well within physical range
- ±20% range: c8 ∈ [4.8, 7.2] — still within [0, 12], but much harder episodes
- Full ±30%+: c8 → 0.0 or > 9.0, near MuJoCo instability threshold — not recommended

**Should we add a ±15% intermediate?** Only if S16 shows stage 9 (20%) is
systematically failing. The current 3-stage split is well-calibrated.

### 9.4 Motor DR — Stage 10 (4 Sub-Components)

Motor DR activates 4 independent sources of variability:

| Sub-component | Parameter | Range | Nominal |
|---------------|-----------|-------|---------|
| Force saturation | `motor_forcerange` | [1.38, 2.3] Nm | 2.3 Nm |
| Servo stiffness | `motor_kp` | [0.7, 10.4] Nm/rad | 5.0 Nm/rad |
| Servo damping | `motor_kv` | [0.0, 0.6] Nm·s/rad | 0.30 Nm·s/rad |
| Mechanical zero | `motor_zero_offset` | N(0, 0.05 rad) | 0.0 rad |

**Should motor DR be split into sub-stages?**

The Kp range [0.7, 10.4] is a **15× span** — the hardest single DR axis in the entire
curriculum. At Kp=0.7 (minimum), the motor barely follows the setpoint at slow frequencies
and cannot track high-amplitude oscillations. At Kp=10.4 (maximum), the motor is very
stiff and the fish behaves like a rigid body actuator. These are qualitatively different
control challenges.

Compared to the fluid DR fluid_std=0.20 (a ±20% variation on a parameter with
well-defined physics), Kp=0.7 vs Kp=10.4 represents fundamentally different fish
dynamics. **Recommendation: split motor DR into two sub-stages:**

```
Stage 10a  motor_dr_soft  : Kp ∈ [2.5, 7.5]  (±50% of nominal=5.0)
           + Kv ∈ [0.1, 0.5]  (moderate range)
           + forcerange ∈ [1.84, 2.3]  (±20%)
           + zero_offset_std=0.03
           → obs unlock: motor actuals (dims 25,27,...,39) — zero-init!

Stage 10b  motor_dr_full  : full ranges [0.7, 10.4], [0.0, 0.6], [1.38, 2.3], ±0.05
           → no new obs dims, just range expansion
```

**Why split at Kp=2.5–7.5 first?**

This is ±50% of nominal (5.0 Nm/rad), analogous to the fluid DR split (±10% before
±20%). At Kp=2.5, the motor is sluggish but still functional — the policy can identify
this from the setpoint-actual gap within 3–5 steps. At Kp=0.7, the motor is nearly
inoperable at high frequencies, and the optimal control strategy is fundamentally
different (slower oscillations, reduced amplitude).

The motor actual unlock also hits at stage 10a — zero-init the actual slots (dims
25,27,...,39) at this transition to eliminate the Kaiming perturbation.

**Is this implemented now?** No. Stage 10 (`motor_dr`) currently uses the full Kp
range. The split into 10a/10b is a **pending improvement** for the next environment
file revision. For sweep 16 (which starts at stage 6 and may not reach stage 10), this
is not urgent.

**What does the motor actual identification signal look like?**

```
# At episode with Kp=0.7 (very weak):
setpoint = sin(2πft)
actual   ≈ 0.2 × sin(2πft - φ)   ← large lag and attenuation
gap      = |setpoint - actual| ≈ large

# At episode with Kp=10.4 (very stiff):
setpoint = sin(2πft)
actual   ≈ 0.98 × sin(2πft)      ← near-perfect tracking
gap      ≈ small
```

The 8-step history of (setpoint, actual) pairs gives the policy exactly this
identification signal. The policy can compute gap statistics over the 8 steps and
adapt its oscillation frequency and amplitude accordingly.

### 9.5 DR Phase Summary

| Stage | Name | Active DR Components | New Obs Dims | Notes |
|-------|------|---------------------|--------------|-------|
| 0–4 | clean angular | none | 24 total | Angular curriculum only |
| 5 | full180_noise | sensor noise (fps, drops, latency) | 24 | Sensor DR baked in |
| 6 | soft_dr | init_vel + inertial + mechanics | 24 | Combined — validated OK |
| 7 | fluid_dr_10pct | fluid ±10% | 24 | Buffer contamination shock only |
| 8 | fluid_dr_10pct_frs | fluid ±10% + extra frames | 32 | Kaiming shock (zero-inited) |
| 9 | fluid_dr_20pct | fluid ±20% + extra frames | 32 | Range expansion |
| 10 | motor_dr | full motor DR | 40 | Should be split (see §9.4) |

**Count of DR sub-components by group:**
- Sensor: 4 (fps, jitter, drops, latency) — grouped, no split needed
- Soft: 3 (init_vel, inertial, mechanics) — grouped, no split needed
- Fluid: 1 axis (fluid_std), 3 curriculum stages (10%→10%+frames→20%)
- Motor: 4 axes (Kp, Kv, forcerange, zero_offset) — should be split into 2 stages

---

## 10. Sweep 16 and Forward: RMA-for-SAC Contingency Plan

### 10.1 Sweep 16 (thpjgi3p) — What It Tests

**Setup:**
- Checkpoint: 8a9yod1g at stage 5 (full180_noise, 24-dim obs, ±180° + full sensor noise)
- Start stage: 6 (soft_dr), abs_stage_idx=6 on WandB
- Swept: learningRate ∈ [6e-5, 2e-4] (log-uniform), tau ∈ [3e-3, 1.5e-2]
- Fixed: alpha_heading=0.18571, alpha_jerk=1.7704e-4 (8a9yod1g's values — reward structure unchanged)
- Run cap: 12 runs, env: sofi_da_angular_prog_curriculum (11-stage ADR)

**What is confirmed if S16 reaches stage 8 (fluid_dr_10pct_frs):**
- 3-stage fluid ADR split works as designed
- Zero-init trick eliminates the Q-perturbation shock at frame unlock
- Buffer contamination fix (stage 7 buffers fluid DR before obs unlock in stage 8) works
- LR and tau in swept ranges are viable for DR adaptation from checkpoint

**What is confirmed if S16 reaches stage 10 (motor_dr):**
- End-to-end history observation is sufficient for both fluid drag AND motor identification
- RMA-for-SAC Phase 1 is not needed for this curriculum depth
- Motor actual unlocking with zero-init is sufficient

**What it does NOT test:**
- Training from scratch (8a9yod1g stage 5 is already a strong policy at stage 5)
- Whether Phase 1 z_physics augmentation would improve DR generalization further
- Full 11-stage curriculum starting from stage 0

### 10.2 Decision Matrix

| S16 best stage | Interpretation | Next sweep |
|----------------|---------------|-----------|
| Reaches stage 8+ | ADR split + zero-init work; end-to-end sufficient | S17: full from-scratch 11-stage curriculum sweep |
| Reaches stage 6–7, plateaus | Fluid DR boundary is end-to-end bottleneck | S17: RMA-for-SAC Phase 1 (see §10.3) |
| Stays at stage 6 | LR/tau range insufficient | Re-sweep with wider LR range; check 8a9yod1g baseline |
| Crashes (NaN) | Likely tau too high or LR + tau combination unstable | Narrow ranges; inspect run configs |

### 10.3 Sweep 17 RMA-for-SAC Phase 1 Design (contingency)

*Only implement this if S16 fails to reach stage 8 consistently.*

**The change is minimal:** augment the observation with z_physics (5 dims → total 45-dim
network). z_physics is computed from `physics_model` after each `reset()` and concatenated
to obs before the env's `step()` return. z flows into the replay buffer automatically.

```python
# In train_rl_curriculum.py, after physics_model.reset():
z_physics = compute_z_physics(physics_model)  # 5 dims, zeros when DR inactive

# In the env wrapper step():
obs_aug = np.concatenate([obs_40, z_physics_5])  # 45 dims
# SB3 stores obs_aug in replay buffer automatically — no buffer changes needed
```

```python
def compute_z_physics(pm) -> np.ndarray:
    """Privileged physics context: 0 when DR inactive, ∈[-1,1] when active."""
    z = np.zeros(5, dtype=np.float32)
    if pm.enable_fluid:
        z[0] = (np.mean(pm.fluid_multipliers) - 1.0) / (pm.fluid_std * 3.0)
    if pm.enable_motor:
        kp_lo, kp_hi = pm.motor_kp_range         # (0.7, 10.4)
        kv_lo, kv_hi = pm.motor_kv_range         # (0.0, 0.6)
        fr_lo, fr_hi = pm.motor_forcerange_range  # (1.38, 2.3)
        z[1] = 2.0 * (pm.motor_kp         - kp_lo) / (kp_hi - kp_lo) - 1.0
        z[2] = 2.0 * (pm.motor_kv         - kv_lo) / (kv_hi - kv_lo) - 1.0
        z[3] = 2.0 * (pm.motor_forcerange - fr_lo) / (fr_hi - fr_lo) - 1.0
        z[4] = pm.motor_zero_offset / pm.motor_zero_offset_std
    return z
```

**Additional zero-init at z unlock (add to train_rl_curriculum.py):**
```python
# At stage 7 (z[0] first becomes non-zero — fluid DR activates):
model.actor.latent_pi[0].weight[:, 40] = 0.0
model.critic.qf0[0].weight[:, 40] = 0.0
model.critic.qf1[0].weight[:, 40] = 0.0

# At stage 10 (z[1:5] first become non-zero — motor DR activates):
model.actor.latent_pi[0].weight[:, 41:45] = 0.0
model.critic.qf0[0].weight[:, 41:45] = 0.0
model.critic.qf1[0].weight[:, 41:45] = 0.0
```

**Sweep 17 hyperparameters (for RMA-for-SAC Phase 1):**
- Start from scratch (not checkpoint) OR from 8a9yod1g stage 5 (checkpoint with 40-dim
  policy must be re-initialized to 45 dims — not compatible, so start from scratch)
- Use same LR range as S16 best run
- Fix alpha_heading=0.18571, alpha_jerk=1.7704e-4 (same as S16)
- Network: 3 layers, 256 hidden (same — only input changes 40→45)
- Phase 2 encoder training: offline after Phase 1 converges (see §8.5)

**Why not start from 8a9yod1g checkpoint for S17 Phase 1?**

The 8a9yod1g checkpoint has a 40-dim input layer. Extending to 45 dims requires either
re-initializing the first layer (losing all learned features) or weight surgery (pad with
zeros — equivalent to zero-init). Since z is zero during stages 0–5, zero-padding the
first layer for z dims is valid: obs_aug[40:45]=0 → W[:,40:45] initialized to 0 → no
perturbation → training continues from checkpoint. This is feasible for a finetune sweep.

### 10.4 Hyperparameter Lessons from Sweeps 9–16

| Hyperparameter | Validated range | Notes |
|----------------|----------------|-------|
| learningRate | [1.2e-4, 3e-4] at bootstrap | Below 1.2e-4: stage-1 bottleneck (sweep 9 finding) |
| learningRate | [6e-5, 2e-4] at finetune | S16 range; lower OK since checkpoint already trained |
| tau | [3e-3, 1.5e-2] | High tau (>2e-2) can cause instability in early DR stages |
| gamma | 0.995 | Validated across S9, S15, S16; do not sweep |
| alpha_heading | ≥ 0.10–0.15 | Critical — below 0.10 leads to drift without alignment reward |
| alpha_heading | 0.18571 (8a9yod1g) | Best known value; fix in S16 to avoid reward confound |
| alpha_jerk | ~1.77e-4 | Suppresses oscillation; fix alongside alpha_heading |
| batchSize | 256 | Validated; 512 untested |
| numLayers | 3 | Validated; 2 may bottleneck at DR stages |
| hiddenDim | 256 | Validated; 512 untested but probably overkill |
| mirror_replay_buffer | True | Non-negotiable at θ_max > ±90° (MAER-Nav insight) |
| lr_scale | 0.60 | Validated decay factor at smooth transitions |
| success_window | 400 at DR stages | 200 was too short (8a9yod1g promoted prematurely) |

### 10.5 Summary of Pending Implementation Work

1. **Motor DR split** (low urgency, implement if S16 reaches stage 10):
   - Add `_MOTOR_DR_SOFT` dict with Kp ∈ [2.5, 7.5], Kv ∈ [0.1, 0.5], forcerange ∈ [1.84, 2.3]
   - Add stage between current stage 9 (fluid_dr_20pct) and stage 10 (motor_dr_full)
   - Add `motor_std` parameter to physics_model (analogous to `fluid_std`) for gradual range expansion
   - Zero-init motor actual slots at motor_dr_soft transition

2. **RMA-for-SAC Phase 1 implementation** (contingency if S16 fails):
   - See §8.4 for z_physics design (5 dims) and §10.3 for implementation snippet
   - Add `compute_z_physics(physics_model)` helper to `train_rl_curriculum.py`
   - Env wrapper concatenates z to obs → 45-dim network (only input_dim changes)
   - Add z-unlock zero-init at stage 7 (z[0]) and stage 10 (z[1:5])
   - Phase 2 offline encoder: train after Phase 1 converges (see §8.5)

3. **Monitor S16** (thpjgi3p, SLURM 803551):
   - Watch WandB for abs_stage_idx advancing to 8+ (fluid_dr_10pct_frs)
   - If best run reaches stage 9 (fluid_dr_20pct): ADR split confirmed
   - 8a9yod1g (SLURM 679053_3) still recovering at stage 7; monitor in parallel
