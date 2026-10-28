# Sim2Real Findings: Motor Model, Fluid Coefficients, and the High-Frequency Exploit

**Date:** 2026-06-25  ·  **Scope:** why the latest Direct-Action policy swims unrealistically,
and the fix. Companion to `CANONICAL_POLICIES.md`.

---

## TL;DR

The re-identified **motor model is good but caps *velocity*, not *frequency***. The **fluid
coefficients were only tuned on ≤4 Hz, large-amplitude swim data**, and MuJoCo's stateless
(velocity²) fluid model has to *inflate* its coefficients to fake the thrust that real fish
get from unsteady vortices. Together these leave a loophole: a policy can command **fast,
small-amplitude tail flicks** that the motor barely tracks but the inflated fluid turns into
**non-physical forward thrust**. The latest DA policy learned exactly this. **Fix: a low-pass
filter on the motor setpoint (4.5 Hz)** to keep the commanded motion inside the frequency band
where the fluid model is actually valid.

---

## 1. The motor model

**What it is.** A MuJoCo position actuator standing in for the real Dynamixel + 200:1 gearbox +
tendon drive, re-identified open-loop on RL-deployment + field data (`fluidID/scripts/23_motor_id.py`,
2026-06-19, joint-damping/kv-free fit):

- `kp = 5.41`, `kv = 0`
- **joint damping `b = 0.48`** → back-EMF / torque-speed droop applied *outside* the actuator
  force clip → a **hard velocity cap ≈ 4.8 rad/s**
- `armature = 0.0113` (reflected rotor inertia), `forcerange = 2.3 N·m`
- transport **dead-time ≈ 14 ms** (substep ring buffer; DR 8–18 ms)

**What it succeeds at.**
- Reproduces the real motor's tracking at the tested frequencies (held-out ≈ 2.27° error, beats
  prior fits).
- Realistic **velocity saturation** via `b` (the real motor can't slew arbitrarily fast).
- Captures the command→response **lag / dead-time**.

**What it fails at (the key gap).**
- It caps **velocity, not frequency**. A *small-amplitude, high-frequency* command keeps the
  peak velocity under the cap, so the motor doesn't refuse it.
- At high frequency the motor **under-tracks badly** — measured: policy commands a setpoint of
  **±1.0 rad p2p but the motor only moves ±0.27 rad (≈27% tracking)**. So the *actual* tail
  motion is tiny — but it is **fast**.
- The motor is doing its job (it heavily attenuates high-freq), but it does **not prevent the
  policy from issuing high-freq commands**, and the residual fast micro-motion is enough to be
  exploited downstream by the fluid model.

---

## 2. The fluid coefficients and how they were tuned

**What they are.** A 15-coefficient **segmented quasi-steady** MuJoCo fluid model
(head / tail / fin × {blunt drag, slender added-mass, angular drag, Kutta lift/thrust, Magnus}).
Two relevant sets: **E13a** (field-SysID) and **refitB_fwdonly** (re-ID'd with an explicit
backward-swimming penalty).

**How they were tuned.** Fit (CMA-ES / windowed multiple-shooting) to **actual swim runs at
1–4 Hz tail-beat frequencies** (the `20260401_first_swim` field data: 1.0/2.0/3.0/4.0 Hz runs).
So the coefficients were optimized to reproduce trajectories from **low-frequency,
large-amplitude** strokes.

**The fundamental limitation.**
- MuJoCo's fluid model is **stateless / quasi-steady**: force is a function of the *instantaneous*
  body velocity and scales with **velocity²**. It has no memory of the flow.
- Real fish thrust is largely **unsteady** — added-mass reaction, vortex shedding, wake capture,
  history effects — which a stateless model **cannot represent**.
- To match the real ≤4 Hz trajectories anyway, the optimizer must **inflate** the coefficients
  (e.g. tail/fin terms of 12–18) so the quasi-steady model produces enough force *at the tested
  frequencies*.

**Why this is a trap.** Because force ∝ velocity² and tail-tip velocity = amplitude × 2π·f, the
inflated coefficients generate **large force for *any* high-velocity motion** — including the
**fast, small-amplitude flicks that were never in the tuning data**. The fluid model is only
trustworthy in the **≤4 Hz / large-amplitude** regime it was fit on. Outside it — fast, tiny
oscillations — it produces **non-physical thrust**. In short: *fast frequencies with small
amplitudes are not captured or validated by the fluid fit.*

---

## 3. What we found — and the fix

### The exploit (measured)
The latest Direct-Action policy (`sofi_da_rl_policy`, trained with **no setpoint filter**) learned
to abuse exactly this loophole. Spectral analysis of a forward-swimming episode:

| | dominant setpoint freq | **setpoint power > 5 Hz** | **> 8 Hz** | motor dom. freq | tail bend (p2p) | COM moved |
|---|---|---|---|---|---|---|
| **DA-MotorID** | 3.6 Hz | **65 %** | **44 %** | 0.1 Hz | **0.70 rad** | 1.32 m |
| **CPG** (hardware-proven) | 1.7 Hz | 5 % | 1 % | 1.7 Hz | 2.13 rad | 3.36 m |

The DA policy pumps most of its command power **above 5 Hz**. The motor can't follow it
(dominant response ≈ 0.1 Hz), so the tail just **flutters tiny and fast** — yet the fish still
moves 1.3 m **with almost no visible tail motion**. That is the velocity²-inflated fluid turning
fast micro-flicks into thrust. It is physically unrealistic and **will not transfer to hardware**.

### Why the earlier fixes didn't catch it
- **Motor re-ID** caps *velocity*, not *frequency* → small-amplitude/high-freq stays under the cap.
- **Fluid re-ID (refitB)** forbids *backward* swimming, but it is still a stateless velocity²
  model tuned ≤4 Hz → it still over-produces force for fast small motions.

### The fix
Low-pass filter the **motor setpoint** so the *commanded* motion stays inside the band where the
fluid model is valid (and where the real fish / CPG actually operate, ≤ ~4 Hz):

- **1st-order EMA on the setpoint, cutoff = 4.5 Hz** (`ã_t = (1−α)ã_{t-1} + α a_t`, `α ≈ 0.361`),
  re-enabled in `rl/environments/da/env_sofi_da.py` (`lpf_cutoff_hz`; default 0 = off).
- **1st-order, not 2nd:** the motor already contributes a physical 2nd-order low-pass; stacking a
  2nd-order would over-damp/over-lag. The combined response is ~3rd-order.
- **4.5 Hz:** passes the ≤4 Hz CPG-band swimming (~0.75 gain) while cutting the exploit band hard.
  Verified — at 4.5 Hz the LPF drops setpoint power **>8 Hz from 46 % → 3 %** (and >5 Hz 69 % → 23 %).

### Validation in progress
Two from-scratch sweeps (6 runs each, both with the 4.5 Hz LPF + re-ID motor + 10-stage curriculum)
to see whether LPF-constrained policies swim realistically, and which fluid generalizes:

| Sweep | Env | Fluid |
|---|---|---|
| `rl/sweepconfigs/sweep_da_lpf_refitB.yml` | `sofi_full_curriculum_da` | refitB_fwdonly |
| `rl/sweepconfigs/sweep_da_lpf_e13a.yml` | `sofi_full_curriculum_da_e13a` | E13a |

**Hypothesis:** with the setpoint constrained to ≤4.5 Hz, the policy can no longer fabricate
thrust from high-frequency flicks, so it must learn realistic low-frequency, large-amplitude
strokes — the regime where the fluid coefficients were actually tuned — and should transfer
better. The A/B tells us whether the inflated-but-anti-backward refitB or the original E13a fluid
behaves better under the frequency cap.

---

## 4. The deeper lesson

This is a **model-validity** problem, not a tuning bug: a stateless fluid model fit on a narrow
frequency band will *always* be exploitable outside that band by an RL policy free to command
arbitrary frequencies. The principled defenses are (a) **constrain the policy to the validated
regime** (the setpoint LPF — done here), and/or (b) **a fluid model with state/unsteady terms**
(future work). Until (b), keep the commanded motion where the data is.

---

## 5. Backward-swimming mechanism + the 2nd-order fix (2026-06-26)

The 1st-order 4.5 Hz LPF (§3) trained good policies (refitB → stage 8, 92% @ stage-8 DR) but
they **still swim backward** under DR. We isolated the mechanism with
`rl/eval/analyze_backward_mechanism.py` (50 ep/policy, DR at each policy's reached stage,
capturing motor command, *actual* motor angle/velocity, passive-tail deflection, fwd/bwd velocity).

### The mechanism — backward IS the residual exploit
Comparing DA-refitB's **backward** vs its own **forward** episodes:

| DA-refitB (DR) | motor amp p2p | tail-beat | **motor-VELOCITY power >4.5 Hz** | raw action >4.5 Hz | post-LPF cmd >4.5 Hz | swim speed |
|---|---|---|---|---|---|---|
| **Backward** (10/50) | 0.82 rad | 0.69 Hz | **50%** | 64% | 28% | −0.059 |
| Forward (36/50) | 1.41 rad | 1.10 Hz | **7%** | 16% | 4% | +0.159 |

Backward swimming = **fast, small-amplitude flutter**: half the energy of the motor *velocity* sits
above 4.5 Hz (vs 7% when swimming forward), and since fluid force ∝ velocity², that rectifies into
reverse thrust. The 1st-order LPF (−3 dB at 4.5 Hz, gentle 6 dB/oct) is **too soft** to remove it.
Forward DA, by contrast, is a clean CPG-like gait (~1.1 Hz, large amplitude).

### Why CPG is immune
CPG's 6-D action drives a smooth oscillator whose frequency is **hard-capped at OMEGA_MAX = 8π =
4 Hz** — it *structurally cannot* command a >4.5 Hz flutter. Its rare backward episode (1/50) is a
deliberate **large-amplitude, low-freq reversed traveling wave** (1.58 rad, 0.32 Hz), not flutter.
This is also **why DA is so much harder to train than CPG**: same plant (1 driven `motor_0` + 5
passive tail segments), but DA's raw 50 Hz setpoint admits arbitrary high frequencies (bounded only
by a soft LPF), so the reverse-thrust flutter is an easy local optimum; CPG's action space is
band-limited and shaped around realistic gaits, so the exploit is unreachable.

### The fix — 2nd-order Butterworth, the *velocity-transfer* argument
The right lens is not the amplitude response but the **velocity transfer** `g(f)·f` (∝ tail velocity
per unit command, the quantity the fluid squares). Measured on the env filters (fs = 50 Hz):

| filter | 1 Hz | 2 | 3 | 4 | 4.5 | 6 | 8 | 10 | >8 Hz power |
|---|---|---|---|---|---|---|---|---|---|
| **gain** 1st @4.5 | 0.97 | 0.87 | 0.77 | 0.67 | 0.63 | 0.52 | 0.42 | 0.36 | 0.180 |
| **gain** 2nd Butter @3.5 | 1.00 | 0.95 | 0.81 | 0.60 | 0.51 | 0.30 | 0.16 | 0.09 | **0.027** |
| **vel `g·f`** 1st @4.5 | 1.0 | 1.7 | 2.3 | 2.7 | 2.8 | 3.1 | **3.4** | **3.6** | — |
| **vel `g·f`** 2nd @3.5 | 1.0 | 1.9 | 2.4 | 2.4 | 2.3 | 1.8 | 1.3 | 0.9 | — |

The 1st-order filter's velocity transfer **keeps rising with frequency** (peaks at the top of the
band) → RL is *rewarded* for high-frequency flutter. The 2nd-order Butterworth @3.5 Hz **peaks at
3–4 Hz then rolls off** → the most velocity-productive motion is now the realistic 1–4 Hz stroke, and
flutter is no longer the path of least resistance. It also preserves the 1–3 Hz swim band *better*
than the old filter (flat passband) while crushing >8 Hz power **6.8×**. Cutoff 3.5 Hz keeps 4 Hz
"possible" (0.60) per the design target (1–3 Hz fully, ≤4 Hz allowed, >4.5 Hz crushed).

**Implementation:** `rl/environments/da/env_sofi_da.py` — `lpf_order` (1 = EMA, 2 = RBJ Butterworth
biquad, Q = 1/√2); plumbed through `train_rl_curriculum.py` and all eval paths
(`eval_reachability`, `eval_compare`, `fetch_and_test`/`test_RLtraj`) so evals stay faithful.

**Final sweep:** `rl/sweepconfigs/sweep_da_lpf2_refitB.yml` (`name: sofi_da_lpf2_refitB`) — refitB
fluid + `lpf_cutoff_hz: 3.5`, `lpf_order: 2`, everything else = the proven DA recipe. If backward
still survives, the next lever is a small `alpha_backward` reward penalty (currently 0); the
structural alternative is a CPG-style band-limited (oscillator) DA action.
