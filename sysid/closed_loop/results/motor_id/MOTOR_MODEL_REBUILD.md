# Motor-model rebuild + new RL sweep (sim2real)

Branch `feat/sim2real-gap`. Owner: Simon. Started 2026-06-12/13.

## 1. What changed and why

The old motor model was a patchwork: action LPF (4–8 Hz EMA) + gear-compliance EMA
(3.8 Hz) + MuJoCo position actuator (kp=5, kv=0.40, armature=0.001) + tendon. The
baseline (`fluidID/scripts/22_motor_baseline.py`, `fluidID/results/motor_baseline/`)
proved:
- the **gear-compliance EMA is strictly worse than no filter at every frequency** —
  it over-damped the whole band to suppress one 4 Hz exploit (sledgehammer);
- the **action LPF** is redundant once the actuator bandwidth + dead time are correct,
  and its second justification (protect fluid from HF) is moot now that fluid is
  re-identified on RL-deployment data;
- the actuator was too sluggish at low/mid freq (overdamped kp/kv = 2 Hz pole);
- the real phase is ~flat ~50 ms = a pure transport delay of ≈2 policy steps.

**Decision (physically honest model):** keep the MuJoCo position actuator (it is the
right structure — a finite-stiffness torque source against the tail load reproduces the
~9 % static droop a kinematic lag cannot), REMOVE the LPF and gear EMA, RE-DERIVE the
armature (= reflected rotor inertia I_rotor·200², it sets bandwidth), and ADD an explicit
transport delay in the env/driver (MuJoCo has no native input delay). The "transfer
function" is implicit in kp/kv/armature/forcerange + dead time; no separate TF block (a
linear TF cannot represent the velocity saturation, which is real and load-bearing).

## 2. Identification (`fluidID/scripts/23_motor_id.py`)

- Data: RL-deployment runs 6,13,14,15,25,26,27,30,32 (command = raw `motor_setpoint`
  ZOH@50 Hz on the canonical 1 kHz grid via `convert.py`'s clock map; `motor_zero`
  applied to BOTH command and measured — else a phantom DC error). Each part is an
  independent open-loop segment (p1 from rest, p2 warm-started); never across the gap.
- **Field runs added to the COMBINED loss** (run_3/4/7/10/13) so the high-amplitude
  saturated 3–4 Hz band (backward-exploit-relevant) is constrained — the RL deploy data
  came from an LPF-trained policy and under-excites it. The April field tail may be
  slightly thinner/bendier than the May RL tail, so the combined fit lands between the
  two states and DR covers the spread.
- Fit fluid-OFF (6× faster, ~6 % coupled), validate fluid-ON. forcerange fixed ±2.3.
  v_max=forcerange/kv is realistic: measured peak 5.47, 99.5th pct 4.89 rad/s (the 5.75
  datasheet is no-load); fitted kv sits inside the existing kv DR [0.38, 0.51].

### Parameter sets (for the record)
| set | kp | kv | armature | T_d | RL pos/vel RMSE | field 4 Hz gain (real 0.345) |
|---|---|---|---|---|---|---|
| old | 5.0 | 0.40 | 0.001 | 0 | 2.63° / 1.53 | 0.391 |
| RL-only | 6.91 | 0.468 | 0.0105 | 42 ms | 2.92 / 0.32 | 0.518 (over-passes) |
| combined (equal) | 3.77 | 0.395 | 0.0068 | 2 ms | 2.66 / 1.13 | 0.331 |
| **RL-primary 3:1 (ADOPTED)** | **4.09** | **0.340** | **0.0044** | **28 ms** | **2.49 / 0.79** | **0.398 ✓** |

ADOPTED: RL-primary (RL weighted 3:1, field = soft 4 Hz cap), since RL is the deployment
distribution. Best RL tracking (2.49°/0.79) with field 4 Hz controlled (0.398 ≈ old 0.391,
no over-pass). v_max=6.77 (kv DR spans hardware 5.75). Dead time: total cmd→meas lag is REAL
and consistent (xcorr: ~73 ms RL, ~60 ms field; co-logged → not a clock artifact); the
delay-vs-pole split is ambiguous but RL velocity needs the delay → model ~28 ms, DR 20–40 ms.
In sofi.py + SOFI.xml. Saved: `fitted_params_rlprimary.json` (adopted), plus
`_combined.json` and `_free_kv.json` on record.

## 3. Code changes (DONE)
- `geometry/sofi.py`: motor_kp/kv/armature in SYSTEMPARAMETERS (+ header rewritten);
  motor joint armature now `{args['motor_armature']}`. `SOFI.xml` regenerated.
- `rl/environments/env_sofi_da.py`: LPF + gear EMA REMOVED; substep dead-time ring buffer
  added (`motor_deadtime_steps`); motor_history stores raw action; docstring updated.
  lpf/gear kwargs accepted-but-ignored (deprecation warning) for old eval/render scripts.
- `rl/train_rl_curriculum.py`: `--motor_deadtime_steps` plumbed into env + eval_env kwargs.
- Parity verified: env motor output == standalone fit to 0.0037°.

NOTE: in the curriculum, kp/kv are overridden per-episode by `rl/models/physics_model.py`
DR (`motor_kp_range`/`motor_kv_range`), so the XML kp/kv only bind for non-DR scripts
(fluid-ID/replay/eval). armature is NOT randomized → the XML armature binds in training.

## 4. New sweep plan — DA-MotorID-S01

Going back to **Direct Action** (post-S28), dropping the SDA/LPF detour (sda_s01–s05),
now with the re-identified motor model + re-identified fluid coeffs. Fresh from-scratch SAC.

- **Name:** `DA-MotorID-S01`   **Config:** `rl/sweepconfigs/sweep_da_motorid_s01.yml`
- **Curriculum:** same 10-stage structure as SDA-S02 (env `sofi_sda_curriculum_v1` →
  `env_sofi_da_lpf_prog_curriculum_v1`), reward = distance + success (CPG-style).
- **Changes vs sweep_sda_s02.yml:**
  - DROP `lpf_cutoff_hz` (the env now has no LPF; the curriculum never ramped it).
  - ADD `motor_deadtime_steps: <combined T_d in steps>` (fixed).
  - `motor_ctrlrange: 1.0` (match XML ctrlrange and the ID; S02 used 1.2 which the
    actuator clipped).
  - Recenter motor DR on the identified values — see integration checklist.

### Integration checklist (before launch)
1. Fill combined-fit params into `sofi.py`; regenerate `SOFI.xml`; re-run
   `23_motor_id.py --no-fit`(parity) + `22_motor_baseline.py 1` (field Bode within tol).
2. Recenter DR: edit the curriculum env's `motor_kp_range`/`motor_kv_range` (and the
   `physics_model.py` nominal defaults kp=5.0,kv=0.30 → identified) around the new values.
   Keep widths for sim2real robustness (the field↔RL tail spread). Decide armature/
   dead-time DR (default: keep both fixed for S01; revisit in S02).
3. `motor_deadtime_steps` set in the sweep config and confirmed reaching the env.
4. Ensure the latest re-ID fluid coeffs are in `sofi.py` before regenerating XML.

### Commands (after committing + pushing)
```bash
# create the sweep (run where W&B is authed — a lab machine, not the laptop)
python batching/_create_sweep.py rl/sweepconfigs/sweep_da_motorid_s01.yml
#   → prints  SWEEPID:srl_ethz/fishsim/<id>

# launch on all machines: cancels existing jobs + git pull + submit (so PUSH first)
bash batching/sweep_launch.sh --sweepId srl_ethz/fishsim/<id>
```
