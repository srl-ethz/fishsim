# Fluid Coefficient Log

A record of notable coefficient sets found during system identification (Step A).
Use this to revisit past results before overwriting `field2sim.yml`.

The `coefficients` vector has length 5, 10, or 15 depending on segmentation mode (see `field2sim.yml`).

---

## Model Configuration Reference

Each entry was trained and must be **evaluated** with a specific mass model and `hingeDamping` setting.
Using the wrong config will change simulation physics and produce inconsistent results.

| Entry | Mass model | `hingeDamping` | `eval_sysid.py` flag |
|-------|-----------|----------------|----------------------|
| E1–E8 | Old: head≈1 g, body=600 g (both shells), COM=[0.128, 0, 0.003] | 0.1 | `--legacy_mass` |
| E9–E10 | Corrected: head=750 g, plate=60 g, body=400 g, COM=[0.0843, 0, 0.0088] | 0.1 | *(none)* |
| E11+ | Corrected (same as E9–E10) | **0.0** | *(none)* |
| E13+ | SW-derived explicit inertial: 830g total (head 580g+body 250g), COM=[−7.97, +1.33, +0.17] mm MuJoCo, Ixx=0.00155 kg·m²; per-joint k/d from R4; motor child ≈0g | per-joint (structural block) | *(none, current model)* |

**To switch configs before running `eval_sysid.py`:**
- `hingeDamping` is in `Geometry/sofi_twin.py` → `SYSTEMPARAMETERS["hingeDamping"]`
- Mass model is controlled by the `--legacy_mass` flag (eval script reads it and patches the XML at runtime)
- After evaluating an older entry, restore `hingeDamping` to the value for the current active experiment

---

## History in one table (E1 to E11, the dead ends before E13)

The full log of every attempt, with per-run loss tables and the physics analysis of each, is kept privately
(archive `05_fluid_identification/`). What matters for the model is the sequence of decisions:

| Entry | Setup | Lesson |
|---|---|---|
| E1 | 15 coefficients on 6 runs, no bounds | overfits: coefficients run to the box edges |
| E2 to E4 | 10 coefficients (head + tail, no separate fin), CMA-ES on 5 or 6 runs; yaw and oscillation terms in the loss tried | yaw weight hurts the path error; a random start avoids the local minimum of the old scalar fluid |
| E5, E6 | start transient weighted 4×, yaw weight 0 | the start-up phase is where added mass shows; weighting it fixes the early path |
| E7 | run 9 (a turn) added to the fit set, bounds widened | one turning run is needed, otherwise the fin lift is unconstrained |
| E8 | back to 15 coefficients (separate fin) on the 6-run set | the fin needs its own coefficients; path error drops |
| E9, E10 | corrected mass model (head 750 g, plate 60 g, body 400 g), sinusoid weight on the tail | mass error had been absorbed by the fluid terms |
| E11 | hinge damping 0, fin angular-drag bound kept | damping belongs to the measured hinges, not to a free parameter |
| E13 | explicit CAD inertia, per-joint stiffness and damping from the tail benchmark (below) | the model used by the deployed policy |

## E13 — 15 coefficients, corrected mass + per-joint stiffness model (2026-04-28)

**Status:** 3 runs completed/recovered (PC crash). Best: E13a (loss=0.040059).

### Model changes from E11 (last documented entry)
- Head mass: 750g → 520g (SolidWorks 2026-04); head+plate total: 580g unchanged
- Body mass: 400g → 194g (body excluding motor child ~56g); body total: 250g unchanged
- Motor visual geom masses: 0.001g → 0 (visual-only; mass lives in motor child body)
- Per-joint stiffness: scalar 0.65 Nm/rad → [k0..k4] from R4 air benchmark (N_TRANSIENT_CYCLES=1.0)
  - R4 values: hinge_stiffness = [2.991, 2.393, 2.125, 1.768, 1.324] Nm/rad (joint0→joint4)
- Per-joint damping: scalar 0.10 Nm·s/rad → [c0..c4] proportional to stiffness from R4
  - R4 values: hinge_damping = [0.034, 0.0272, 0.0241, 0.0201, 0.015] Nm·s/rad
- Explicit `<inertial>` from SolidWorks 2026-04-29: 830g (head 580g + body 250g), COM=[−7.97, +1.33, +0.17] mm; Ixx=0.00155, Iyy=0.00276, Izz=0.00172 kg·m² — replaces auto-computed hollow-mesh COM (+77.4mm → −7.97mm, 85mm correction); geom masses zeroed to prevent double-counting
- _PHYS_BOUNDS_15 c12 fin angular drag: [2.00, 40.0] → [2.00, 20.0] (was raised for E11 because old stiffness 0.65 Nm/rad used c_angular as synthetic spring; now with 4–5× stiffer joints this should not be needed)

### MuJoCo structural validation (step3_mujoco_validation.py)

**Sub-test A (air):** k_eff_mujoco shows ~19.4% error vs the analytical k_target from R4.

Root cause: `k_eff_from_joints()` uses the analytical formula `r² / Σ(d_i²/k_i)` with geometric
perpendicular moment arms (TENDON_MOMENT_ARMS). MuJoCo computes the tendon Jacobian ∂L/∂q from
the actual 3D tendon routing path, which differs from the analytical d_i. This is a systematic
structural offset, not a model error — the per-joint stiffness values come from real hardware (R4).

Impact on E13: MuJoCo spring forces are ~19% stiffer/softer than the physical fish. The fluid
optimizer sees a stiffer structural model; it may inflate or deflate angular drag coefficients
slightly to compensate. If this matters for E14+, re-fit per-joint stiffness by scaling all
hinge_stiffness values by (k_eff_R4 / k_eff_mujoco_static) to make MuJoCo match hardware exactly.

**Sub-test B (water):** physically plausible shift (added mass increases I_eff, drag increases Im(H)).

### Run command (all three runs used identical flags)

```bash
python digital_twin/orchestrators/opt_field2sim.py \
    ../fishdeploy/data/01_field_openloop_20260401/run_03_1.0Hz_9.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_04_2.0Hz_9.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_07_3.0Hz_7.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_10_4.0Hz_7.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_11_2.0Hz_12.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_09_2.0Hz_6.0mm_-3.0mm \
    --n_coef 15 --optimizer cma --physics_bounds --iter 1000 \
    --start_weight 4.0 --random_start --sinusoid_weight 0.4
```

Training runs: same 6 runs as E9/E10/E11 (run\_3, run\_4, run\_7, run\_10, run\_11, run\_9).

### Results

| Run | Loss | Evals | Status | File |
|-----|------|-------|--------|------|
| E13a | **0.040059** | 2028 | Checkpoint (recovered) | `checkpoint_20260501_044853.yml` |
| E13b | 0.056795 | 792 | Completed normally | `opt_result_20260501_175208.yml` |
| E13c | 0.065674 | 516 | Checkpoint (recovered, not converged) | `checkpoint_20260502_015522.yml` |

**Best: E13a** (`checkpoint_20260501_044853.yml`, loss=0.040059).

### E13a — best result (loss=0.040059, 2028 evals)

**File:** `Outputs/sysid/checkpoint_20260501_044853.yml`

**Coefficients (n\_coef=15):**
```
[0.005822, 0.044600, 0.812577, 0.521112, 0.902770,    ← head
 0.039474, 0.551234, 5.427544, 0.000956, 1.581176,    ← tail body
 0.170727, 0.003855, 18.310332, 10.148229, 5.951302]  ← fin
```

**Assessment:** Pending visual validation. Best E13 result and better than E11 (0.052981). Slightly worse than E10 (0.031446) but E10 used old scalar stiffness — not directly comparable.

### E13b — completed run (loss=0.056795, 792 evals)

**File:** `Outputs/sysid/opt_result_20260501_175208.yml`

**Coefficients (n\_coef=15):**
```
[0.001424, 0.158581, 0.163296, 0.472895, 0.976274,    ← head
 0.032966, 3.491412, 3.977277, 0.426587, 0.496284,    ← tail body
 0.158538, 0.028832, 19.953507, 7.951760, 4.075066]   ← fin
```

**Note on c6 (tail slender):** 3.49 — the only E13 run where this is nonzero. All prior entries (E5–E11) had c6≈0; this is an outlier worth noting but likely a basin artifact since E13a and E13c both found c6≈0.5 independently. Pairwise L2 distance to E13a = 4.86, to E13c = 8.07 — these are genuinely different random-start basins, not the same run stopped early.

**Assessment:** Written to `field2sim.yml` automatically on completion (before crash). Pending visual validation.

### E13c — not converged, to be continued (loss=0.065674, 516 evals)

**File:** `Outputs/sysid/checkpoint_20260502_015522.yml`

**Coefficients at crash (n\_coef=15):**
```
[0.000438, 0.454930, 0.460862, 0.818786, 0.999778,    ← head
 0.193199, 0.496315, 3.009333, 0.389980, 4.707513,    ← tail body
 0.042392, 0.020134, 15.432589, 11.737680, 5.592467]  ← fin
```

**Note:** c4 (head magnus) = 1.000, exactly at bound ceiling [0, 1.0]. Only 516 evals / ~43 CMA-ES generations — far from convergence. **Resume command** (warm-start from best checkpoint coef, no `--random_start` so x0 = best known point; CMA-ES state is not persisted so this is a warm restart, not true continuation):

```bash
python digital_twin/orchestrators/opt_field2sim.py \
    ../fishdeploy/data/01_field_openloop_20260401/run_03_1.0Hz_9.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_04_2.0Hz_9.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_07_3.0Hz_7.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_10_4.0Hz_7.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_11_2.0Hz_12.0mm \
    ../fishdeploy/data/01_field_openloop_20260401/run_09_2.0Hz_6.0mm_-3.0mm \
    --n_coef 15 --optimizer cma --physics_bounds --iter 1000 \
    --start_weight 4.0 --sinusoid_weight 0.4 \
    --warm_start Outputs/sysid/checkpoint_20260502_015522.yml
```

**Assessment:** Pending. c4 at ceiling suggests this basin wants more head magnus — expanding the bound to [0, 1.5] might be worth testing in E14.

### Cross-run coefficient comparison (E13)

| # | Segment | Meaning | E13a (0.040) | E13b (0.057) | E13c (0.066) | Note |
|---|---------|---------|-------------|-------------|-------------|------|
| c0 | head | blunt drag | 0.0058 | 0.0014 | 0.0004 | All near zero — consistent |
| c1 | head | slender | 0.0446 | 0.1586 | 0.4549 | Spread — no consensus |
| c2 | head | angular | 0.8126 | 0.1633 | 0.4609 | Spread |
| c3 | head | kutta | 0.5211 | 0.4729 | 0.8188 | Mid-range all, broadly consistent |
| c4 | head | magnus | 0.9028 | 0.9763 | **1.000** | ⚠️ All near/at ceiling [0,1] — bound must expand for E14 |
| c5 | tail | blunt | 0.0395 | 0.0330 | 0.1932 | Low, consistent |
| c6 | tail | slender | 0.5512 | **3.491** | 0.4963 | ⚠️ E13b outlier — prior entries all ~0; likely spurious basin |
| c7 | tail | angular | 5.4275 | 3.9773 | 3.0093 | Lower than E10/E11 — per-joint stiffness taking over some damping |
| c8 | tail | kutta | 0.0010 | 0.4266 | 0.3900 | ~zero in E13a, small in b/c |
| c9 | tail | magnus | 1.5812 | 0.4963 | 4.7075 | Large spread |
| c10 | fin | blunt | 0.1707 | 0.1585 | 0.0424 | Small |
| c11 | fin | slender | 0.0039 | 0.0288 | 0.0201 | Near zero — consistent |
| c12 | fin | angular | 18.3103 | **19.955** | 15.4326 | High but below ceiling (20) in a/c; E13b at ceiling |
| c13 | fin | kutta | 10.1482 | 7.9518 | 11.7377 | High — fin as main thrust driver confirmed |
| c14 | fin | magnus | 5.9513 | 4.0751 | 5.5925 | Moderate |

**Key cross-run findings:**
- **c4 (head magnus) at/near ceiling in all three runs** — bound [0, 1.0] is too tight; raise to [0, 1.5] for E14
- **c13 (fin kutta) consistently high (8–12)** — fin-as-propulsor interpretation from E8 onward is robustly confirmed
- **c12 (fin angular drag) consistently high (15–20)** — staying near bound; consider raising ceiling again
- **c7 (tail angular drag) dropped from E10/E11 (6.8–19.2) to 3–5.4** — per-joint structural damping is absorbing what fluid angular drag was previously compensating for; physically correct direction
- **c6 (tail slender)**: E13b is an outlier (3.49); E13a and c confirm ~zero — prior finding holds
