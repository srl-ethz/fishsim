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
- `hingeDamping` is in `Geometry/auto_SOFI_field2sim.py` → `SYSTEMPARAMETERS["hingeDamping"]`
- Mass model is controlled by the `--legacy_mass` flag (eval script reads it and patches the XML at runtime)
- After evaluating an older entry, restore `hingeDamping` to the value for the current active experiment

---

## Entry 1 — First 15-segment result (overfitted)

**Coefficients:**
```
[0.02, 0.02, 0, 5.74255, 0.02, 0.02, 1.00489, 3.09255, 7, 1.58535, 0.02, 7, 7, 1.37136, 0.195883]
```

**Optimizer result:** best loss = 0.4906 (57% through 500-eval run, stopped at eval 284)

**Files:** `Outputs/sysid/opt_result_20260409_122613.yml` · `Outputs/sysid/convergence_20260409_122613.png`
*(parallel duplicates with identical coef/loss: `opt_result_20260411_170027.yml`, `opt_result_20260411_230734.yml`)*

**Assessment:** Overfitted. Produces really weird tail behaviour and is sensitive (in a bad way) to initial conditions and small changes in motor commands. Do not use as a starting point for further tuning.

**Data used:** `run_3_1.0Hz_9.0mm` (20260401 first swim)

---

## Entry 2 — 10-segment result (6-run CMA-ES, no yaw in loss)

**Coefficients (n_coef=10):**
```
[0.0068, 0.4856, 1.0654, 0.0000, 0.0133, 0.0000, 0.5529, 6.9838, 4.7427, 2.9879]
```

**Optimizer:** CMA-ES, σ₀=2.0, physics_bounds, 1000 iter (stopped early at gen ~99)

**Loss function:** Mean over 6 training runs of `pos_weight × MSE(XY position) + yaw_weight × MSE(yaw)` — **yaw weight = 0** (position-only loss)

**Best loss:** 0.6506

**Files:** `Outputs/sysid/opt_result_20260421_122550.yml` · `Outputs/sysid/convergence_20260421_122550.png`

**Training runs:** run_1 (1.0Hz 5mm), run_3 (1.0Hz 9mm), run_4 (2.0Hz 9mm), run_7 (3.0Hz 7mm), run_10 (4.0Hz 7mm), run_11 (2.0Hz 12mm) — 20260401 first swim

**Validation on run_3 replay:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260421_123318\replay.mp4`

**Assessment:** Reasonable first result but two clear issues visible in run_3 replay:
- Yaw oscillation amplitude is **larger** than the real fish
- Forward speed is roughly **half** of the real fish
Likely caused by zero yaw weight in the loss — the optimizer never penalized heading error, so yaw dynamics were ignored. Also run_7 (3.0Hz) had a persistent ~97° heading offset that pulled coefficients toward low-thrust values — see run_7 bug note below.

> **run_7 preprocessing bug (now fixed):** Time-zero was anchored to the first AprilTag detection, not the motor start. For run_7 the tag was obscured for the first 0.479 s, so the pipeline clipped all IMU data before that point and computed `motor_center = df_imu['Actual'].iloc[0]` mid-swim (tick 2634) instead of at the true neutral (tick 2418). The 216-tick difference is 0.33 rad, baked in as a constant DC offset on `Ghost_Alpha` for the entire run — causing the tail to beat asymmetrically and the twin fish to continuously turn. All other runs had the tag visible from the start, so the bug went unnoticed.

---

## Entry 3 — 10-segment result (5-run CMA-ES, fixed run_7, yaw_weight=0.3)

**Coefficients (n_coef=10):**
```
[0.00232927, 0.586052, 1.84695, 4.09335e-06, 0.0396167, 0.00310839, 0.0545748, 5.39232, 7.90145, 1.47895]
```

**Optimizer:** CMA-ES, σ₀=2.0, physics_bounds, yaw_weight=0.3, oscillation_weight=0.0, stopped early

**Best loss:** 0.2569

**Files:** `Outputs/sysid/opt_result_20260421_163408.yml` · `Outputs/sysid/convergence_20260421_163408.png`
*(file records loss=0.121512 — position-only component; 0.2569 is the combined pos+0.3×yaw as logged above)*

**Training runs:** run_3 (1.0Hz 9mm), run_4 (2.0Hz 9mm), run_7 (3.0Hz 7mm, fixed), run_10 (4.0Hz 7mm), run_11 (2.0Hz 12mm) — 20260401 first swim

**Validation on run_3 replay:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260421_164036\replay.mp4`

**Assessment:** Better forward speed than Entry 2. However: weird backward swimming at the start of the run, and completely fails for turns. Yaw oscillation amplitude of the twin is still larger than the real fish. Do not use for runs with asymmetric actuation or curved trajectories. Backward swimming likely caused by startup transient (tail kicks backward before momentum builds). Oscillation issue carries over because yaw loss alone doesn't penalize oscillation amplitude.

---

## Entry 4 — 10-segment result (5-run CMA-ES, osc_weight=0.4, random_start)

**Coefficients (n_coef=10):**
```
[0.0557, 0.3677, 2.2449, 0.1524, 0.0045, 0.0871, 0.0014, 6.7730, 10.5067, 2.4264]
```

**Optimizer:** CMA-ES, σ₀=2.0, physics_bounds, random_start, 1000 iter (completed)

**Files:** `Outputs/sysid/opt_result_20260421_225224.yml` · `Outputs/sysid/convergence_20260421_225224.png`

**Loss:** 0.490096 (pos + 0.3×yaw + 0.4×osc; not directly comparable to Entries 2–3 due to osc term)

**Training runs:** run_3 (1.0Hz 9mm), run_4 (2.0Hz 9mm), run_7 (3.0Hz 7mm, fixed), run_10 (4.0Hz 7mm), run_11 (2.0Hz 12mm) — 20260401 first swim

**Validation video:** (none yet — no qualitative assessment)

**Assessment:** Not yet evaluated visually. Notable: c8 (tail kutta) = 10.51, the highest of any result so far — likely still high-thrust / high-oscillation. Superseded by Entry 5 which uses time-weighted position loss (cleaner loss formulation).

---

## Entry 5 — 10-segment result (5-run CMA-ES, yaw_weight=0.3, start_weight=4.0, random_start)

**Coefficients (n_coef=10):**
```
[0.000116431, 0.610869, 2.85852, 7.70909e-06, 4.21869e-07, 0.00238181, 0.000512476, 12.0, 9.5582, 3.99441]
```

**Optimizer:** CMA-ES, physics_bounds, random_start, stopped at iter 181/500 (36%), σ=0.105

**Files:** `Outputs/sysid/opt_result_20260422_103649.yml` · `Outputs/sysid/convergence_20260422_103649.png`

**Loss function:** `mean_over_runs(wpos_mse + 0.3×yaw_mse)` with `start_weight=4.0` time-weighting. **Not directly comparable to Entries 2–4.**

**Best loss:** 0.079827 (composite). Per-run breakdown at best (wpos | yaw×0.3): `0.018+0.007 | 0.045+0.005 | 0.020+0.009 | 0.040+0.003 | 0.135+0.117`

**Training runs:** run_3 (1.0Hz 9mm), run_4 (2.0Hz 9mm), run_7 (3.0Hz 7mm, fixed), run_10 (4.0Hz 7mm), run_11 (2.0Hz 12mm) — 20260401 first swim

**Validation video:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260422_105052\replay.mp4`

**Assessment:** Validated on run_3 (1.0Hz 9mm, straight swimming):
- ✓ No backward swimming at startup (fixed vs Entry 3)
- ✓ Good position tracking especially at startup
- ~ Loses position slightly in steady-state (longer into run) — expected given start_weight=4.0 time-weighting
- ✗ Yaw oscillation amplitude still larger than real fish
- Turn quality: pending

### Physics analysis

| # | Meaning | Bounds | E2 | E3 | E4 | **E5** | Flag |
|---|---------|--------|-----|-----|-----|--------|------|
| c0 | head blunt drag | [0.00, 0.15] | 0.007 | 0.002 | 0.056 | **0.000** | OK — streamlined head |
| c1 | head slender (added mass) | [0.00, 2.0] | 0.486 | 0.586 | 0.368 | **0.611** | OK — consistent across entries |
| c2 | head angular drag | [1.00, 5.0] | 1.065 | 1.847 | 2.245 | **2.859** | OK — trending up, physically plausible |
| c3 | head kutta (lift) | [0.00, 1.0] | 0.000 | ~0.0 | 0.152 | **~0.000** | OK — head produces negligible lift |
| c4 | head magnus | [0.00, 0.5] | 0.013 | 0.040 | 0.005 | **~0.000** | OK — head doesn't spin |
| c5 | tail blunt drag | [0.00, 0.2] | 0.000 | 0.003 | 0.087 | **0.002** | OK |
| c6 | tail slender (added mass) | [0.00, 5.0] | 0.553 | 0.055 | 0.001 | **0.001** | ⚠️ Collapsed to ~zero across E3–E5; unphysical for a flexible tail fin — lower bound raised to 0.1 |
| c7 | tail angular drag | [2.00, **12.0**] | 6.98 | 5.39 | 6.77 | **12.00** | ⚠️ At upper bound — upper raised to 20.0 |
| c8 | tail kutta (thrust) | [0.00, 12.0] | 4.74 | 7.90 | 10.51 | **9.56** | OK — high but within range |
| c9 | tail magnus | [0.00, **4.0**] | 2.99 | 1.48 | 2.43 | **3.994** | ⚠️ At upper bound — upper raised to 6.0 |

**Bounds updated for next run:**
- c3 (head kutta): `[0.0, 1.0]` → `[0.0, 2.0]` — head acts as a lifting surface during turns; straight-swimming loss never explored this, ceiling may be too tight
- c6 (tail slender): lower bound kept at `[0.0, 5.0]` — two independent runs (E5, E6) converged to c6≈0 from different loss functions; this is likely a real signal, not degeneracy. Slender body theory applies to elongated bodies, not thin planar fins, so low c6 may be physically correct. Forcing a lower bound risks artificially constraining the search — if turning data (run_9) genuinely requires higher c6, the optimizer will find it without being forced.
- c7 (tail angular drag): `[2.0, 12.0]` → `[2.0, 20.0]` — was pinned at ceiling; true value likely higher
- c9 (tail magnus): `[0.0, 4.0]` → `[0.0, 6.0]` — was pinned at ceiling

**Turning validation failure (run_9_2.0Hz_6.0mm_-3.0mm):** Fish swims backwards instead of forwards+turning. Root cause: all 5 training runs use symmetric actuation — the fluid model was never penalized for asymmetric/turning behavior, so the coefficients don't generalize. Adjusted bounds alone will not fix this. The next optimization must include at least one turning run (run_9 is the obvious candidate) alongside the existing straight-swimming runs. Including run_9 in training is valid for SysID — physics parameters are being identified, not a generalizing ML model. A turning run with a large position error from backwards swimming will strongly push the optimizer away from those coefficient regions.

---

## Entry 6 — 10-segment result (5-run CMA-ES, yaw_weight=0.0, start_weight=4.0, random_start)

**Coefficients (n_coef=10):**
```
[0.00308294, 0.601444, 2.83827, 0.000130125, 0.00705327, 0.00130717, 0.000796469, 11.9251, 9.5796, 3.99994]
```

**Optimizer:** CMA-ES, physics_bounds, random_start, stopped early

**Files:** `Outputs/sysid/opt_result_20260422_111317.yml` · `Outputs/sysid/convergence_20260422_111317.png`

**Loss function:** `mean_over_runs(wpos_mse)` with `start_weight=4.0` time-weighting. Position-only, no yaw penalty.

**Best loss:** 0.051145 (position-only; lower than Entry 5's 0.079827 but those include yaw terms — not directly comparable)

**Training runs:** run_3 (1.0Hz 9mm), run_4 (2.0Hz 9mm), run_7 (3.0Hz 7mm, fixed), run_10 (4.0Hz 7mm), run_11 (2.0Hz 12mm) — same 5 runs as Entry 5, no turning run

**Validation video run\_3:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260422_111852\replay.mp4`

**Validation video run\_9:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260422_125058\replay.mp4` — backwards swimming, same failure as Entry 5. Confirms root cause is training data gap, not coefficients.

**Comparison with Entry 5 (same training data, different loss):**

| # | Meaning | Entry 5 (yaw=0.3) | Entry 6 (yaw=0.0) | Note |
|---|---------|-------------------|-------------------|------|
| c0 | head blunt drag | 0.000116 | 0.00308 | Similar — both near zero |
| c1 | head slender | 0.6109 | 0.6014 | Nearly identical |
| c2 | head angular drag | 2.8585 | 2.8383 | Nearly identical |
| c3 | head kutta | ~0.000 | 0.000130 | Similar — both negligible |
| c4 | head magnus | ~0.000 | 0.00705 | Similar — both negligible |
| c5 | tail blunt drag | 0.00238 | 0.00131 | Similar |
| c6 | tail slender | 0.000512 | 0.000796 | Both ~zero — below new bound of 0.3; will be forced higher next run |
| c7 | tail angular drag | 12.000 | 11.925 | Both pinned at old ceiling — confirms bound expansion to 20 is needed |
| c8 | tail kutta | 9.5582 | 9.5796 | Nearly identical |
| c9 | tail magnus | 3.99441 | 3.99994 | Both pinned at old ceiling — confirms bound expansion to 6 is needed |

**Key insight:** Two independent runs with different loss functions (one including yaw penalty) converged to nearly identical coefficients. This strongly suggests this is a robust local minimum of the position loss landscape — not noise. The new bounds (c6 ≥ 0.3, c7 ≤ 20, c9 ≤ 6) will genuinely force the next run to explore new territory rather than re-finding this same basin.

**Assessment:** run\_3 qualitatively identical to Entry 5 — consistent with near-identical coefficients, confirms both runs found the same basin. run\_9 (turning): backwards swimming, same failure as Entry 5.

---

## Entry 7 — 10-segment result (6-run CMA-ES, run\_9 added, expanded bounds, yaw\_weight=0.0, start\_weight=4.0, random\_start)

**Coefficients (n\_coef=10):**
```
[0.0403026, 0.104665, 1.15357, 0.151295, 0.366973, 9.73067e-06, 0.27068, 17.1319, 3.33268, 5.22507]
```

**Optimizer:** CMA-ES, physics\_bounds (updated), random\_start, stopped at iter 142/1000 (14%), σ=0.092

**Files:** `Outputs/sysid/opt_result_20260422_184359.yml` · `Outputs/sysid/convergence_20260422_184359.png`

**Loss function:** `mean_over_runs(wpos_mse)` with `start_weight=4.0`. Position-only, no yaw penalty.

**Best loss:** 0.053887. Per-run breakdown (run\_3 | run\_4 | run\_7 | run\_10 | run\_11 | run\_9): `0.027 | 0.063 | 0.025 | 0.011 | 0.136 | 0.062`

**Training runs:** run\_3 (1.0Hz 9mm), run\_4 (2.0Hz 9mm), run\_7 (3.0Hz 7mm, fixed), run\_10 (4.0Hz 7mm), run\_11 (2.0Hz 12mm), **run\_9 (2.0Hz 6mm/−3mm, turning)** — first entry to include a turning run

**Validation video run\_3:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260422_185424\replay.mp4`

**Validation video run\_9:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260422_191604\replay.mp4`

**Comparison with E5/E6 — structurally different solution:**

| # | Meaning | Bounds | E5/E6 | **E7** | Note |
|---|---------|--------|-------|--------|------|
| c0 | head blunt drag | [0.00, 0.15] | ~0.000 | **0.040** | First time head blunt drag is non-negligible |
| c1 | head slender | [0.00, 2.0] | ~0.606 | **0.105** | Dramatic drop — different head dynamics regime |
| c2 | head angular drag | [1.00, 5.0] | ~2.848 | **1.154** | Much lower — dropped to near lower bound |
| c3 | head kutta | [0.00, 2.0] | ~0.000 | **0.151** | First time meaningful — expanded bound + turning run activated this |
| c4 | head magnus | [0.00, 0.50] | ~0.000 | **0.367** | Near upper bound — was zero in all prior entries; ⚠️ physically questionable for a fish head |
| c5 | tail blunt drag | [0.00, 0.20] | ~0.002 | **~0.000** | Near zero |
| c6 | tail slender | [0.00, 5.0] | ~0.001 | **0.271** | Finally non-zero — turning run pulled this up |
| c7 | tail angular drag | [2.00, 20.0] | 12.000 | **17.132** | Expanded bound was essential — E5/E6 were capped at 12 |
| c8 | tail kutta | [0.00, 12.0] | ~9.580 | **3.333** | Dramatic drop — thrust mechanism has fundamentally shifted |
| c9 | tail magnus | [0.00, 6.0] | ~3.997 | **5.225** | Higher — was at old ceiling (4.0), now 5.2 with expanded bound; approaching new ceiling |

**Key insight — different physical regime:** E5/E6 used high kutta (c8≈9.6) + moderate magnus (c9≈4.0) as the thrust mechanism, with the tail angular drag pinned at its ceiling. E7 instead uses low kutta (c8=3.3) + high angular drag (c7=17.1) + high magnus (c9=5.2), with head coefficients that were completely inactive before (c3, c4) now contributing. This is a genuinely different basin, not a refinement of E5/E6 — forced there by run\_9 and the expanded bounds.

**Turning result:** run\_9 loss = **0.062** — the first solution that doesn't swim backwards on the turning run. The remaining outlier is run\_11 (2.0Hz 12mm, highest amplitude, loss=0.136), not the turning run.

**Physics flags:**
- c4 (head magnus) = 0.367 near upper bound [0, 0.5] — Magnus force requires spinning; fish heads don't spin in a meaningful way. This is likely a spurious fit compensating for something else. ⚠️
- c9 (tail magnus) = 5.22, approaching new ceiling of 6.0 — may need further expansion.
- c7 = 17.1/20.0 — not at ceiling, good.

**Assessment:** Both runs track the real fish qualitatively. run\_9 turns in the correct direction (first entry to do so) but at a larger radius than reality. Remaining issues:
- ✓ No backward swimming (run\_3)
- ✓ Turns in the correct direction (run\_9)
- ~ Turning radius too large — lateral force still insufficient
- ✗ Yaw/body oscillation amplitude still larger than real fish (persistent across all entries)
- ✗ Tail fin appears "stuck" — mid-body bends but the fin tip barely moves; consistent with c7=17.1 overdamping the entire tail+fin segment as one unit

**Root cause of stuck fin + persistent oscillation:** With 10 coefficients the tail body and caudal fin share the same fluid parameters. High angular drag (c7=17.1) is needed to damp oscillation, but it simultaneously locks the fin tip that should flutter freely to generate thrust. These are conflicting requirements that a single set of tail+fin coefficients cannot satisfy. → **Next step: switch to 15-coefficient mode** (head / tail body / fin separately).

**15-coefficient bounds updated** — the old `_PHYS_BOUNDS_15` were stale (predated all 10-coeff experiments, had non-zero lower bounds without justification, c7 tail-angular capped at 5.5 which is far too low). New bounds informed by E2–E7 findings:
- Head (c0–c4): mirrors validated 10-coeff head bounds
- Tail body (c5–c9) and fin (c10–c14): **identical bounds** — same as the validated 10-coeff tail+fin bounds. We have no empirical data about how the combined values split between tail and fin, so imposing asymmetric bounds would be an unjustified assumption that could bias the optimizer. Let the data drive the split.

---

## Entry 8 — 15-segment result (6-run CMA-ES, run\_9 included, yaw\_weight=0.0, start\_weight=4.0, random\_start)

**Coefficients (n\_coef=15):**
```
[0.00805642, 0.0543903, 2.11526, 0.204783, 0.105516,   ← head
 0.00155237, 0.00543734, 11.2755, 0.00382415, 1.57786,  ← tail body
 0.051948, 5.84617e-05, 11.7347, 6.44239, 1.16184]      ← fin
```

**Optimizer:** CMA-ES, physics\_bounds (15-coeff), random\_start

**Files:** `Outputs/sysid/opt_result_20260423_074659.yml` · `Outputs/sysid/convergence_20260423_074659.png`

**Loss function:** `mean_over_runs(wpos_mse)` with `start_weight=4.0`. Position-only, no yaw penalty.

**Best loss:** 0.039488 — **best result so far**, better than E7 (0.053887) on the same 6 training runs.

**Training runs:** run\_3 (1.0Hz 9mm), run\_4 (2.0Hz 9mm), run\_7 (3.0Hz 7mm, fixed), run\_10 (4.0Hz 7mm), run\_11 (2.0Hz 12mm), run\_9 (2.0Hz 6mm/−3mm, turning)

**Validation video run\_3:** (pending)
**Validation video run\_9:** (pending)

**Coefficient analysis vs E7 (10-coeff, same training data):**

| # | Segment | Meaning | E7 (10-coeff) | **E8 (15-coeff)** | Note |
|---|---------|---------|---------------|-------------------|------|
| c0 | head | blunt drag | 0.0403 | **0.0081** | Lower — more consistent with streamlined head |
| c1 | head | slender (added mass) | 0.1047 | **0.0544** | Further drop — head added mass collapsing toward zero |
| c2 | head | angular drag | 1.1536 | **2.1153** | Higher — back in mid-range, more physical |
| c3 | head | kutta (lift) | 0.1513 | **0.2048** | Slightly higher — head lift during turns confirmed by both E7 and E8 |
| c4 | head | magnus | 0.3670 | **0.1055** | ⬇ Much lower — E7's high value (0.367) was likely compensating for missing fin DOF; now more physical |
| c5 | tail | blunt drag | ~0.000 | **0.0016** | Near zero — consistent |
| c6 | tail | slender (added mass) | 0.2707 | **0.0054** | Collapsed again — consistent finding across all entries |
| c7 | tail | angular drag | 17.132 | **11.276** | Lower per-segment, but total tail+fin = 23.0 — optimizer wants MORE total damping than 10-coeff allowed |
| c8 | tail | kutta (thrust) | 3.333 | **~0.000** | ⚠️ Essentially zero — tail body generates no thrust |
| c9 | tail | magnus | 5.225 | **1.578** | Much lower — magnus split between tail and fin |
| c10 | fin | blunt drag | — | **0.0519** | Small but nonzero — fin leading edge |
| c11 | fin | slender (added mass) | — | **~0.000** | Near zero — consistent with tail body finding |
| c12 | fin | angular drag | — | **11.735** | Nearly identical to tail body (11.28) — balanced damping |
| c13 | fin | kutta (thrust) | — | **6.442** | ← Main thrust generator; physically correct |
| c14 | fin | magnus | — | **1.162** | Moderate |

**Key findings:**

**✓ Physically most sensible result so far.** The 15-coeff split reveals a clear functional separation that 10-coeff was forced to blur:
- **Tail body**: high angular drag (damps oscillation) + zero kutta (no thrust) — acts as a motion transmitter, not a thrust generator
- **Fin**: high angular drag + significant kutta — the actual propulsor; this is exactly the carangiform swimming model from literature

**✓ Total angular drag = 11.28 + 11.73 = 23.0** — higher than E7's 17.1 because the 10-coeff bound was [2, 20] for a single parameter. With two separate segments both allowed [2, 20], the optimizer found it needs more total damping. This explains why 10-coeff kept hitting the ceiling at 12→17.

**✓ Head magnus dropped from 0.367 → 0.105** — E7's high head magnus was a compensatory artifact. The head no longer needs to compensate for the missing fin degree of freedom.

**⚠️ c6 tail slender and c11 fin slender both ~zero** — consistent across all entries. Slender body added mass is consistently negligible for this fish geometry. Probably a real physical signal (thin planar surfaces don't fit slender body assumptions well).

**⚠️ Whether the "stuck fin" is fixed** — c12 fin angular (11.74) ≈ c7 tail angular (11.28), so both segments are similarly damped. The fin has significant kutta now (6.44) which should drive it to move. Pending video review.

**Assessment:** Pending visual review on run\_3 and run\_9.

---

## Entry 9 — 15-segment result (6-run CMA-ES, corrected mass model, sinusoid\_weight=0.4, random\_start)

**Coefficients (n\_coef=15):**
```
[0.003824272448004096, 0.0008075555992586535, 1.0482615628117764, 0.40336902200940694, 0.4999409616725345,    ← head
 0.0003667597872783871, 0.0037009694569704868, 3.422936441781973, 0.01983709518349099, 0.5868366160406017,   ← tail body
 0.002644483709059951, 0.387065349356458, 19.960106465312155, 5.378367471972695, 5.847794641633156]          ← fin
```

**Optimizer:** CMA-ES, physics\_bounds (15-coeff), random\_start, stopped at σ=0.122 (~12.6h)

**Files:** `Outputs/sysid/opt_result_20260424_022801.yml` · `Outputs/sysid/convergence_20260424_022801.png`

**Loss function:** `mean_over_runs(wpos_mse)` with `start\_weight=4.0` + `sinusoid\_weight=0.4` blending (each run evaluated with both real Ghost\_Alpha and fitted sinusoid; optimizer minimizes the blended loss). **First entry using sinusoid blending.**

**Best loss:** 0.058859. Per-run wpos breakdown (run\_3 | run\_4 | run\_7 | run\_10 | run\_11 | run\_9): `0.019 | 0.047 | 0.038 | 0.019 | 0.152 | 0.077`

**Sinusoid validation gaps** (sin\_pos − wpos per run): `−0.009 | −0.002 | −0.015 | +0.009 | +0.008 | −0.009` — gaps near zero, indicating good generalisation to clean sinusoidal input.

**Mass model:** **Corrected** — head=750g, head-plate=60g, body=400g, COM=[0.0843, 0, 0.0088]. First entry with the physically correct mass distribution. **Not directly comparable to E1–E8** which used head≈1g, bodyMass=600g, COM=[0.128, 0, 0.003].

**Training runs:** run\_3 (1.0Hz 9mm), run\_4 (2.0Hz 9mm), run\_7 (3.0Hz 7mm, fixed), run\_10 (4.0Hz 7mm), run\_11 (2.0Hz 12mm), run\_9 (2.0Hz 6mm/−3mm, turning) — same 6 runs as E8

**Validation video run\_3:** (pending — use eval\_sysid.py WITHOUT --legacy\_mass)

**Coefficient analysis vs E8 (same 6 runs, old mass model):**

| # | Segment | Meaning | Bounds | **E8** | **E9** | Note |
|---|---------|---------|--------|--------|--------|------|
| c0 | head | blunt drag | [0.00, 0.15] | 0.0081 | **0.0038** | Low — consistent |
| c1 | head | slender | [0.00, 2.0] | 0.0544 | **0.0008** | Collapsed further — near zero |
| c2 | head | angular drag | [1.00, 5.0] | 2.1153 | **1.0483** | Near lower bound — lighter head needs less damping? |
| c3 | head | kutta (lift) | [0.00, 2.0] | 0.2048 | **0.4034** | Doubled — head lift increasingly important for turning |
| c4 | head | magnus | [0.00, 0.5] | 0.1055 | **0.4999** | ⚠️ AT UPPER BOUND — 0.367→0.105→0.500, non-monotone; bound must expand |
| c5 | tail | blunt drag | [0.00, 0.20] | 0.0016 | **0.0004** | Near zero — consistent |
| c6 | tail | slender | [0.00, 5.0] | 0.0054 | **0.0037** | Near zero — consistent across all entries |
| c7 | tail | angular drag | [2.00, 20.0] | 11.276 | **3.4229** | ⬇ Dramatic drop — damping shifted almost entirely to fin |
| c8 | tail | kutta | [0.00, 12.0] | ~0.000 | **0.0198** | Still ~zero — tail body generates no thrust |
| c9 | tail | magnus | [0.00, 6.0] | 1.578 | **0.5868** | Lower |
| c10 | fin | blunt drag | [0.00, 0.20] | 0.0519 | **0.0026** | Dropped — near zero |
| c11 | fin | slender | [0.00, 5.0] | ~0.000 | **0.3871** | ⬆ First time meaningfully nonzero — sinusoid blending revealed this |
| c12 | fin | angular drag | [2.00, 20.0] | 11.735 | **19.9601** | ⚠️ AT UPPER BOUND — bound must expand |
| c13 | fin | kutta | [0.00, 12.0] | 6.4424 | **5.3784** | Slightly lower — still main thrust driver |
| c14 | fin | magnus | [0.00, 6.0] | 1.1618 | **5.8478** | ⚠️ AT UPPER BOUND — bound must expand |

**Key findings:**

**✓ Total angular damping conserved** — c7+c12 = 3.42+19.96 = **23.4**, nearly identical to E8's 11.28+11.74 = 23.0. The optimizer consistently wants ~23 total angular damping units regardless of how it is distributed. However E9 concentrates almost all of it in the fin (ratio 1:6) vs E8's even split (1:1). This is a structurally different physical regime.

**✓ Sinusoid validation gaps near zero** — for 5 of 6 runs, |sin\_pos − wpos| < 0.015. This means the fluid model responds similarly to clean sinusoidal motor commands and to the real (slightly noisy, THD≈1.5%) motor signal. Good for RL deployment where the policy sends clean commands.

**⚠️ Three bounds at ceiling (c4, c12, c14)** — optimizer is still searching beyond the current bounds. E10 must expand all three:
- c4 head magnus: `[0.0, 0.5]` → `[0.0, 1.0]`
- c12 fin angular drag: `[2.0, 20.0]` → `[2.0, 30.0]`
- c14 fin magnus: `[0.0, 6.0]` → `[0.0, 10.0]`

**⚠️ Head angular drag c2 = 1.048 ≈ lower bound (1.0)** — may be trying to go below 1.0. Consider relaxing lower bound to 0.5 for E10 (or 0.0, given the corrected heavier head may not need forced damping).

**⚠️ Loss 0.058859 > E8's 0.039488** — but direct comparison is inappropriate. E9 (a) uses a different mass model (physics changed), (b) blends real+sinusoid in the loss (harder constraint), and (c) optimized for generalization, not just wpos minimisation. The sinusoid validation signal is the relevant metric, not the absolute loss.

**Assessment:** Visually unvalidated. Evaluate with `eval_sysid.py Outputs/sysid/opt_result_20260424_022801.yml` (no `--legacy_mass` flag). Structurally plausible — tail-as-transmitter / fin-as-propulsor interpretation holds, now with even stronger fin angular damping driving the separation.

**Shared 2026-04-24** with colleague and supervisor as the current best result for SoFi SysID.

**Bounds to update for E10:**
- c2 head angular drag lower bound: `1.0` → `0.0` (heavier head, may not need forced floor)
- c4 head magnus upper bound: `0.5` → `1.0`
- c12 fin angular drag upper bound: `20.0` → `30.0`
- c14 fin magnus upper bound: `6.0` → `10.0`

---

## Entry 10 — 15-segment result (6-run CMA-ES, expanded bounds, hingeDamping=0.1)

**Coefficients (n\_coef=15):**
```
[0.01635624835974899, 0.0004786561682654519, 1.41715154656536, 1.1809161765978933, 0.31901340138370904,    ← head
 0.028287545708593004, 0.02861642511564261, 6.792836151478609, 1.0581739078826322, 1.529155425224066,      ← tail body
 0.003591800883496392, 0.22804604331306363, 29.072027084731072, 8.547834020039708, 4.798994878887511]      ← fin
```

**Optimizer:** CMA-ES, physics\_bounds (15-coeff, expanded E10 bounds), random\_start, stopped at gen 145/500 (29%), σ=0.219 (converged)

**Files:** `Outputs/sysid/opt_result_20260425_014022.yml` · `Outputs/sysid/convergence_20260425_014022.png`

**Loss function:** `mean_over_runs(wpos_mse)` with `start\_weight=4.0` + `sinusoid\_weight=0.4`. Same as E9.

**Best loss:** 0.031446 — **new best overall**, including E8 (0.039488) which used a simpler loss.

**Training runs:** run\_3 (1.0Hz 9mm), run\_4 (2.0Hz 9mm), run\_7 (3.0Hz 7mm, fixed), run\_10 (4.0Hz 7mm), run\_11 (2.0Hz 12mm), run\_9 (2.0Hz 6mm/−3mm, turning) — same 6 runs as E8/E9

**Model note:** Still running with `hingeDamping=0.1` on tail joints — changed to 0.0 for E11 after analysis showed this is an unphysical internal load the optimizer was compensating for.

**Coefficient analysis vs E9:**

| # | Segment | Meaning | Bounds | **E9** | **E10** | Note |
|---|---------|---------|--------|--------|--------|------|
| c0 | head | blunt | [0.00, 0.15] | 0.0038 | **0.0164** | Still small |
| c1 | head | slender | [0.00, 2.0] | 0.0008 | **0.0005** | ~zero |
| c2 | head | angular | [0.00, 5.0] | 1.0483 | **1.4172** | Mid-range; lower bound relaxed freed it |
| c3 | head | kutta | [0.00, 2.0] | 0.4034 | **1.1809** | ⬆ Nearly tripled — head lift strongly active |
| c4 | head | magnus | [0.00, 1.0] | 0.4999 | **0.3190** | ✓ No longer at ceiling; expanded bound worked |
| c5 | tail | blunt | [0.00, 0.20] | 0.0004 | **0.0283** | Small |
| c6 | tail | slender | [0.00, 5.0] | 0.0037 | **0.0286** | ~zero — consistent across all entries |
| c7 | tail | angular | [2.00, 20.0] | 3.4229 | **6.7928** | Doubled — more tail damping than E9 |
| c8 | tail | kutta | [0.00, 12.0] | 0.0198 | **1.0582** | Non-zero for first time — tail body generates small thrust |
| c9 | tail | magnus | [0.00, 6.0] | 0.5868 | **1.5292** | Higher |
| c10 | fin | blunt | [0.00, 0.20] | 0.0026 | **0.0036** | ~zero |
| c11 | fin | slender | [0.00, 5.0] | 0.3871 | **0.2280** | Reduced |
| c12 | fin | angular | [2.00, **30.0**] | 19.9601 | **29.0720** | ⚠️ AT UPPER BOUND AGAIN — must expand to 40 for E11 |
| c13 | fin | kutta | [0.00, 12.0] | 5.3784 | **8.5478** | ⬆ Higher — fin thrust increased |
| c14 | fin | magnus | [0.00, **10.0**] | 5.8478 | **4.7990** | ✓ No longer at ceiling; expanded bound worked |

**Key findings:**

**✓ New best loss 0.031446** — improved over E9 (0.058859) and E8 (0.039488) on the same 6 training runs.

**✓ c4 and c14 no longer at ceiling** — expanding those bounds worked; the optimizer found genuine optima below the new ceilings.

**⚠️ c12 fin angular at ceiling again (29.07/30.0)** — must expand to [2, 40] for E11.

**⚠️ Total angular damping increased** — c7+c12 = 6.79+29.07 = **35.86**, up from 23.4 in E9. This is still partially driven by `hingeDamping=0.1` — E11 (with damping=0) should reveal whether this total drops.

**⬆ c3 head kutta nearly tripled (0.40→1.18)** — with `c4 head magnus` now free to find its true value (0.32, not forced to ceiling), the optimizer re-distributed turning forces differently. Head lift is a major turning driver.

**Validation video run\_3:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260426_195800\replay.mp4`

**Validation video run\_9:** `\\wsl.localhost\Ubuntu\home\sramc\Robotic_Fish\fishCREATE\Outputs\Field2Sim\Run_20260426_201543\replay.mp4`

**Assessment:** Visually validated — both run_3 and run_9 look quite nice. E11 will repeat with `hingeDamping=0.0` and `c12` bound kept at 20 (testing whether removing joint damping alone brings angular drag down).

---

## Entry 11 — 15-segment result (6-run CMA-ES, hingeDamping=0.0, c12 bound kept at 20)

**Coefficients (n\_coef=15):**
```
[0.0088, 0.0015, 0.5011, 0.9897, 0.9666,    ← head
 0.0857, 0.0094, 19.1857, 0.0182, 5.0509,   ← tail body
 0.1683, 0.0141, 19.9879, 11.9811, 9.7111]  ← fin
```

**Optimizer:** CMA-ES, physics\_bounds (15-coeff), random\_start

**Files:** `Outputs/sysid/opt_result_20260426_173337.yml`

**Loss function:** `mean_over_runs(wpos_mse)` with `start\_weight=4.0` + `sinusoid\_weight=0.4`. Same as E9/E10.

**Best loss:** 0.052981 — **worse than E10 (0.031446)**

**Training runs:** same 6 runs as E9/E10

**Model note:** `hingeDamping=0.0`. c12 bound kept at `[2.0, 20.0]` (not expanded to 40) — the hypothesis was that removing joint damping would reduce the need for high angular drag coefficients.

**Coefficient analysis vs E10:**

| # | Segment | Meaning | Bounds | **E10** | **E11** | Note |
|---|---------|---------|--------|---------|---------|------|
| c0 | head | blunt | [0.00, 0.15] | 0.0164 | **0.0088** | Halved |
| c1 | head | slender | [0.00, 2.0] | 0.0005 | **0.0015** | ~zero |
| c2 | head | angular | [0.00, 5.0] | 1.4172 | **0.5011** | Dropped — less head damping without joint friction |
| c3 | head | kutta | [0.00, 2.0] | 1.1809 | **0.9897** | Similar |
| c4 | head | magnus | [0.00, 1.0] | 0.3190 | **0.9666** | ⬆ Near ceiling |
| c5 | tail | blunt | [0.00, 0.20] | 0.0283 | **0.0857** | ⬆ Higher |
| c6 | tail | slender | [0.00, 5.0] | 0.0286 | **0.0094** | ~zero — consistent |
| c7 | tail | angular | [2.00, 20.0] | 6.7928 | **19.1857** | ⚠️ NEARLY AT CEILING — almost tripled vs E10 |
| c8 | tail | kutta | [0.00, 12.0] | 1.0582 | **0.0182** | Collapsed to zero |
| c9 | tail | magnus | [0.00, 6.0] | 1.5292 | **5.0509** | ⬆ High |
| c10 | fin | blunt | [0.00, 0.20] | 0.0036 | **0.1683** | ⬆ Much higher |
| c11 | fin | slender | [0.00, 5.0] | 0.2280 | **0.0141** | Collapsed |
| c12 | fin | angular | [2.00, **20.0**] | 29.0720 | **19.9879** | ⚠️ AT CEILING — would go higher if bound allowed |
| c13 | fin | kutta | [0.00, 12.0] | 8.5478 | **11.9811** | ⬆ Near ceiling |
| c14 | fin | magnus | [0.00, 10.0] | 4.7990 | **9.7111** | ⬆ Near ceiling |

**Key findings:**

**✗ Hypothesis REJECTED** — removing `hingeDamping` did NOT reduce angular drag requirements. The opposite happened: c7 nearly tripled (6.79 → 19.19) and c12 hit its ceiling again.

**✗ Total angular damping INCREASED** — c7+c12 = 19.19 + 19.99 = **39.18**, up from E10's 35.86. Removing joint damping forced the optimizer to find more fluid angular drag.

**✗ Loss is worse than E10 (0.052981 vs 0.031446)** — the model fits less well overall without joint damping.

**⚠️ c12 and c7 both at/near ceiling** — both bounds still too tight.

**Assessment:** E11 decisively shows that `hingeDamping=0.1` was doing genuine physical work (providing real damping on the joints), NOT causing the optimizer to inflate fluid coefficients in compensation. The high angular drag requirements are structural — they come from the tiny tail segment geometry, not from the joint damping setting. See analysis in `FLUID_COEFFICIENTS.md` header notes above. Revert to `hingeDamping=0.1` for E12+.

**Validation video:** pending (`eval_sysid.py Outputs/sysid/opt_result_20260426_173337.yml` with `hingeDamping=0.0`)

---

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
    ../sofi_control/field_test/data/20260401_first_swim/run_3_1.0Hz_9.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_4_2.0Hz_9.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_7_3.0Hz_7.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_10_4.0Hz_7.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_11_2.0Hz_12.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_9_2.0Hz_6.0mm_-3.0mm \
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
    ../sofi_control/field_test/data/20260401_first_swim/run_3_1.0Hz_9.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_4_2.0Hz_9.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_7_3.0Hz_7.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_10_4.0Hz_7.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_11_2.0Hz_12.0mm \
    ../sofi_control/field_test/data/20260401_first_swim/run_9_2.0Hz_6.0mm_-3.0mm \
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
