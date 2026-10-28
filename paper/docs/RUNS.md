# The twelve open-loop field runs

Session `20260401_first_swim`, 1 April 2026. These are the runs the fluid model was
identified and validated on, and they are the reference for every open-loop number in the
paper. Data: `upstream/sofi_control/field_test/data/20260401_first_swim/<run>/processed/03_field2sim_ready.csv`.

## Conventions

| quantity | convention | reason |
|---|---|---|
| amplitude | motor shaft angle [rad], from the encoder | avoids the 12.5 / 17.5 mm ambiguity below |
| frequency | measured per half-stroke from the encoder | a run's name is what was commanded, not what happened |
| speed | arc length of the tail-beat-smoothed path over the 50–80 % window | a swim-axis regression undercounts a curving path |
| vision scale | ×0.80 on all distances | the tag size was entered as 0.10 m; the printed tags were 0.08 m |

**On the two pulley radii.** Both constants are correct, for different quantities. The field
software sized its ticks assuming r = 12.5 mm; the physical pulley is r = 17.5 mm. So
`motor angle [rad] = nominal_mm / 12.5`, while `physical tendon travel [mm] = angle × 17.5`,
a factor 1.4 above nominal. Confirmed on run_3: nominal 9 mm gives a commanded 0.713 rad,
and 9/12.5 = 0.72. Working in shaft angle throughout means neither number is ever needed.

## The runs

`A_cmd` and `A_ach` are commanded and achieved half-stroke peak shaft angle. `trk` is their
ratio. `u` and `path` carry the ×0.80 vision correction.

| run | role | f [Hz] | A_cmd [rad] | A_ach [rad] | trk | A·f | u [m/s] | path [m] | dur [s] |
|---|---|---|---|---|---|---|---|---|---|
| run_3 (1.0 Hz, 9 mm) | fit | 1.00 | 0.713 | 0.637 | 0.89 | 0.637 | 0.220 | 2.59 | 14.4 |
| run_4 (2.0 Hz, 9 mm) | fit | 2.00 | 0.718 | 0.504 | 0.70 | 1.009 | 0.335 | 2.63 | 9.8 |
| run_7 (3.0 Hz, 7 mm) | fit | 3.00 | 0.560 | 0.331 | 0.59 | 0.995 | 0.340 | 2.47 | 9.2 |
| run_10 (4.0 Hz, 7 mm) | fit | 4.00 | 0.559 | 0.231 | 0.41 | 0.925 | 0.383 | 2.64 | 8.4 |
| run_11 (2.0 Hz, 12 mm) | fit | 2.00 | 0.958 | 0.520 | 0.54 | 1.042 | 0.343 | 2.71 | 9.9 |
| run_9 (2.0 Hz, 6 mm, −3 mm) | fit, turn | 2.00 | 0.478 | 0.403 | 0.84 | 0.806 | 0.223 | 2.18 | 12.2 |
| run_1 (1.0 Hz, 5 mm) | validation | 1.00 | 0.395 | 0.353 | 0.89 | 0.353 | 0.143 | 2.45 | 21.3 |
| run_2 (2.0 Hz, 5 mm) | validation, see note | 2.00 | 0.398 | 0.349 | 0.88 | 0.698 | 0.219 | 2.78 | 14.8 |
| run_5 (1.0 Hz, 7 mm) | validation | 1.00 | 0.555 | 0.500 | 0.90 | 0.500 | 0.175 | 2.73 | 20.1 |
| run_6 (2.0 Hz, 7 mm) | validation | 2.00 | 0.559 | 0.460 | 0.82 | 0.920 | 0.287 | 2.53 | 11.3 |
| run_12 (2.0 Hz, 12 mm, P=1600) | validation | 1.99 | 0.958 | 0.511 | 0.53 | 1.018 | 0.317 | 2.50 | 9.7 |
| run_8 (2.0 Hz, 6 mm, +3 mm) | validation, turn | 2.00 | 0.478 | 0.405 | 0.85 | 0.811 | 0.219 | 1.66 | 9.8 |

The E13a fluid coefficients were fitted on run_3, 4, 7, 10, 11 and the turn run_9. The other
six, including the opposite-direction turn run_8, were never seen by the fit. Runs 13, 14 and
15 exist in the session but were not processed to field2sim-ready form and are not used.

**run_2 was nudged by hand shortly after release and slid off line.** Its steady-state speed
error is ordinary; only its path shape is corrupted. It is kept as a domain point and excluded
from path-shape statistics. The signature is visible in the data: it is the only run where
arc-length speed exceeds the swim-axis regression by more than 1 % (ratio 1.064).

## Why the table can be trusted

Three independent checks, all run against methods that predate this table.

1. **Speed.** The arc-length method agrees with the earlier swim-axis regression to within
   0.3–0.7 % on nine of the ten straight runs, both uncorrected. The one outlier is run_2,
   for the reason above. The turns were absent from the earlier summary entirely, which is
   why a path-based method was needed.
2. **Amplitude.** `A_ach` reproduces the independently computed `act_mm / 17.5` to three
   decimals on all twelve runs. run_3: 11.24/17.5 = 0.642 against 0.637. run_10: 4.09/17.5 =
   0.234 against 0.231.
3. **The half-stroke reduction is faithful here.** Measured peak |dθ/dt| divided by A·2πf has
   a median of 1.002 across all twelve runs. The achieved motion is closer to a triangle than
   a sine, and this is the check that it does not matter for these runs. It does not hold for
   the closed-loop deployment logs, where the same ratio is 1.30.

## Twin against measurement

Two phase-matched metrics, both ratios, so the vision scale factor cancels and the E13a fit
frame can be used directly.

| run | A·f | u_real | u_twin | speed error | path error |
|---|---|---|---|---|---|
| R3 | 0.637 | 0.275 | 0.236 | −14.2 % | 3.3 % |
| R4 | 1.009 | 0.419 | 0.366 | −12.6 % | 4.6 % |
| R7 | 0.995 | 0.425 | 0.365 | −14.0 % | 6.6 % |
| R10 | 0.925 | 0.479 | 0.386 | −19.5 % | 4.0 % |
| R11 | 1.042 | 0.429 | 0.370 | −13.6 % | 5.3 % |
| R9 (turn) | 0.806 | 0.279 | 0.302 | +8.3 % | 10.7 % |
| R1 | 0.353 | 0.178 | 0.145 | −18.7 % | 6.2 % |
| R2 (nudged) | 0.698 | 0.274 | 0.259 | −5.3 % | 20.3 % |
| R5 | 0.500 | 0.218 | 0.195 | −10.7 % | 8.6 % |
| R6 | 0.920 | 0.359 | 0.332 | −7.5 % | 10.8 % |
| R12 | 1.018 | 0.396 | 0.365 | −7.9 % | 8.1 % |
| R8 (turn) | 0.811 | 0.274 | 0.292 | +6.7 % | 19.6 % |

Two results:

- **The twin is about 13 % too slow on every straight run**, between −5 % and −19 %, all with
  the same sign. That is a directional bias, not scatter.
- **On turns the sign reverses and the twin runs about 7 % fast** (R9 +8.3 %, R8 +6.7 %), and
  the turns also carry the two worst path errors. Turning is the weaker axis.

Error does not grow at low A·f. The correlation between A·f and speed error is −0.17 across
all runs and −0.19 on the straights; against path error it is −0.12. Straight runs average
12.2 % speed error below A·f = 0.8 and 12.5 % above it.

An earlier pass compared twin and measurement at a fixed t = 8 s. That metric was defective
and its conclusions were withdrawn: runs last 8.4 to 21.3 s, so 8 s falls mid-transient for
the slow runs and near the end for the fast ones, and it was partly measuring acceleration
mismatch weighted by run duration.

## Where the evidence actually is

Achieved A·f spans 0.35 to 1.04 while commanded values reach 1.92 (run_11 commands 1.92 and
achieves 1.04). The plant caps A·f at roughly 1.0. Tracking falls from 0.89 to 0.70 to 0.59
to 0.41 across 1 to 4 Hz.

As a result **five of the six fitted runs sit on the A·f ≈ 0.92–1.04 frontier**: the plant's
saturation collapsed a two-dimensional sampling grid onto a one-dimensional curve. A hull of
the fitted runs alone would be a sliver. What makes the identified domain genuinely
two-dimensional is the validation runs, which fill in below the frontier — run_1 at 0.353,
run_5 at 0.500.

The region with evidence behind it is therefore the convex hull of all twelve runs:
A ∈ [0.23, 0.64] rad, f ∈ [1, 4] Hz, bounded above by the plant frontier. Since error is flat
inside it, that is a defensible statement of where the model was tested. The low-amplitude,
low-frequency corner near 0.23 rad at 1 Hz is unsampled, so squaring the hull off there would
claim a corner no run visited.

**The A·f ceiling does not bound frequency.** A policy can sit at 8 Hz with A = 0.12 rad:
A·f = 0.96, under the plant's speed ceiling, at a frequency the identification never visited.
The two mechanisms are complementary. The plant bounds A·f; the action space is what bounds f.
