# System identification: fitting the model to the fish

The measurements come from [fishdeploy](https://github.com/Ramchi99/fishdeploy), cloned next to this
repository (or at `$FISHDEPLOY_ROOT`). Three fits, in the order they were done.

## 1. Tail stiffness and damping (in air)

Input: the 100 motor sweeps `fishdeploy/sysid/tail/data/20260320_*_r17.5_*_Benchmark_air_*.csv`
(three tendon amplitudes, four frequencies, several repeats; commanded and measured motor position).

```bash
python sysid/tail_stiffness/step1_stiffness.py            # fit; writes the per-joint k and d
python sysid/tail_stiffness/step2_fluid.py                # air versus water: added mass and drag (optional)
python sysid/tail_stiffness/step3_mujoco_validation.py    # static deflection check of the MuJoCo model
```

`step1` treats the tail as a second-order system seen from the motor: the real part of the measured
transfer function is `k_eff - I_eff * omega^2`, and its intercept at `omega^2 -> 0` gives the effective
stiffness (0.98 N m/rad); the imaginary part gives the damping. The effective values are distributed over
the five hinges with a Euler-Bernoulli profile, giving `hingeStiffness = [2.99, 2.39, 2.13, 1.77, 1.32]`
N m/rad and `hingeDamping = [0.033, 0.027, 0.024, 0.020, 0.015]` N m s/rad, which `geometry/auto_SOFI_cpg.py`
carries. Paper Fig. 4 is this fit (`paper/fig_tailstiff_paper.py`). A dynamic fit was used, not a static one,
so friction and backlash average out, and the fit does not depend on the fluid or motor models.

## 2. Fluid coefficients (the 12 open-loop field runs)

Input: `fishdeploy/field_test/data/20260401_first_swim/run_*/processed/03_field2sim_ready.csv`, the
resampled pose, motor and IMU of each run.

`sysid/fluid/opt_field2sim.py` builds a two-fish model (`geometry/auto_SOFI_field2sim.py`): a "ghost"
fish is driven to the measured state through an error-state Kalman filter (`es_ekf.py`) and the "twin"
runs free with the same motor command. The loss is the twin's position error over time, with the start
transient weighted four times and a sinusoid-fit term on the tail. CMA-ES over the 15 coefficients
(head, tail body and fin: three ellipsoid drag coefficients plus Kutta lift and Magnus terms each),
inside physical bounds:

```bash
python sysid/fluid/opt_field2sim.py \
    ../fishdeploy/field_test/data/20260401_first_swim/run_3_1.0Hz_9.0mm \
    ../fishdeploy/field_test/data/20260401_first_swim/run_4_2.0Hz_9.0mm \
    ../fishdeploy/field_test/data/20260401_first_swim/run_7_3.0Hz_7.0mm \
    ../fishdeploy/field_test/data/20260401_first_swim/run_10_4.0Hz_7.0mm \
    ../fishdeploy/field_test/data/20260401_first_swim/run_11_2.0Hz_12.0mm \
    ../fishdeploy/field_test/data/20260401_first_swim/run_9_2.0Hz_6.0mm_-3.0mm \
    --n_coef 15 --optimizer cma --physics_bounds --iter 1000 \
    --start_weight 4.0 --random_start --sinusoid_weight 0.4
python sysid/fluid/eval_sysid.py       # held-out runs 1, 2, 5, 6, 8, 12; the twin is ~13 % slow straight, ~7 % fast in turns
```

The result used by the deployed policy is **E13a** (loss 0.040, 2 028 evaluations; the optimiser checkpoint
is in the archive, `05_fluid_identification/field2sim_cmaes/checkpoint_20260501_044853.yml`):

```
head      0.005822  0.044600  0.812577  0.521112  0.902770
tail body 0.039474  0.551234  5.427544  0.000956  1.581176
fin       0.170727  0.003855 18.310332 10.148229  5.951302
```

These are `SYSTEMPARAMETERS["fluidCoef"]` in `geometry/auto_SOFI_cpg.py` and `fluid_tuning.coefficients`
in `sysid/fluid/field2sim.yml`. `sysid/fluid/FLUID_COEFFICIENTS.md` is the log of every attempt (E1 to
E13) with what changed between them. Paper Fig. 5 (`paper/fig5_fluidfit_paper.py`) replays the fitted and
held-out runs through this model.

The direct-action model (`geometry/auto_SOFI_da.py`) uses a later refit, **refitB**, made on the
closed-loop deployment runs with a windowed multiple-shooting loss (the pipeline was not kept public; its
results are in the archive, `05_fluid_identification/closed_loop_refit/`). A further refit after correcting
the centre of mass (refitC) did not lower the fin coefficients, which is the evidence in the paper that the
inflated fin terms stand in for unsteady thrust rather than for a mass error (`docs/com_frame_bug.md`).

## 3. Motor (sim versus real)

`sysid/motor/` compares the servo in MuJoCo with the field and deployment logs: torque and speed limits,
tracking range, the position gain and rotor inertia that reproduce the measured tracking (`06_motor_calibration.py`,
paper Table "motor"), the Bode response, and the sensor noise levels used by `rl/models/sensor_model.py`.
Run the numbered scripts in order from the repository root; `results/*.json` are the numbers as used.
`00_quality_screen.py` and `05_noise_estimation.py` also look for deployment sessions before 2026-05-20
that are not public and simply skip them.
