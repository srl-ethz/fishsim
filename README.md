# fishsim

Simulation side of **Swim2Real / "All You Need Is Low Fidelity"** (ICRA 2026 submission, Soft Robotics Lab,
ETH Zurich): a tendon-driven soft robotic fish (SOFI) learns to swim to targets in a MuJoCo simulation whose
fluid is a stateless, quasi-steady ellipsoid model, and the policy transfers to the pool without any
adaptation. This repository is the model, the fits that identify it from measurements, the reinforcement
learning, and the paper's figures. The hardware, the field tests and the measured data are in
[fishdeploy](https://github.com/Ramchi99/fishdeploy); clone it next to this repository (or set
`FISHDEPLOY_ROOT`) and everything below finds its data.

## In one picture

```
 fishdeploy measurements                 fishsim                                     fishdeploy
 ─────────────────────                   ───────                                     ──────────
 tail sweeps in air  ──► sysid/tail_stiffness ──► hinge k, d ─┐
                                                              ├─► geometry/auto_SOFI_cpg.py ─► SOFI_cpg.xml ─► rl/ (SAC, CPG actions, curriculum) ─► policies/sofi_cpg_rl_policy.zip ─► deploy/
 12 open-loop pool runs ─► sysid/fluid (ghost/twin, CMA-ES) ─► 15 fluid coefficients ─┘                                                                                    │
                                                                                                                              43 closed-loop runs ◄──────────────────────────┘
                                                                                                                                        │
                                                                                                                   paper/ (the ten figures) ◄─┘
```

## Setup

```bash
micromamba env create -f environment.yml && micromamba activate fishsim
git clone https://github.com/Ramchi99/fishdeploy ../fishdeploy
```

Python 3.11, MuJoCo 3.3.1, Stable-Baselines3 2.6.0. The paper figures also need LaTeX with `dvipng`
(matplotlib `usetex`), and Inkscape or Chrome/Edge for the three SVG figures.

## The five commands

Run from the repository root.

```bash
python geometry/auto_SOFI_cpg.py                                  # 1. build the model (geometry/SOFI_cpg.xml)
python sysid/tail_stiffness/step1_stiffness.py                    # 2. fit the hinge stiffness and damping to the air sweeps
python sysid/fluid/opt_field2sim.py <six run dirs> --n_coef 15 --optimizer cma --physics_bounds \
       --iter 1000 --start_weight 4.0 --random_start --sinusoid_weight 0.4    # 3. fit the 15 fluid coefficients (docs/SYSID.md)
python rl/train_rl_curriculum.py --env sofi_full_curriculum_cpg   # 4. train the policy (docs/TRAINING.md; or the W&B sweep)
make -C paper figures check                                       # 5. rebuild the ten paper figures and compare with the submitted ones
```

## Layout

| Folder | What |
|---|---|
| `geometry/` | `auto_SOFI_cpg.py`: the identified model (E13a fluid, measured hinge stiffness and damping, CAD inertia, kp 800 servo) that generates `SOFI_cpg.xml`. `auto_SOFI_da.py`: the direct-action variant (refitB fluid, identified servo). `auto_SOFI_field2sim.py`: the two-fish model the fluid fit runs. `meshes/`: the four CAD meshes the models load |
| `sysid/` | `tail_stiffness/`: the dynamic stiffness fit (paper Fig. 4). `fluid/`: the fluid-coefficient fit by ghost-and-twin replay with CMA-ES, its config, and the log of every attempt. `motor/`: the sim-versus-real motor study (paper Table "motor") |
| `rl/` | `environments/cpg/`: the Gymnasium environment with CPG actions and the sensor-noise, physics-randomisation and target curriculum stages. `environments/da/`: the direct-action comparison. `models/`: sensor and physics randomisation. `train_rl_curriculum.py`, `opt_hyperRL.py`, `sweepconfigs/`, `eval/` |
| `policies/` | The deployed policy (md5 `7bbe08b8`) and the direct-action policy, with provenance |
| `paper/` | One script per figure, `make figures`, `make check`, the static images, the cached simulation inputs, the submitted PDFs as reference |
| `docs/` | `SYSID.md`, `TRAINING.md`, and the working notes on the curriculum and on the direct-action exploit |
| `tests/` | unit tests of the motor study |

## Reproducibility status

Rebuilt from a clean clone of this branch and of fishdeploy on 2026-10-28 and compared with the PDFs as
submitted (`make -C paper check`): nine of the ten figures match in text and byte size; Fig. 5 (the fluid-fit
replay) matches in text, differs by 28 bytes, and renders pixel-identical except for sub-pixel shifts along
the simulated trajectory, i.e. floating-point differences of a MuJoCo replay made on another machine.
`geometry/SOFI_cpg.xml` and `SOFI_da.xml` regenerate byte-identically from their generators, and the tail
stiffness fit reproduces k_eff = 0.98 N m/rad.

## Credits

Mike Y. Michelis (original fishsim: model generator, system identification and RL baseline), Liam Maloney
(CPG environment, curriculum trainer, hardware), Simon Ramchandani (tail and fluid identification,
sim-to-real analysis, direct-action study, paper figures). Soft Robotics Lab, ETH Zurich, 2025 to 2026.
MIT license.
