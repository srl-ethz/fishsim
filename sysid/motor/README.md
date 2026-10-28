# sim2real — SOFI Sim-to-Real Gap Analysis

Run modules in order. Review `run_index.json` after step 1 before proceeding.

```bash
conda activate mujoco
cd sofi_control/

python sim2real/00_quality_screen.py   # → run_index.json (REVIEW THIS)
python sim2real/01_motor_torque.py     # → results/01_motor_torque.json
python sim2real/02_motor_range.py      # → results/02_motor_range.json
python sim2real/03_pd_gains.py         # → results/03_pd_gains.json
python sim2real/05_noise_estimation.py # → results/05_noise_estimation.json
python sim2real/04_domain_randomization.py  # → results/04_domain_randomization.json

pytest tests/sim2real/ -v
```

Plots saved to `sim2real/plots/` (gitignored). Results JSON committed.

## Data paths (relative to repo root)
- Field test: `field_test/data/20260401_first_swim/run_*/processed/02_fish_log_trimmed.csv`
- System ID: `system_id/dynamixel_tracking_limits/data/`
- RL deploy (May 20): `rl_deploy/runs/20260520_*/raw/run.csv`
- RL deploy (May 8 static hold): `rl_deploy/runs/20260508_*straight_then_left/raw/run.csv`
- Frame timestamps: `rl_deploy/runs/20260520_*/raw/frame_timestamps.csv`
- May 7 noise config: `rl_deploy/data/20260507_*/noise_config.json`
