"""Shared data loaders for the sim2real analysis pipeline."""

import re
import numpy as np
import pandas as pd
from pathlib import Path

import os
REPO_ROOT = Path(__file__).resolve().parent.parent.parent          # fishsim
FISHDEPLOY = Path(os.environ.get("FISHDEPLOY_ROOT", REPO_ROOT.parent / "fishdeploy")).resolve()
FIELD_TEST_ROOT = FISHDEPLOY / "field_test/data/20260401_first_swim"
SYSID_ROOT      = FISHDEPLOY / "sysid/tail/data"
RL_DEPLOY_ROOT  = FISHDEPLOY / "deploy/runs"

TICKS_PER_REV = 4096
RAD_PER_TICK  = 2 * np.pi / TICKS_PER_REV
TORQUE_CONST  = 1.769   # Nm/A
CURRENT_SCALE = 0.00269 # A per raw Load unit


def load_field_test_runs():
    """Return dict[run_name → DataFrame] for all 15 field test runs.

    DataFrame columns added beyond raw CSV:
      goal_rad   — Goal ticks converted to radians (zeroed at first sample)
      actual_rad — Actual ticks converted to radians (zeroed at first sample)
      vel_rad_s  — Vel raw * 0.229 RPM/unit * 2π/60
      torque_Nm  — Load_raw * CURRENT_SCALE * TORQUE_CONST (Load=-1 row → NaN)
      speed_limit_rad_s — 5.55 * (Volt / 12.0) per row
    """
    runs = {}
    for path in sorted(FIELD_TEST_ROOT.glob("run_*/processed/02_fish_log_trimmed.csv")):
        run_name = path.parts[-3]  # e.g. run_3_1.0Hz_9.0mm
        df = pd.read_csv(path)

        zero_ticks = df["Actual"].iloc[0]
        df["goal_rad"]   = (df["Goal"]   - zero_ticks) * RAD_PER_TICK
        df["actual_rad"] = (df["Actual"] - zero_ticks) * RAD_PER_TICK

        df["vel_rad_s"] = df["Vel"] * 0.229 * 2 * np.pi / 60.0

        raw_load = df["Load"].astype(float)
        raw_load[raw_load == -1] = np.nan   # sentinel first row
        df["torque_Nm"] = raw_load * CURRENT_SCALE * TORQUE_CONST

        df["speed_limit_rad_s"] = 5.55 * (df["Volt"] / 12.0)

        runs[run_name] = df
    return runs


def parse_run_name_field(run_name):
    """Extract (freq_Hz, amp_mm) from 'run_N_F.FHz_A.Amm' string."""
    m = re.search(r"_([\d.]+)Hz_([\d.]+)mm", run_name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _parse_two_section_csv(path):
    """Parse a CSV with two header rows separated by a blank header row.

    Returns (meta_df, data_df). The file looks like:
        Col1,Col2,Col3
        val, val, val
        DataColA,DataColB
        0, 0
        25, 63
    """
    lines = Path(path).read_text().splitlines()
    # Find second header: first line after the metadata block that looks like a header
    split_idx = None
    for i in range(1, len(lines)):
        # A header line has no pure-numeric first token
        try:
            float(lines[i].split(",")[0])
        except ValueError:
            if i > 1:  # skip the very first header
                split_idx = i
                break
    if split_idx is None:
        raise ValueError(f"Could not find second header in {path}")

    from io import StringIO
    meta_df = pd.read_csv(StringIO("\n".join(lines[:split_idx])), nrows=1)
    data_df = pd.read_csv(StringIO("\n".join(lines[split_idx:])))
    return meta_df, data_df


def load_stiffness_calibration():
    """Return list of dicts with keys: pulley_mm, settle_s, max_current_mA, data_df."""
    results = []
    for path in sorted(SYSID_ROOT.glob("*Stiffness_Calibration.csv")):
        if "delete" in path.name.lower():
            continue
        meta, data = _parse_two_section_csv(path)
        results.append({
            "path": path,
            "pulley_mm":      float(meta["PulleyRadius_mm"].iloc[0]),
            "settle_s":       float(meta["SettleTime_s"].iloc[0]),
            "max_current_mA": float(meta["MaxCurrent_mA"].iloc[0]),
            "data":           data,  # columns: Commanded_Current_mA, Measured_Delta_Ticks
        })
    return results


def load_benchmark_runs():
    """Return list of dicts with keys parsed from filename + data_df."""
    results = []
    pattern = re.compile(
        r"(\d+)_r([\d.]+)_d([\d.]+)_Benchmark_(air|water)_([\d.]+)Hz_([\d.]+)Amp"
    )
    for path in sorted(SYSID_ROOT.glob("*Benchmark*.csv")):
        m = pattern.search(path.name)
        if not m:
            continue
        meta, data = _parse_two_section_csv(path)
        zero_ticks = data["Position"].iloc[0]
        data["pos_rad"]  = (data["Position"]     - zero_ticks) * RAD_PER_TICK
        data["goal_rad"] = (data["GoalPosition"]  - zero_ticks) * RAD_PER_TICK
        data["vel_rad_s"] = data["Velocity"] * 0.229 * 2 * np.pi / 60.0
        raw_load = data["Load"].astype(float)
        data["torque_Nm"] = raw_load * CURRENT_SCALE * TORQUE_CONST
        results.append({
            "path":      path,
            "run_id":    int(m.group(1)),
            "pulley_mm": float(m.group(2)),
            "depth_mm":  float(m.group(3)),
            "env":       m.group(4),
            "freq_Hz":   float(m.group(5)),
            "amp_pct":   float(m.group(6)),
            "data":      data,
        })
    return results


def load_rl_deploy_runs(pattern="20260520_*"):
    """Return dict[run_name → DataFrame] for RL deploy runs matching glob pattern.

    DataFrame columns added:
      motor_pos_rad — arctan2(obs_01, obs_00)
      step_dt_s     — diff(pi_time), first row = median
    """
    runs = {}
    for run_dir in sorted(RL_DEPLOY_ROOT.glob(pattern)):
        csv_path = run_dir / "raw" / "run.csv"
        if not csv_path.exists():
            continue
        run_name = run_dir.name
        df = pd.read_csv(csv_path)
        df["motor_pos_rad"] = np.arctan2(df["obs_01"].values, df["obs_00"].values)
        dt = np.diff(df["pi_time"].values, prepend=np.nan)
        dt[0] = np.nanmedian(dt[1:])
        df["step_dt_s"] = dt
        runs[run_name] = df
    return runs


def load_frame_timestamps(run_name, pattern="20260520_*"):
    """Return DataFrame from frame_timestamps.csv for a single run, or None."""
    for run_dir in RL_DEPLOY_ROOT.glob(pattern):
        if run_dir.name == run_name:
            ts_path = run_dir / "raw" / "frame_timestamps.csv"
            if ts_path.exists():
                return pd.read_csv(ts_path)
            return None
    return None


def load_may7_noise_configs():
    """Return list of noise_config dicts from May 7 shadow_zero runs."""
    import json
    configs = []
    for path in sorted((FISHDEPLOY / "deploy/data").glob("20260507_*/noise_config.json")):
        with open(path) as f:
            cfg = json.load(f)
        cfg["_path"] = str(path)
        configs.append(cfg)
    return configs


def load_run_index(path=None):
    """Load run_index.json. Raises FileNotFoundError if not yet generated."""
    import json
    if path is None:
        path = Path(__file__).parent / "run_index.json"
    if not Path(path).exists():
        raise FileNotFoundError(
            f"run_index.json not found at {path}. Run 00_quality_screen.py first."
        )
    with open(path) as f:
        return json.load(f)


def get_usable_rl_df(run_name, df, run_index, min_quality=("A", "B")):
    """Slice a run DataFrame to its usable_window for the given min_quality tiers.

    Returns None if the run is not in min_quality.
    """
    entry = run_index.get("rl_deploy", {}).get(run_name)
    if entry is None or entry.get("quality") not in min_quality:
        return None
    start, end = entry["usable_window"]
    return df.iloc[start:end + 1].copy()
