"""Unit tests for quality screening logic (no real data required)."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sysid.motor._quality_utils import (
    compute_rl_run_signals,
    assign_quality_tier,
    compute_usable_window,
)


def _make_df(n=200, *, frozen=False, done_row=None, vision_age=None):
    motor_pos = np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)
    if frozen:
        motor_pos[:] = 0.5
    obs_00 = np.cos(motor_pos).astype(np.float32)
    obs_01 = np.sin(motor_pos).astype(np.float32)
    pi_time = np.linspace(0, n * 0.02, n)
    done = np.zeros(n, dtype=int)
    if done_row is not None:
        done[done_row] = 1
    if vision_age is None:
        vision_age = np.full(n, 50.0)
    motor_setpoint = np.sin(np.linspace(0, 4 * np.pi, n)).astype(np.float32)
    frame_seq = np.arange(n, dtype=int)
    df = pd.DataFrame({
        "pi_time": pi_time,
        "obs_00": obs_00,
        "obs_01": obs_01,
        "motor_setpoint": motor_setpoint,
        "done": done,
        "vision_age_ms": vision_age,
        "frame_seq": frame_seq,
    })
    df["motor_pos_rad"] = np.arctan2(df["obs_01"], df["obs_00"])
    df["step_dt_s"] = np.diff(pi_time, prepend=pi_time[0])
    return df


def test_motor_frozen_detection():
    df = _make_df(frozen=True)
    signals = compute_rl_run_signals(df)
    assert signals["motor_frozen"] is True


def test_motor_active_not_frozen():
    df = _make_df(frozen=False)
    signals = compute_rl_run_signals(df)
    assert signals["motor_frozen"] is False


def test_reached_target_via_done_column():
    df = _make_df(done_row=150)
    signals = compute_rl_run_signals(df)
    assert signals["reached_target"] is True


def test_not_reached_target():
    df = _make_df()
    signals = compute_rl_run_signals(df)
    assert signals["reached_target"] is False


def test_vision_lost_end_row():
    age = np.full(200, 50.0)
    age[180:] = 600.0  # vision lost for last 20 rows
    df = _make_df(vision_age=age)
    signals = compute_rl_run_signals(df)
    assert signals["vision_lost_end_row"] == 179


def test_quality_A():
    signals = {
        "reached_target": True,
        "vision_lost_early": False,
        "motor_frozen": False,
        "run_duration_s": 30.0,
        "jitter_fraction": 0.01,
    }
    assert assign_quality_tier(signals) == "A"


def test_quality_X_frozen():
    signals = {
        "reached_target": False,
        "vision_lost_early": False,
        "motor_frozen": True,
        "run_duration_s": 20.0,
        "jitter_fraction": 0.01,
    }
    assert assign_quality_tier(signals) == "X"


def test_quality_X_short():
    signals = {
        "reached_target": False,
        "vision_lost_early": False,
        "motor_frozen": False,
        "run_duration_s": 3.0,
        "jitter_fraction": 0.01,
    }
    assert assign_quality_tier(signals) == "X"


def test_quality_B():
    signals = {
        "reached_target": False,
        "vision_lost_early": False,
        "motor_frozen": False,
        "run_duration_s": 20.0,
        "jitter_fraction": 0.05,
    }
    assert assign_quality_tier(signals) == "B"


def test_quality_C_vision_lost_early():
    signals = {
        "reached_target": False,
        "vision_lost_early": True,
        "motor_frozen": False,
        "run_duration_s": 20.0,
        "jitter_fraction": 0.05,
    }
    assert assign_quality_tier(signals) == "C"


def test_usable_window_returns_valid_range():
    df = _make_df(n=200)
    start, end = compute_usable_window(df, vision_lost_end_row=180)
    assert start >= 0
    assert end <= 180
    assert start < end
