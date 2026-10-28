"""Unit tests for noise estimation logic."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sysid.motor._noise_utils import (
    estimate_stationary_std,
    extract_fps_from_frame_timestamps,
    compute_frame_drop_rate,
)


def test_stationary_std_recovers_injected_noise():
    rng = np.random.default_rng(42)
    signal = rng.normal(0.0, 0.001, size=1000)
    std_est = estimate_stationary_std(signal)
    assert abs(std_est - 0.001) < 0.0003  # within 30%


def test_stationary_std_on_constant_signal():
    signal = np.zeros(500)
    assert estimate_stationary_std(signal) == 0.0


def test_fps_extraction():
    # Simulate 30 frames at 37 Hz
    timestamps = np.arange(30) / 37.0
    fps, jitter_std = extract_fps_from_frame_timestamps(timestamps)
    assert abs(fps - 37.0) < 1.0
    assert jitter_std < 0.002


def test_fps_extraction_with_jitter():
    rng = np.random.default_rng(0)
    base = np.arange(100) / 37.0
    jitter = rng.normal(0.0, 0.003, size=100)
    timestamps = base + np.cumsum(jitter)
    fps, jitter_std = extract_fps_from_frame_timestamps(timestamps)
    assert abs(fps - 37.0) < 3.0
    assert 0.001 < jitter_std < 0.01


def test_frame_drop_rate_no_drops():
    frame_seq = np.arange(100)
    rate = compute_frame_drop_rate(frame_seq)
    assert rate == 0.0


def test_frame_drop_rate_with_drops():
    # Every 10th frame is dropped (gap of 2)
    frame_seq = np.array([i for i in range(100) if i % 10 != 5])
    rate = compute_frame_drop_rate(frame_seq)
    assert 0.05 < rate < 0.20
