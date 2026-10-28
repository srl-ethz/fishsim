"""Noise estimation helpers — pure functions, no I/O."""

import numpy as np


def estimate_stationary_std(signal: np.ndarray) -> float:
    """Robust std estimate using median absolute deviation."""
    if len(signal) == 0:
        return 0.0
    mad = np.median(np.abs(signal - np.median(signal)))
    return float(mad * 1.4826)  # consistent estimator for Gaussian


def extract_fps_from_frame_timestamps(timestamps: np.ndarray):
    """Return (mean_fps, jitter_std_s) from raw frame arrival timestamps."""
    if len(timestamps) < 2:
        return 0.0, 0.0
    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]
    if len(intervals) == 0:
        return 0.0, 0.0
    mean_fps  = float(1.0 / np.median(intervals))
    jitter_std = float(np.std(intervals))
    return mean_fps, jitter_std


def compute_frame_drop_rate(frame_seq: np.ndarray) -> float:
    """Fraction of expected frames that were dropped (gap > 1 in frame_seq)."""
    if len(frame_seq) < 2:
        return 0.0
    gaps = np.diff(frame_seq.astype(int))
    drops = (gaps - 1).clip(min=0).sum()
    expected = int(gaps.sum())
    return float(drops / max(expected, 1))
