"""Quality screening helpers — pure functions, no I/O, easily testable."""

import numpy as np


def compute_rl_run_signals(df):
    """Compute quality signals from a single RL deploy DataFrame.

    Expects columns: motor_pos_rad, motor_setpoint, done, vision_age_ms,
                     frame_seq, pi_time, step_dt_s.
    Returns dict of scalar signals.
    """
    n = len(df)

    motor_frozen = float(df["motor_pos_rad"].std()) < 0.01

    reached_target = bool((df["done"] == 1).any())

    # Last row where vision is fresh (age < 500 ms)
    fresh_mask = df["vision_age_ms"] < 500
    if fresh_mask.any():
        vision_lost_end_row = int(df.index[fresh_mask][-1] - df.index[0])
    else:
        vision_lost_end_row = 0

    run_duration_s = float(df["pi_time"].iloc[-1] - df["pi_time"].iloc[0])

    # Vision lost before 30% of run
    vision_lost_early = (vision_lost_end_row / max(n - 1, 1)) < 0.30

    # Control loop jitter: fraction of steps with dt > 35 ms
    jitter_fraction = float((df["step_dt_s"] > 0.035).mean())

    # Spurious zeros: motor_pos == 0 surrounded by non-zero
    pos = df["motor_pos_rad"].values
    is_zero = pos == 0.0
    surrounded = is_zero & np.roll(~is_zero, 1) & np.roll(~is_zero, -1)
    spurious_zero_rows = list(np.where(surrounded)[0])

    return {
        "motor_frozen":        motor_frozen,
        "reached_target":      reached_target,
        "vision_lost_end_row": vision_lost_end_row,
        "vision_lost_early":   vision_lost_early,
        "run_duration_s":      run_duration_s,
        "jitter_fraction":     jitter_fraction,
        "spurious_zero_rows":  spurious_zero_rows,
    }


def assign_quality_tier(signals):
    """Map signals dict → quality tier string: A, B, C, or X."""
    if signals["motor_frozen"] or signals["run_duration_s"] < 5.0:
        return "X"
    if signals["reached_target"] and not signals["vision_lost_early"]:
        return "A"
    if not signals["motor_frozen"] and signals["jitter_fraction"] < 0.20:
        if signals["vision_lost_early"]:
            return "C"
        return "B"
    return "C"


def compute_usable_window(df, vision_lost_end_row):
    """Return (start_row, end_row) relative to df.index[0].

    start_row: first row after the motor starts moving (setpoint std > 0.01
    in a trailing 10-row window).
    end_row: vision_lost_end_row.
    """
    setpoint = df["motor_setpoint"].values
    start_row = 0
    for i in range(10, len(setpoint)):
        if setpoint[i - 10:i].std() > 0.01:
            start_row = i - 10
            break
    end_row = min(vision_lost_end_row, len(df) - 1)
    return int(start_row), int(end_row)
