"""Unit tests for torque conversion and saturation logic."""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sysid.motor._torque_utils import (
    load_raw_to_torque,
    is_saturated,
    STALL_TORQUE_NM,
)


def test_zero_load_gives_zero_torque():
    assert load_raw_to_torque(0) == 0.0


def test_positive_load_positive_torque():
    assert load_raw_to_torque(100) > 0.0


def test_negative_load_negative_torque():
    assert load_raw_to_torque(-100) < 0.0


def test_torque_below_stall_ceiling():
    # Stall current = 2.3 Nm / 1.769 Nm/A = 1.3 A → 1.3 / 0.00269 ≈ 483 raw units
    # A raw value at stall should be within 5% of the stall ceiling
    assert abs(load_raw_to_torque(483)) < STALL_TORQUE_NM * 1.05  # 5% margin


def test_saturation_torque_threshold():
    assert is_saturated(load_raw=800, vel_rad_s=0.0, speed_limit=5.55) is True
    assert is_saturated(load_raw=799, vel_rad_s=0.0, speed_limit=5.55) is False


def test_saturation_speed_threshold():
    # 80% of speed limit triggers saturation
    assert is_saturated(load_raw=0, vel_rad_s=0.9 * 5.55, speed_limit=5.55) is True
    assert is_saturated(load_raw=0, vel_rad_s=0.7 * 5.55, speed_limit=5.55) is False


def test_voltage_scaled_speed_limit():
    # At 10V, limit should be < 5.55 rad/s
    from sysid.motor._torque_utils import voltage_scaled_speed_limit
    limit_10v = voltage_scaled_speed_limit(10.0)
    limit_12v = voltage_scaled_speed_limit(12.0)
    assert limit_10v < limit_12v
    assert abs(limit_12v - 5.55) < 0.01
