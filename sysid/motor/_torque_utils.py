"""Torque conversion helpers — pure functions, no I/O."""

CURRENT_SCALE   = 0.00269   # A per raw Load unit
TORQUE_CONST    = 1.769     # Nm/A
STALL_TORQUE_NM = 2.3       # @ 12V (XW430-T200-R spec)
NO_LOAD_SPEED   = 5.55      # rad/s @ 12V (53 RPM)


def load_raw_to_torque(load_raw: float) -> float:
    return load_raw * CURRENT_SCALE * TORQUE_CONST


def voltage_scaled_speed_limit(volt: float) -> float:
    return NO_LOAD_SPEED * (volt / 12.0)


def is_saturated(load_raw: float, vel_rad_s: float, speed_limit: float) -> bool:
    return abs(load_raw) >= 800 or abs(vel_rad_s) >= 0.8 * speed_limit
