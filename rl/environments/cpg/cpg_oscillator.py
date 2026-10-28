import numpy as np

# Action-space limits: RL [-1, 1] → physical units.
# These are the TRAINING values of the deployed CPG policy e06g760j (git commit
# 733f45a, verified via W&B config). NOTE: the deploy side (sofi_control/rl_deploy/
# pi/policy_runner.py) uses a DIFFERENT scaling (0.30/0.15/12π/0.15) — a deploy-time
# change, NOT what the policy trained on. For sim reproduction use these training values.
K1_MAX           = 0.8        # rad   — first-harmonic amplitude
K2_MAX           = 0.4        # rad   — second-harmonic amplitude
OMEGA_MAX        = 8 * np.pi  # rad/s — 4 Hz swimming-frequency ceiling
THETA_CENTER_MAX = 0.8        # rad   — max static tail bias (clipped by motor_limit)
FILTER_TAU       = 0.1        # s     — CPG first-order filter time constant


class CPGOscillator:
    """First-order-filtered CPG producing a bounded motor-position setpoint.

    State: phase phi and six filtered parameters [k1c, k1s, k2c, k2s, omega, theta_center].
    Each step(desired) advances the filter, advances phase by omega*dt, and
    returns the clipped motor setpoint.

    All quantities are in applied units (radians, rad/s) — action scaling is
    the caller's job.
    """

    def __init__(self, dt_phys: float, tau: float, motor_limit: float):
        self.dt_phys = float(dt_phys)
        self.tau = float(tau)
        self.motor_limit = float(motor_limit)
        self._applied = np.zeros(6, dtype=np.float64)
        self._phase = 0.0

    def reset(self):
        self._applied[:] = 0.0
        self._phase = 0.0

    def step(self, desired: np.ndarray) -> float:
        alpha = self.dt_phys / self.tau
        self._applied += alpha * (np.asarray(desired, dtype=np.float64) - self._applied)

        k1c, k1s, k2c, k2s, omega, theta_c = self._applied
        self._phase = (self._phase + omega * self.dt_phys) % (2 * np.pi)

        theta_m = (theta_c
                   + k1c * np.cos(self._phase)     + k1s * np.sin(self._phase)
                   + k2c * np.cos(2 * self._phase) + k2s * np.sin(2 * self._phase))
        return float(np.clip(theta_m, -self.motor_limit, self.motor_limit))

    @property
    def applied_params(self) -> np.ndarray:
        return self._applied.copy()

    @property
    def phase(self) -> float:
        return self._phase


