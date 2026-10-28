"""
Shared utilities for SoFi tail system identification.

Conventions used throughout:
  - All physical quantities in SI (rad, Nm, kg, m, s)
  - Complex phasor X defined so that x(t) ≈ Re[X * exp(j*omega*t)]
  - Frequency response H = tau_phasor / theta_phasor
      real(H) = k_eff - I_eff * omega^2
      imag(H) = c_eff * omega
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dynamixel XL430 unit conversion constants
# ---------------------------------------------------------------------------
TICKS_PER_REV   = 4096
TICKS_TO_RAD    = 2 * np.pi / TICKS_PER_REV         # 1.534e-3 rad / tick
VEL_TO_RAD_S    = 0.229 * 2 * np.pi / 60            # 0.02399 rad/s per unit
CURRENT_UNIT_A  = 2.69e-3                             # A per raw Load unit
KT_NM_PER_A     = 1.5 / 1.193                        # Nm/A  (stall torque / stall current)
LOAD_TO_NM      = CURRENT_UNIT_A * KT_NM_PER_A      # 3.382e-3 Nm per raw Load unit

# ---------------------------------------------------------------------------
# SoFi geometry (from geometry/sofi_twin.py)
# ---------------------------------------------------------------------------
PULLEY_RADIUS_M = 0.0175          # m  (r17.5 dataset)

# Passive hinge joint heights: cylinder half-height × 2.
# From XML: joint{i} geom size = "0.002 {half_h}"
JOINT_HALF_HEIGHTS = np.array([0.0558, 0.0517, 0.0459, 0.0382, 0.0286])  # m
JOINT_HEIGHTS = 2 * JOINT_HALF_HEIGHTS                                     # full span in Z

# Connector plate thickness (PLA bending element at each joint pivot).
# From XML: connector box geom y half-size = 0.0005 m → full thickness 1 mm.
CONNECTOR_THICKNESS = 0.001   # m

# Connector span (the beam length from joint centre to adjacent body/segment edge).
# joint0: body-tail0 / 2 = 0.019 / 2 = 0.0095 m
# joint1-4: tail(i-1)-tail(i) / 2 = 0.022 / 2 = 0.011 m
CONNECTOR_SPANS = np.array([0.0095, 0.011, 0.011, 0.011, 0.011])   # m

# Tendon moment arms (Y-distance from joint axis to tendon eyelet).
# From XML: tail{i}tendon values in SYSTEMPARAMETERS.
TENDON_MOMENT_ARMS = np.array([0.0195, 0.0145, 0.01075, 0.007, 0.0045])   # m

# Nominal PLA Young's modulus for FDM printing (will be calibrated by measurement).
E_PLA_NOM = 2.7e9   # Pa

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "fishdeploy", "data", "03_sysid", "tail_benchmark", "air"
)

def find_latest_file(env: str, d_mm: float, freq_hz: float, amp_frac_pct: float) -> str | None:
    """
    Return path to the latest-timestamped CSV for a given (env, d, freq, amp) condition.
    amp_frac_pct is 20, 40, 60, 80, or 100.
    Returns None if no matching file exists.
    """
    pattern = os.path.join(
        DATA_DIR,
        f"*_r17.5_d{d_mm:04.1f}_Benchmark_{env}_{freq_hz:.1f}Hz_{amp_frac_pct:.1f}Amp.csv"
    )
    files = glob.glob(pattern)
    return max(files) if files else None  # lexicographic max = latest timestamp prefix


def list_available_conditions(env: str) -> pd.DataFrame:
    """Return a DataFrame listing all unique (d_mm, freq_hz, amp_pct) available for env."""
    pattern = os.path.join(DATA_DIR, f"*_r17.5_*_Benchmark_{env}_*.csv")
    rows = []
    for f in sorted(set(glob.glob(pattern))):
        meta, _ = load_csv(f)
        rows.append({
            "d_mm":    float(meta["Dataset"].iloc[0].split("_d")[1]),
            "freq_hz": float(meta["Freq"].iloc[0]),
            "amp_pct": float(meta["AmpFrac"].iloc[0] * 100),
            "travel_mm": float(meta["CmdAmpTravel_mm"].iloc[0]),
        })
    df = pd.DataFrame(rows).drop_duplicates().sort_values(["d_mm", "freq_hz", "amp_pct"])
    return df.reset_index(drop=True)


def load_csv(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a benchmark CSV.
    Row 0 is metadata (Env, Dataset, PulleyRadius_mm, Freq, AmpFrac, CmdAmpTravel_mm, CmdAmpTicks).
    Row 2+ is time-series data (Time(s), Position, GoalPosition, Velocity, Load).
    Returns (meta_df, data_df).
    """
    meta = pd.read_csv(filepath, nrows=1)
    data = pd.read_csv(filepath, header=2)
    return meta, data


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def to_si(data: pd.DataFrame, center_position: int | None = None) -> dict:
    """
    Convert raw Dynamixel columns to SI units.

    Returns dict with keys:
      t          [s]    - time array
      theta      [rad]  - motor angle (zero = mean position)
      theta_dot  [rad/s]
      tau        [Nm]   - motor torque (positive = in direction of positive theta)
      goal_theta [rad]  - commanded angle (zero = mean of GoalPosition)
    """
    t = data["Time (s)"].to_numpy(dtype=float)
    pos = data["Position"].to_numpy(dtype=float)
    vel = data["Velocity"].to_numpy(dtype=float)
    load = data["Load"].to_numpy(dtype=float)
    goal = data["GoalPosition"].to_numpy(dtype=float)

    center = float(np.mean(pos)) if center_position is None else float(center_position)

    return {
        "t":          t,
        "theta":      (pos - center) * TICKS_TO_RAD,
        "theta_dot":  vel * VEL_TO_RAD_S,
        "tau":        load * LOAD_TO_NM,
        "goal_theta": (goal - center) * TICKS_TO_RAD,
    }


def trim_transient(si: dict, n_cycles: float, freq_hz: float) -> dict:
    """
    Remove the first n_cycles cycles (startup transient).
    Returns a new dict with the same keys, trimmed to steady state.
    """
    t_start = n_cycles / freq_hz
    mask = si["t"] >= (si["t"][0] + t_start)
    return {k: v[mask] for k, v in si.items()}


def cycle_average(si: dict, freq_hz: float, n_points: int = 512) -> dict:
    """
    Synchronous cycle averaging: divide steady-state signal into individual complete
    cycles, interpolate each to a common uniform grid, and average.

    Why: random noise averages down as 1/√N (N = number of cycles). Deterministic
    signal (the sinusoidal response) stays intact. Cycle-to-cycle transients also
    cancel, giving a cleaner phasor estimate than fitting the full raw record.

    Returns the same dict structure as `si` but on a uniform grid of n_points
    spanning one period [0, T). Adds keys 'n_cycles' and 'period'.

    Requires at least 2 complete cycles; returns si unchanged if not enough data.
    """
    T = 1.0 / freq_hz
    t_rel = si["t"] - si["t"][0]
    n_cycles = int(t_rel[-1] / T)

    if n_cycles < 2:
        return si  # not enough cycles — fall back to single-window fit

    t_grid = np.linspace(0, T, n_points, endpoint=False)
    signal_keys = [k for k in si if k not in ("t",)]
    averaged = {k: np.zeros(n_points) for k in signal_keys}
    n_good = 0

    for i in range(n_cycles):
        t0 = i * T
        t1 = (i + 1) * T
        mask = (t_rel >= t0) & (t_rel < t1)
        if mask.sum() < 4:
            continue
        t_cyc = t_rel[mask] - t0         # local time within this cycle: [0, T)
        for k in signal_keys:
            averaged[k] += np.interp(t_grid, t_cyc, si[k][mask])
        n_good += 1

    for k in signal_keys:
        averaged[k] /= n_good

    averaged["t"]        = t_grid
    averaged["n_cycles"] = n_good
    averaged["period"]   = T
    return averaged


def extract_phasor(t: np.ndarray, x: np.ndarray, freq_hz: float) -> complex:
    """
    Least-squares extraction of the complex phasor X at frequency freq_hz.

    Fits x(t) ≈ A_cos*cos(ωt) + A_sin*sin(ωt), then returns:
        X = A_cos - j*A_sin
    so that x(t) ≈ Re[X * exp(j*ω*t)].

    Using exp(jωt) convention:
        x(t) = A_cos*cos(ωt) + A_sin*sin(ωt)
             = Re[(A_cos)*exp(jωt)] + Re[(-j*A_sin)*exp(jωt)]
             = Re[(A_cos - j*A_sin)*exp(jωt)]
    """
    omega = 2 * np.pi * freq_hz
    A = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    c, _, _, _ = np.linalg.lstsq(A, x, rcond=None)
    return complex(c[0], -c[1])  # A_cos - j*A_sin


def compute_H(si: dict, freq_hz: float) -> complex:
    """
    Compute the mechanical frequency response H(ω) = tau_phasor / theta_phasor.

    Under the exp(jωt) convention and Newton's law for a 1-DOF oscillator:
        τ = k*θ + c*θ̇ + I*θ̈
        τ(t) = Re[(k - I*ω² + j*c*ω) * Θ * exp(jωt)]
        H(ω) = T/Θ = (k - I*ω²) + j*(c*ω)

    real(H) = k_eff - I_eff * ω²   [stiffness - inertia contribution]
    imag(H) = c_eff * ω             [viscous damping contribution]
    """
    X_theta = extract_phasor(si["t"], si["theta"], freq_hz)
    X_tau   = extract_phasor(si["t"], si["tau"],   freq_hz)
    return X_tau / X_theta


def fit_spring_inertia(omega_sq: np.ndarray, real_H: np.ndarray) -> tuple[float, float]:
    """
    Fit real(H) = k_eff - I_eff * ω² to measured (ω², real(H)) pairs.
    Returns (k_eff [Nm/rad], I_eff [kg·m²]).
    """
    A = np.column_stack([np.ones_like(omega_sq), -omega_sq])
    c, _, _, _ = np.linalg.lstsq(A, real_H, rcond=None)
    return float(c[0]), float(c[1])


# ---------------------------------------------------------------------------
# Structural mechanics
# ---------------------------------------------------------------------------

def beam_theory_stiffness(E_pla: float = E_PLA_NOM) -> np.ndarray:
    """
    Individual passive joint stiffnesses k_joint_i [Nm/rad] from Euler-Bernoulli
    beam theory for a rectangular cross-section connector plate.

        k_i = E * I_z_i / L_i
        I_z_i = h_i * t^3 / 12      (second moment of area for lateral bending)

    h_i = JOINT_HEIGHTS[i]          (plate height in Z, the long direction)
    t   = CONNECTOR_THICKNESS       (plate thickness in Y, the bending direction)
    L_i = CONNECTOR_SPANS[i]        (half-span from joint centre to segment edge)

    Note: E_pla should be calibrated from the measured k_eff (see step1_stiffness.py).
    The default 2.7 GPa is a literature value for FDM PLA — actual value depends on
    print settings and likely falls in [1.5, 3.5] GPa.
    """
    I_z = JOINT_HEIGHTS * CONNECTOR_THICKNESS**3 / 12
    return E_pla * I_z / CONNECTOR_SPANS


def k_eff_from_joints(k_joints: np.ndarray,
                      r_pulley: float = PULLEY_RADIUS_M,
                      moment_arms: np.ndarray = TENDON_MOMENT_ARMS) -> float:
    """
    Effective stiffness at the motor shaft [Nm/rad], accounting for the tendon
    moment arm at each joint.

    Derivation (see ANALYSIS.md §3.2):
        φ_i = F_tendon * d_i / k_joint_i        (equilibrium at each joint)
        ΔL   = Σ φ_i * d_i                       (tendon length = sum of arc lengths)
        τ_motor = F_tendon * r_pulley
        θ_motor = ΔL / r_pulley

        k_eff = τ_motor / θ_motor = r² / Σ(d_i² / k_joint_i)
    """
    return r_pulley**2 / np.sum(moment_arms**2 / k_joints)


def scale_joints_to_keff(k_eff_target: float,
                          E_nom: float = E_PLA_NOM) -> tuple[np.ndarray, float]:
    """
    Scale beam-theory joint stiffnesses so that k_eff matches the measured value.
    Returns (k_joints [Nm/rad array], E_eff [Pa]).

    Since k_eff is linear in E (through all k_joint_i), the scale factor is:
        S = k_eff_target / k_eff_nominal
        E_eff = S * E_nom
    """
    k_nom = beam_theory_stiffness(E_nom)
    k_eff_nom = k_eff_from_joints(k_nom)
    S = k_eff_target / k_eff_nom
    return k_nom * S, E_nom * S


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    k = beam_theory_stiffness()
    k_eff = k_eff_from_joints(k)
    print("Beam-theory joint stiffnesses (E_PLA = 2.7 GPa):")
    for i, ki in enumerate(k):
        print(f"  k_joint{i} = {ki:.3f} Nm/rad")
    print(f"k_eff at motor shaft = {k_eff:.3f} Nm/rad")
    print(f"  → peak torque at 9mm (0.514 rad): {k_eff * 0.514:.3f} Nm  "
          f"(motor stall = 1.5 Nm, {k_eff * 0.514 / 1.5 * 100:.0f}% of stall)")
    print(f"\nLOAD_TO_NM = {LOAD_TO_NM:.4e} Nm/unit")
    print(f"TICKS_TO_RAD = {TICKS_TO_RAD:.4e} rad/tick")
