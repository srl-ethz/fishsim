#!/usr/bin/env python3
"""
FluidID — Stage 03: Windowed Multiple-Shooting Fluid Coefficient Optimizer
==========================================================================
Identifies 15 fluid coefficients of the single-fish MuJoCo model (geometry/SOFI_motorid.xml)
by minimizing windowed multiple-shooting trajectory drift on curated real swimming data.

Run from the fishCREATE/ directory:

  # Default: all starts_at_rest=True parts for training, rolling-start for validation
  python sysid/closed_loop/refit_fluid.py

  # Custom train/val split
  python sysid/closed_loop/refit_fluid.py \\
      --train "006_Normal Run__p1" "013_Near Miss Turn Run__p1" \\
      --val "002_Relief__p1"

  # CMA-ES with custom settings
  python sysid/closed_loop/refit_fluid.py --iter 500 --T 1.5 --yaw_weight 0.1

  # Multiple parallel runs (PID-namespaced temp files — safe)
  python sysid/closed_loop/refit_fluid.py --random_start --workers 8
"""

from __future__ import annotations

import argparse
import copy
import multiprocessing
import os
import signal
import sys
import time
import glob
from collections import defaultdict, deque
from datetime import datetime

import cma
import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(os.path.dirname(_SCRIPT_DIR))  # fishsim/

sys.path.insert(0, _REPO_ROOT)
from paths import CANONICAL as _CANONICAL  # noqa: E402

# The geometry generator is swappable via an env var (read at import time, so inherited by the
# forked workers); the default is the direct-action model whose fluid this fit identifies.
import importlib
_GENERATOR_MODULE = os.environ.get("FLUIDID_GENERATOR", "geometry.sofi_motorid")
_gen_mod = importlib.import_module(_GENERATOR_MODULE)
generate_xml, SYSTEMPARAMETERS = _gen_mod.generate_xml, _gen_mod.SYSTEMPARAMETERS
print(f"[refit_fluid] geometry generator: {_GENERATOR_MODULE}  "
      f"centerOfMass={SYSTEMPARAMETERS.get('centerOfMass')}")
from sysid.closed_loop import canonical as CUR

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Physics bounds for 15 fluid coefficients
# (head[0:5] / tail[5:10] / fin[10:15])
# Copied from digital_twin/orchestrators/opt_field2sim.py _PHYS_BOUNDS_15
# ---------------------------------------------------------------------------
_PHYS_BOUNDS_15 = [
    # head segment — mirrors 10-coeff head bounds (well-validated across E2–E7)
    (0.00, 0.15),  # c0  head blunt   — max ever seen: 0.056; streamlined head
    (0.00, 2.0),   # c1  head slender — 10-coeff consistently ~0.6; upper kept generous
    (0.00, 5.0),   # c2  head angular — E9 hit lower bound (1.048≈1.0); relaxed to 0.0
    (0.00, 3.0),   # c3  head kutta   — relaxed 2.0→3.0 (2026-06-18): 0612 fit hit 88%
    (0.00, 2.0),   # c4  head magnus  — relaxed 1.0→2.0 (2026-06-18): 0612 fit pinned at 1.0. Capped at 2 (not 4): physically suspect on a streamlined head + L2 prior tracks midpoint
    # tail body segment
    (0.00, 0.2),   # c5  tail blunt
    (0.00, 5.0),   # c6  tail slender
    (10.0, 20.0),  # c7  tail angular — LOWER bound 2→10 (2026-06-18): floor enforces FORWARD swimming. RL exploits low tail drag to swim BACKWARD (0608 c7=3.5→−1.4m). Lowest CONFIRMED-forward is 0612 c7=10.9→+1.0m (0618 c7=19.9→+1.2m); threshold is far below 10.9, so floor=10 is very likely safe — but not yet confirmed at exactly 10.
    (0.00, 12.0),  # c8  tail kutta   — same instability ceiling as 10-coeff
    (0.00, 6.0),   # c9  tail magnus  — mirrors 10-coeff [0,6]
    # fin segment
    (0.00, 0.4),   # c10 fin blunt    — relaxed 0.2→0.4 (2026-06-18): 0612 fit hit 91%
    (0.00, 5.0),   # c11 fin slender
    (2.00, 20.0),  # c12 fin angular  — lowered from 40 back to 20
    (0.00, 17.0),  # c13 fin kutta    — relaxed 12→17 (2026-06-18): pinned in BOTH 0608/0612 fits. WARNING past MuJoCo instability wall — watch for NaN
    (0.00, 15.0),  # c14 fin magnus   — relaxed 10→15 (2026-06-18): 0612 fit pinned at 10.0
]

# Neutral priors for L2 regularization: midpoint of each bound.
# Normalisation factor: bound range (so a unit deviation in any coef contributes equally).
_L2_PRIORS = np.array([(lo + hi) / 2.0 for lo, hi in _PHYS_BOUNDS_15])
_L2_RANGES = np.array([max(hi - lo, 1e-6)  for lo, hi in _PHYS_BOUNDS_15])


def _l2_penalty(coef: np.ndarray) -> float:
    """Normalised L2 distance from bound midpoints (dimensionless, [0, 0.25])."""
    return float(np.sum(((coef - _L2_PRIORS) / _L2_RANGES) ** 2))


# ---------------------------------------------------------------------------
# Joint helper
# ---------------------------------------------------------------------------

def _named_joints(model: mujoco.MjModel, fish_id: int = 0) -> dict:
    """Return {joint_name: (qpos_adr, dof_adr)} for all known joints of fish_id."""
    candidates = [
        "headX", "headY", "headZ",
        "headRollX", "headRollY", "headRollZ",
        "motor",
        "joint0", "joint1", "joint2", "joint3", "joint4",
    ]
    out = {}
    for s in candidates:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{fish_id}")
        if jid == -1:
            continue
        out[s] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
    return out


# ---------------------------------------------------------------------------
# XML generation helper
# ---------------------------------------------------------------------------

def make_xml(coef15: list, out_path: str) -> None:
    """Write a SOFI_motorid.xml with the given 15 fluid coefficients."""
    p = copy.deepcopy(SYSTEMPARAMETERS)
    p["fluidCoef"] = [float(c) for c in coef15]
    p["filename"]  = out_path
    generate_xml(p)


def _tmp_xml_path() -> str:
    """PID-namespaced temp XML path under geometry/ (meshes are relative to that dir).

    Temp files are placed next to SOFI_motorid.xml so that MuJoCo can resolve mesh paths.
    Files are named candidate_{pid}.xml to avoid collisions across parallel runs.
    """
    geo_dir = os.path.join(_REPO_ROOT, "geometry")
    pid = multiprocessing.current_process().pid or os.getpid()
    return os.path.join(geo_dir, f"candidate_{pid}.xml")


# ---------------------------------------------------------------------------
# Backward-swimming penalty (optional, --backward_penalty)
# ---------------------------------------------------------------------------
# Box bounds cannot forbid backward swimming (it's a multi-coef, gait-dependent
# head-region effect — confirmed by causal sweeps). Instead we replay captured
# real backward gaits through the candidate fluid coeffs + the SWEEP's motor model
# and penalise any net-backward motion. No reward for forward/still — penalty only.
_BWD_WEIGHT = 0.0
_BWD_EPISODES: list = []   # each: dict(actions, qpos, qvel)
# Motor model the new sweep will use (re-ID 2026-06-19): kv=0, joint damping b=0.48.
_BWD_MOTOR = dict(kp=5.41, kv=0.0, arm=0.0113, b=0.48, dt=14)


def _load_backward_episodes(dir_path: str) -> list:
    eps = []
    for f in sorted(glob.glob(os.path.join(dir_path, "*.npz"))):
        z = np.load(f)
        eps.append(dict(actions=z["actions"], qpos=z["init_qpos"], qvel=z["init_qvel"]))
    return eps


def _backward_penalty(coef) -> float:
    """λ · mean over captured gaits of max(0, backward_distance) [m]. 0 if forward/still."""
    if _BWD_WEIGHT <= 0.0 or not _BWD_EPISODES:
        return 0.0
    xml = _tmp_xml_path()
    try:
        make_xml(coef, xml)
        m = mujoco.MjModel.from_xml_path(xml)
    except Exception:
        return 0.0
    finally:
        try:
            os.remove(xml)
        except OSError:
            pass
    aid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_0")
    vadr = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "motor_0")])
    sid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "COM_0")
    hRZ  = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "headRollZ_0")])
    # Override the XML motor with the sweep's motor model (forcerange already 2.3).
    m.actuator_gainprm[aid][0] = _BWD_MOTOR["kp"]
    m.actuator_biasprm[aid][1] = -_BWD_MOTOR["kp"]
    m.actuator_biasprm[aid][2] = -_BWD_MOTOR["kv"]
    m.dof_armature[vadr] = _BWD_MOTOR["arm"]
    m.dof_damping[vadr]  = _BWD_MOTOR["b"]
    dt = int(_BWD_MOTOR["dt"])
    pens = []
    for ep in _BWD_EPISODES:
        d = mujoco.MjData(m)
        d.qpos[:] = ep["qpos"]; d.qvel[:] = ep["qvel"]
        mujoco.mj_forward(m, d)
        buf = deque([0.0] * (dt + 1), maxlen=dt + 1)
        com0 = d.site_xpos[sid][:2].copy(); h0 = float(ep["qpos"][hRZ])
        bad = False
        for a in ep["actions"]:
            th = float(np.clip(a, -1.0, 1.0))   # ctrlrange = 1.0
            for _ in range(20):
                buf.append(th); d.ctrl[aid] = buf[0]; mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos[0]):
                bad = True; break
        if bad:
            continue   # unstable coeff already gets 1e6 from the tracking loss
        ang = h0 + np.pi
        net_fwd = float((d.site_xpos[sid][:2] - com0) @ np.array([np.cos(ang), np.sin(ang)]))
        pens.append(max(0.0, -net_fwd))   # backward distance; 0 if forward/still
    return _BWD_WEIGHT * float(np.mean(pens)) if pens else 0.0


# ---------------------------------------------------------------------------
# Multiple-shooting simulation
# ---------------------------------------------------------------------------

_DT = 0.001  # canonical CSV timestep

# Rolling-start preamble (applies only to starts_at_rest=False segments):
#
#   [0 ms ──── 300 ms]  SG context window — used to estimate velocity at t=300ms.
#                        The SG derivative is most accurate at the CENTRE of its
#                        window, so we use [0, 600ms] data and evaluate at t=300ms.
#                        gyro_z[300ms] gives the direct IMU omega at that instant.
#   [300 ms ── 800 ms]  Warm-up — passive tail joints settle (τ_mech ≈ 60 ms,
#                        500ms = 8× margin).  Re-anchor pos+yaw at t=800ms.
#   [800 ms ──────────  Scored windows begin.
#
# Total preamble: 800 ms.  Rest-start segments have exact zero ICs and no preamble.

_SG_STEPS     = 300   # 0.3 s  — SG context; sim starts at this index
_WARMUP_STEPS = 500   # 0.5 s  — tail-settling warm-up after SG start point


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _sg_velocity(pos: np.ndarray, window_ms: int = 101, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay derivative of a 1 kHz position signal [m/s].

    Uses the first `window_ms` samples to fit; the returned value at index 0 is a
    clean velocity estimate at t=0, replacing the noisy finite-difference gradient.
    window_ms must be odd and ≤ len(pos).
    """
    from scipy.signal import savgol_filter
    n = min(len(pos), window_ms)
    if n < poly + 2:
        return np.gradient(pos[:n], _DT)
    # Ensure odd window
    w = n if n % 2 == 1 else n - 1
    return savgol_filter(pos[:w], window_length=w, polyorder=poly,
                         deriv=1, delta=_DT)


def simulate_segment(
    xml_path: str,
    df: pd.DataFrame,
    starts_at_rest: bool,
    T_window: float = 2.0,
    yaw_weight: float = 0.1,
    whole_traj: bool = False,
) -> float:
    """Evaluate fluid coefficients against one canonical segment.

    Two modes controlled by ``whole_traj``:

    **Windowed mode** (``whole_traj=False``, default):
        Runs T_window-second chunks, re-anchoring pos+yaw to real values at each
        boundary.  Loss = mean squared displacement error per window.  IC errors
        are reset, so long-horizon drift is hidden.

    **Whole-trajectory mode** (``whole_traj=True``):
        Runs the full segment from the initial IC with NO re-anchoring.  Loss =
        mean squared absolute position (and yaw) error at every step.  Accumulated
        drift is fully visible; the optimizer must get the dynamics right at all
        timescales.  Requires good ICs — exact for rest-starts, SG+warm-up for
        rolling starts.

    Parameters
    ----------
    xml_path : str
        Path to the (per-process temp) MuJoCo XML.
    df : pd.DataFrame
        Pre-loaded canonical CSV.
    starts_at_rest : bool
        If True, body velocity ICs are zero (exact known state).
        If False, use SG velocity estimate + _WARMUP_STEPS warm-up before scoring.
    T_window : float
        Scoring window length [s] — used only when whole_traj=False.
    yaw_weight : float
        Weight on yaw error squared in total loss.
    whole_traj : bool
        If True, skip windowing and evaluate the entire trajectory open-loop.

    Returns
    -------
    float
        Mean per-step (whole_traj) or per-window (windowed) loss, or 1e6 on failure.
    """
    from scipy.signal import savgol_filter  # lazy import — only needed here

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    joints = _named_joints(model, fish_id=0)
    if "headX" not in joints or "headY" not in joints or "headRollZ" not in joints or "motor" not in joints:
        return 1e6

    hX_q,  hX_v  = joints["headX"]
    hY_q,  hY_v  = joints["headY"]
    hRZ_q, hRZ_v = joints["headRollZ"]
    mot_q, mot_v = joints["motor"]

    T_steps = int(round(T_window / _DT))
    N = len(df)

    # Pre-extract arrays for speed
    pos_x  = df["pos_x"].values
    pos_y  = df["pos_y"].values
    yaw    = df["yaw"].values
    gyro_z = df["gyro_z"].values   # direct IMU yaw rate — used for rolling-start omega IC
    mpos   = df["motor_pos_meas"].values
    mvel   = df["motor_vel"].values

    # --- Set initial conditions ---
    mujoco.mj_resetData(model, data)

    if starts_at_rest:
        # Spec §3.3: pose measured, velocity 0, tail 0 — exact known ICs.
        data.qpos[hX_q]  = pos_x[0]
        data.qpos[hY_q]  = pos_y[0]
        data.qpos[hRZ_q] = yaw[0]
        data.qpos[mot_q] = mpos[0]
        data.qvel[hX_v]  = 0.0
        data.qvel[hY_v]  = 0.0
        data.qvel[hRZ_v] = 0.0
        data.qvel[mot_v] = 0.0
        data.ctrl[0]     = mpos[0]
        mujoco.mj_forward(model, data)
        w_start = 0

    else:
        # Rolling-start: start simulation at t = _SG_STEPS (300 ms).
        #
        # Velocity ICs at t=300ms:
        #   vx, vy — SG filter centred at t=300ms (half-window each side) evaluated
        #            at the centre gives the least-biased derivative estimate.
        #   omega  — gyro_z[300ms]: direct IMU reading, zero differentiation noise.
        #   pos, yaw — directly from canonical CSV at t=300ms.
        #
        # Then 500ms warm-up (motor teleport, body+tail evolve freely) so the
        # passive tail joints reach a physically consistent shape before scoring.
        sg_center = min(_SG_STEPS, N - 1)
        sg_half   = _SG_STEPS // 2           # 150 ms each side
        sg_left   = max(0, sg_center - sg_half)
        sg_right  = min(N, sg_center + sg_half + 1)
        sg_len    = sg_right - sg_left
        sg_len    = sg_len if sg_len % 2 == 1 else sg_len - 1  # must be odd

        if sg_len >= 5:
            vx0 = float(savgol_filter(
                pos_x[sg_left:sg_left + sg_len], sg_len, 3,
                deriv=1, delta=_DT)[sg_center - sg_left])
            vy0 = float(savgol_filter(
                pos_y[sg_left:sg_left + sg_len], sg_len, 3,
                deriv=1, delta=_DT)[sg_center - sg_left])
        else:
            vx0 = float(np.gradient(pos_x[sg_left:sg_right], _DT)[sg_center - sg_left])
            vy0 = float(np.gradient(pos_y[sg_left:sg_right], _DT)[sg_center - sg_left])

        data.qpos[hX_q]  = pos_x[sg_center]
        data.qpos[hY_q]  = pos_y[sg_center]
        data.qpos[hRZ_q] = yaw[sg_center]
        data.qpos[mot_q] = mpos[sg_center]
        data.qvel[hX_v]  = vx0
        data.qvel[hY_v]  = vy0
        data.qvel[hRZ_v] = float(gyro_z[sg_center])   # direct IMU at t=300ms
        data.qvel[mot_v] = float(mvel[sg_center])
        data.ctrl[0]     = mpos[sg_center]
        mujoco.mj_forward(model, data)

        # Warm-up: 500ms of free dynamics from t=300ms.
        warmup_end = min(sg_center + _WARMUP_STEPS, N - 1)
        for i in range(sg_center, warmup_end):
            data.qpos[mot_q] = mpos[i]
            data.qvel[mot_v] = mvel[i]
            data.ctrl[0]     = mpos[i]
            mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos[hX_q]) and np.isfinite(data.qpos[hY_q])):
                return 1e6

        # Re-anchor pos+yaw to real at t=800ms; carry world-frame velocity + tail.
        data.qpos[hX_q]  = pos_x[warmup_end]
        data.qpos[hY_q]  = pos_y[warmup_end]
        data.qpos[hRZ_q] = yaw[warmup_end]
        mujoco.mj_forward(model, data)
        w_start = warmup_end

    # ------------------------------------------------------------------
    # Whole-trajectory mode: run open-loop, no re-anchoring
    # ------------------------------------------------------------------
    if whole_traj:
        total_sq = 0.0
        n_steps  = 0
        for i in range(w_start, N - 1):
            data.qpos[mot_q] = mpos[i]
            data.qvel[mot_v] = mvel[i]
            data.ctrl[0]     = mpos[i]
            mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos[hX_q]) and np.isfinite(data.qpos[hY_q])):
                return 1e6
            err_x   = data.qpos[hX_q]  - pos_x[i + 1]
            err_y   = data.qpos[hY_q]  - pos_y[i + 1]
            err_yaw = _wrap_angle(data.qpos[hRZ_q] - yaw[i + 1])
            total_sq += err_x**2 + err_y**2 + yaw_weight * err_yaw**2
            n_steps  += 1
        return total_sq / n_steps if n_steps > 0 else 1e6

    # ------------------------------------------------------------------
    # Windowed multiple-shooting mode (default)
    # ------------------------------------------------------------------
    total_loss = 0.0
    n_windows  = 0

    while w_start + T_steps < N:
        w_end = w_start + T_steps

        # Record real and sim pose at window start
        rx0   = pos_x[w_start]
        ry0   = pos_y[w_start]
        ryaw0 = yaw[w_start]
        sx0   = float(data.qpos[hX_q])
        sy0   = float(data.qpos[hY_q])
        syaw0 = float(data.qpos[hRZ_q])

        # --- Run T_steps with motor teleport ---
        nan_hit = False
        for i in range(T_steps):
            idx = w_start + i
            data.qpos[mot_q] = mpos[idx]
            data.qvel[mot_v] = mvel[idx]
            data.ctrl[0]     = mpos[idx]
            mujoco.mj_step(model, data)
            if not (np.isfinite(data.qpos[hX_q]) and np.isfinite(data.qpos[hY_q])):
                nan_hit = True
                break

        if nan_hit:
            return 1e6

        # Real displacement
        drx   = pos_x[w_end] - rx0
        dry   = pos_y[w_end] - ry0
        dryaw = _wrap_angle(yaw[w_end] - ryaw0)

        # Sim displacement (translation-invariant: subtract window-start pose)
        sx_end   = float(data.qpos[hX_q])
        sy_end   = float(data.qpos[hY_q])
        syaw_end = float(data.qpos[hRZ_q])
        dsx   = sx_end  - sx0
        dsy   = sy_end  - sy0
        dsyaw = _wrap_angle(syaw_end - syaw0)

        err_x   = dsx   - drx
        err_y   = dsy   - dry
        err_yaw = _wrap_angle(dsyaw - dryaw)

        total_loss += err_x**2 + err_y**2 + yaw_weight * err_yaw**2
        n_windows  += 1

        # --- Re-anchor: snap pos+yaw to real, carry world-frame velocity + tail state.
        # Translation+rotation invariance: fluid forces don't depend on absolute pose,
        # so snapping pos/yaw injects zero force disturbance. Velocity carries forward;
        # the brief velocity-yaw inconsistency (O(v·sin(Δyaw))) dissipates in ~0.1–0.3 s.
        data.qpos[hX_q]  = pos_x[w_end]
        data.qpos[hY_q]  = pos_y[w_end]
        data.qpos[hRZ_q] = yaw[w_end]
        mujoco.mj_forward(model, data)

        w_start = w_end

    if n_windows == 0:
        return 1e6

    return total_loss / n_windows


# ---------------------------------------------------------------------------
# Worker function (one coef × one CSV)
# ---------------------------------------------------------------------------

def _worker_evaluate_single(
    job_id: int,
    coef15: list,
    csv_path: str,
    starts_at_rest: bool,
    T_window: float,
    yaw_weight: float,
    whole_traj: bool = False,
) -> tuple:
    """Worker: generate XML, load CSV, simulate, return (job_id, loss, detail)."""
    xml_path = _tmp_xml_path()
    try:
        make_xml(coef15, xml_path)
    except Exception as e:
        return job_id, 1e6, f"XML_FAIL({type(e).__name__}: {e})"

    try:
        df = pd.read_csv(csv_path)
        loss = simulate_segment(xml_path, df, starts_at_rest, T_window, yaw_weight, whole_traj)
        if not np.isfinite(loss):
            loss = 1e6
        detail = f"loss={loss:.5f}"
    except Exception as e:
        loss   = 1e6
        detail = f"SIM_FAIL({type(e).__name__}: {e})"
    finally:
        try:
            os.remove(xml_path)
        except OSError:
            pass

    return job_id, loss, detail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_native(obj):
    """Convert numpy scalars/arrays to plain Python for YAML serialisation."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _load_parts(layer2_path: str) -> list[dict]:
    """Return flattened list of accepted parts from a layer-2 manifest."""
    l2 = CUR.load_layer2(layer2_path)
    return CUR.accepted_parts(l2)


def _canonical_csv(part_id: str, canonical_dir: str) -> str:
    """Resolve a part_id to its canonical CSV path."""
    # part_id format: "006_Normal Run__p1" → file "006_Normal Run__p1.csv"
    return os.path.join(canonical_dir, f"{part_id}.csv")


_GLOBAL_POOL = None


def _signal_handler(sig, frame):
    print("\n\n  [!] Interrupt received — shutting down workers...")
    if _GLOBAL_POOL is not None:
        _GLOBAL_POOL.terminate()
        _GLOBAL_POOL.join()
    raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _GLOBAL_POOL
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--train", nargs="+", default=None, metavar="PART_ID",
        help="Part IDs to use for training (e.g. '006_Normal Run__p1'). "
             "Default: all starts_at_rest=True parts.",
    )
    parser.add_argument(
        "--val", nargs="+", default=None, metavar="PART_ID",
        help="Part IDs to use for validation. "
             "Default: all starts_at_rest=False parts.",
    )
    parser.add_argument("--iter",       type=int,   default=500,  help="CMA-ES max generations.")
    parser.add_argument("--T",          type=float, default=2.0,  help="Scoring window length [s] — only used in windowed mode (ignored when --whole_traj).")
    parser.add_argument(
        "--whole_traj", action="store_true",
        help="Evaluate full trajectory open-loop (no re-anchoring). "
             "Exposes long-horizon drift. Recommended for rest-start segments.",
    )
    parser.add_argument("--yaw_weight", type=float, default=0.1,  help="Weight on yaw delta error.")
    parser.add_argument("--l2_weight",  type=float, default=0.001,
                        help="L2 regularization weight toward bound midpoints (spec §4.4). "
                             "Normalised by bound range so all 15 coefs contribute equally. "
                             "Default 0.001 — light enough not to dominate trajectory loss.")
    parser.add_argument(
        "--sigma0", type=float, default=0.5,
        help="CMA-ES initial step size (default 0.5; tighter than field2sim since bounds are known).",
    )
    parser.add_argument("--popsize",    type=int,   default=None, help="CMA-ES population size.")
    parser.add_argument(
        "--random_start", action="store_true",
        help="Initialise CMA-ES from a random point within bounds.",
    )
    parser.add_argument(
        "--warm_start_checkpoint", type=str, default=None,
        help="Path to a checkpoint.yml; seed CMA-ES x0 from its best_coef (re-seeds the "
             "search MEAN only — the adapted covariance is NOT restored, so pair with a "
             "smaller --sigma0 for local refinement). Takes precedence over --random_start.",
    )
    parser.add_argument(
        "--warm_start_history", type=str, default=None,
        help="Path to a checkpoint.yml; reconstruct the search distribution from ALL its "
             "sampled points. x0=best_coef and per-coordinate CMA_stds are set to the elite "
             "sample std (see --warm_elite_frac), restoring the learned per-dimension scale "
             "that best_coef alone loses. With this, --sigma0 multiplies the elite width "
             "(use ~2.5 to re-explore, ~1.0 to polish). Takes precedence over the others.",
    )
    parser.add_argument(
        "--warm_elite_frac", type=float, default=0.25,
        help="Fraction of sampled points (lowest-loss) used to estimate the warm-start "
             "per-coordinate stds for --warm_start_history (default 0.25).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes. Default: all CPU cores.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Override output directory (default: outputs/closed_loop/{timestamp}/).",
    )
    parser.add_argument(
        "--layer2",     type=str,
        default=str(_CANONICAL / "layer2_parts.yml"),
        help="Path to layer2_parts.yml.",
    )
    parser.add_argument(
        "--canonical_dir", type=str,
        default=str(_CANONICAL),
        help="Directory containing canonical CSV files.",
    )
    parser.add_argument(
        "--backward_penalty", type=float, default=0.0,
        help="Weight λ on a backward-swimming penalty (0=off). Replays captured backward "
             "gaits through the candidate fluid + the sweep's motor model; penalises net-"
             "backward motion only (no reward for forward/still). Forbids backward directly.",
    )
    parser.add_argument(
        "--backward_episodes_dir", type=str,
        default=os.path.join(_SCRIPT_DIR, "data", "backward_episodes"),   # three captured backward gaits of the unconstrained DA policy
        help="Directory of *.npz backward gaits (actions, init_qpos, init_qvel).",
    )
    parser.add_argument(
        "--c7_lower", type=float, default=None,
        help="Override the TAIL_angular (c7) lower bound (current default 10.0). The c7 floor "
             "does NOT prevent backward swimming (proven); set ~2 to restore fitting freedom.",
    )

    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(_REPO_ROOT, "outputs", "closed_loop", ts)
    os.makedirs(out_dir, exist_ok=True)

    # Optional c7 (tail_angular) lower-bound override.
    if args.c7_lower is not None:
        globals()["_PHYS_BOUNDS_15"][7] = (float(args.c7_lower), _PHYS_BOUNDS_15[7][1])
        globals()["_L2_PRIORS"] = np.array([(lo + hi) / 2.0 for lo, hi in _PHYS_BOUNDS_15])
        globals()["_L2_RANGES"] = np.array([max(hi - lo, 1e-6) for lo, hi in _PHYS_BOUNDS_15])
        print(f"  [bounds] c7 (tail_angular) lower bound -> {args.c7_lower}")

    # Optional backward-swimming penalty.
    globals()["_BWD_WEIGHT"] = float(args.backward_penalty)
    if args.backward_penalty > 0.0:
        globals()["_BWD_EPISODES"] = _load_backward_episodes(args.backward_episodes_dir)
        if not _BWD_EPISODES:
            raise SystemExit(f"--backward_penalty set but no *.npz in {args.backward_episodes_dir}")
        print(f"  [backward] penalty λ={args.backward_penalty} on {len(_BWD_EPISODES)} gait(s) "
              f"(motor kv=0,b=0.48); penalises net-backward only.")

    # ------------------------------------------------------------------
    # Build train / val part lists
    # ------------------------------------------------------------------
    all_parts = _load_parts(args.layer2)
    all_ids   = {p["part_id"] for p in all_parts}

    def _resolve_parts(ids_arg, default_filter):
        if ids_arg is not None:
            missing = [pid for pid in ids_arg if pid not in all_ids]
            if missing:
                raise ValueError(
                    f"Unknown part IDs: {missing}\n"
                    f"Available: {sorted(all_ids)}"
                )
            return [p for p in all_parts if p["part_id"] in ids_arg]
        return [p for p in all_parts if default_filter(p)]

    train_parts = _resolve_parts(args.train, lambda p: p["starts_at_rest"])
    val_parts   = _resolve_parts(args.val,   lambda p: not p["starts_at_rest"])

    # Verify CSV files exist
    for p in train_parts + val_parts:
        csv = _canonical_csv(p["part_id"], args.canonical_dir)
        if not os.path.exists(csv):
            raise FileNotFoundError(
                f"Canonical CSV not found: {csv}\n"
                f"Run fishdeploy/analysis/canonical/convert_segments.py first."
            )

    print(f"\n{'='*66}")
    print(f"  FluidID 03 — Windowed Multiple-Shooting CMA-ES Optimizer")
    print(f"{'='*66}")
    print(f"  Train parts ({len(train_parts)}):")
    for p in train_parts:
        print(f"    {p['part_id']}  [starts_at_rest={p['starts_at_rest']}]")
    if val_parts:
        print(f"  Val parts ({len(val_parts)}):")
        for p in val_parts:
            print(f"    {p['part_id']}  [starts_at_rest={p['starts_at_rest']}]")
    else:
        print(f"  Val parts: none")
    if args.whole_traj:
        print(f"  Mode       = WHOLE-TRAJECTORY (no re-anchoring, open-loop drift visible)")
    else:
        print(f"  Mode       = windowed  T_window={args.T} s  ({int(round(args.T / _DT))} steps)")
    print(f"  yaw_weight = {args.yaw_weight}")
    print(f"  sigma0     = {args.sigma0}")
    print(f"  max iter   = {args.iter}")
    print(f"  Output dir : {out_dir}")
    print(f"{'='*66}\n")

    # ------------------------------------------------------------------
    # Physics bounds
    # ------------------------------------------------------------------
    lb = [b[0] for b in _PHYS_BOUNDS_15]
    ub = [b[1] for b in _PHYS_BOUNDS_15]

    # ------------------------------------------------------------------
    # CMA-ES initial point
    # ------------------------------------------------------------------
    warm_stds = None   # per-coordinate CMA_stds reconstructed from sample history
    if args.warm_start_history:
        with open(args.warm_start_history) as _f:
            _ckpt = yaml.safe_load(_f)
        x0 = [float(c) for c in _ckpt["best_coef"]]
        if len(x0) != 15:
            raise ValueError(f"checkpoint best_coef has {len(x0)} coefs, expected 15")
        _hist = _ckpt.get("eval_history", [])
        _coef = np.array([e["coef"] for e in _hist], dtype=float)
        _loss = np.array([e["loss"] if e["loss"] is not None else 1e6 for e in _hist])
        _k = max(15, int(args.warm_elite_frac * len(_loss)))
        _elite = _coef[np.argsort(_loss)[:_k]]
        _std = _elite.std(axis=0)
        # floor each std at 1% of the bound range so pinned dims keep minimal exploration
        _floor = np.array([0.01 * (ub[i] - lb[i]) for i in range(15)])
        warm_stds = np.maximum(_std, _floor)
        print(f"  [CMA-ES] WARM START (history) from {args.warm_start_history}")
        print(f"           x0=best_coef (best_loss={_ckpt.get('best_loss')}); "
              f"CMA_stds from elite {_k}/{len(_loss)} pts (frac={args.warm_elite_frac}).")
        print(f"           per-dim CMA_stds = {[f'{v:.4f}' for v in warm_stds]}")
    elif args.warm_start_checkpoint:
        with open(args.warm_start_checkpoint) as _f:
            _ckpt = yaml.safe_load(_f)
        x0 = [float(c) for c in _ckpt["best_coef"]]
        if len(x0) != 15:
            raise ValueError(f"checkpoint best_coef has {len(x0)} coefs, expected 15")
        print(f"  [CMA-ES] WARM START from {args.warm_start_checkpoint}")
        print(f"           seeded x0 from best_coef (best_loss={_ckpt.get('best_loss')}); "
              f"covariance NOT restored.")
    elif args.random_start:
        x0 = [np.random.uniform(lb[i], ub[i]) for i in range(15)]
        print(f"  [CMA-ES] Initialising with RANDOM x0 within bounds.")
    else:
        # Use current SYSTEMPARAMETERS fluidCoef as warm start (already 15 coefs)
        x0 = [float(c) for c in SYSTEMPARAMETERS["fluidCoef"]]
        if len(x0) != 15:
            x0 = [(lb[i] + ub[i]) / 2 for i in range(15)]
        print(f"  [CMA-ES] Initialising with SYSTEMPARAMETERS fluidCoef as warm start.")

    # Clip x0 to bounds
    x0 = [float(np.clip(x0[i], lb[i], ub[i])) for i in range(15)]

    popsize = args.popsize or (4 + int(3 * np.log(15)))
    print(f"  [CMA-ES] sigma0={args.sigma0}, popsize={popsize}, maxiter={args.iter}")
    print(f"  [CMA-ES] x0 = {[f'{v:.4f}' for v in x0]}\n")

    # ------------------------------------------------------------------
    # Train CSV lookup
    # ------------------------------------------------------------------
    train_csvs = [
        (_canonical_csv(p["part_id"], args.canonical_dir), p["starts_at_rest"])
        for p in train_parts
    ]
    n_csvs = len(train_csvs)

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------
    best_loss    = [np.inf]
    best_coef    = [list(x0)]
    loss_history = []
    eval_count   = [0]
    eval_history = []
    t_start      = time.time()

    checkpoint_path = os.path.join(out_dir, "checkpoint.yml")
    conv_plot_path  = os.path.join(out_dir, "convergence.png")

    print(f"  [checkpoint] {checkpoint_path}")
    print(f"  [plot]       {conv_plot_path}\n")

    def _save_checkpoint():
        if best_coef[0] is None:
            return
        ckpt = {
            "timestamp":    ts,
            "is_checkpoint": True,
            "best_loss":    float(best_loss[0]),
            "best_coef":    [float(c) for c in best_coef[0]],
            "train_parts":  [p["part_id"] for p in train_parts],
            "val_parts":    [p["part_id"] for p in val_parts],
            "config": {
                "T_window":   args.T,
                "yaw_weight": args.yaw_weight,
                "sigma0":     args.sigma0,
                "popsize":    popsize,
                "iter":       args.iter,
                "random_start": args.random_start,
                "whole_traj": args.whole_traj,
            },
            "eval_history": to_native(eval_history),
        }
        tmp = checkpoint_path + ".tmp"
        try:
            with open(tmp, "w") as fh:
                yaml.safe_dump(ckpt, fh)
            os.replace(tmp, checkpoint_path)
        except Exception as e:
            tqdm.write(f"  [warn] Checkpoint save failed: {e}")

    _live_fig, _live_ax = None, None
    try:
        _live_fig, _live_ax = plt.subplots(figsize=(7, 4))
    except Exception:
        pass

    def _update_plot():
        if _live_fig is None or not loss_history:
            return
        _live_ax.clear()
        evals = list(range(1, len(loss_history) + 1))
        best_so_far = np.minimum.accumulate(loss_history)
        _live_ax.plot(evals, loss_history, color="steelblue", alpha=0.35, lw=0.8, label="Eval")
        _live_ax.plot(
            evals, best_so_far, color="crimson", lw=1.5,
            label=f"Best = {best_loss[0]:.5f}",
        )
        _live_ax.set_xlabel("Evaluation")
        _live_ax.set_ylabel("Loss")
        _live_ax.set_title(f"FluidID CMA-ES | n_coef=15 | train={n_csvs}")
        _live_ax.legend(loc="upper right", fontsize=7)
        _live_ax.grid(True, alpha=0.3)
        _live_fig.tight_layout()
        try:
            _live_fig.savefig(conv_plot_path, dpi=150, bbox_inches="tight")
        except Exception:
            pass
        _save_checkpoint()

    # ------------------------------------------------------------------
    # CMA-ES optimisation
    # ------------------------------------------------------------------
    opts = cma.CMAOptions()
    opts["bounds"]  = [lb, ub]
    opts["maxiter"] = args.iter
    opts["popsize"] = popsize
    opts["seed"]    = 42
    opts["verbose"] = -9  # suppress internal cma output
    if warm_stds is not None:
        # per-coordinate initial std = sigma0 * CMA_stds[i] — restores the learned
        # per-dimension scale from the sample history (covariance diagonal).
        opts["CMA_stds"] = list(warm_stds)

    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)

    n_workers = args.workers or multiprocessing.cpu_count()
    print(f"  Workers: {n_workers} / {multiprocessing.cpu_count()} CPU cores\n")
    _GLOBAL_POOL = multiprocessing.Pool(processes=n_workers)

    pbar = tqdm(
        total=args.iter, desc="CMA-ES", unit="gen",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    try:
        while not es.stop():
            solutions = es.ask()

            # Build all (sol_idx × csv) jobs
            all_jobs = [
                (s_idx, list(sol), csv_path, sar, args.T, args.yaw_weight, args.whole_traj)
                for s_idx, sol in enumerate(solutions)
                for (csv_path, sar) in train_csvs
            ]

            raw = _GLOBAL_POOL.starmap(_worker_evaluate_single, all_jobs)

            # Aggregate per solution
            per_sol: dict[int, list] = defaultdict(list)
            for job_id, loss, detail in raw:
                per_sol[job_id].append((loss, detail))

            fitnesses = []
            for s_idx, sol in enumerate(solutions):
                runs    = per_sol[s_idx]
                losses  = [r[0] for r in runs]
                details = [r[1] for r in runs]
                traj_loss = float(np.mean(losses))
                bwd_pen   = _backward_penalty(sol)
                agg_loss  = traj_loss + args.l2_weight * _l2_penalty(np.array(sol)) + bwd_pen
                fitnesses.append(agg_loss)

                eval_count[0] += 1
                loss_history.append(agg_loss)
                eval_history.append({
                    "coef": [float(c) for c in sol],
                    "loss": agg_loss,
                    "traj_loss": traj_loss,
                })

                if agg_loss < best_loss[0]:
                    best_loss[0] = agg_loss
                    best_coef[0] = list(sol)
                    detail_str = " | ".join(details)
                    tqdm.write(
                        f"  [{eval_count[0]:4d}] *** NEW BEST *** "
                        f"loss={agg_loss:.6f} (traj={traj_loss:.6f}+bwd={bwd_pen:.6f}+"
                        f"l2={agg_loss-traj_loss-bwd_pen:.6f})  "
                        f"({detail_str})  t={time.time()-t_start:.0f}s"
                    )

            es.tell(solutions, fitnesses)
            _update_plot()
            pbar.set_postfix({"best": f"{best_loss[0]:.4f}", "σ": f"{es.sigma:.3f}"})
            pbar.update(1)

    except KeyboardInterrupt:
        print("\n  [!] Interrupted — saving best result...")

    finally:
        pbar.close()
        try:
            _GLOBAL_POOL.terminate()
            _GLOBAL_POOL.join()
        except Exception:
            pass

    # CMA-ES final best
    if es.result.xbest is not None and es.result.fbest < best_loss[0]:
        best_coef[0] = list(es.result.xbest)
        best_loss[0] = float(es.result.fbest)

    # ------------------------------------------------------------------
    # Validation evaluation
    # ------------------------------------------------------------------
    val_loss = None
    if val_parts:
        print(f"\n  [Validation] Evaluating best coef on {len(val_parts)} val part(s)...")
        val_losses = []
        for p in val_parts:
            csv_path  = _canonical_csv(p["part_id"], args.canonical_dir)
            df        = pd.read_csv(csv_path)
            xml_path  = _tmp_xml_path()
            try:
                make_xml(best_coef[0], xml_path)
                vl = simulate_segment(xml_path, df, p["starts_at_rest"], args.T, args.yaw_weight, args.whole_traj)
                val_losses.append(vl)
                print(f"    {p['part_id']}: loss={vl:.5f}")
            except Exception as e:
                print(f"    {p['part_id']}: FAILED ({e})")
                val_losses.append(1e6)
            finally:
                try:
                    os.remove(xml_path)
                except OSError:
                    pass
        val_loss = float(np.mean(val_losses))
        print(f"  Val loss (mean): {val_loss:.6f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result = {
        "timestamp":   ts,
        "best_loss":   float(best_loss[0]),
        "val_loss":    val_loss,
        "best_coef":   [float(c) for c in best_coef[0]],
        "train_parts": [p["part_id"] for p in train_parts],
        "val_parts":   [p["part_id"] for p in val_parts],
        "config": {
            "T_window":    args.T,
            "yaw_weight":  args.yaw_weight,
            "sigma0":      args.sigma0,
            "popsize":     popsize,
            "iter":        args.iter,
            "random_start": args.random_start,
            "whole_traj":  args.whole_traj,
            "backward_penalty": float(args.backward_penalty),
            "backward_n_gaits": len(_BWD_EPISODES),
            "c7_lower":    _PHYS_BOUNDS_15[7][0],
            "generator":   _GENERATOR_MODULE,
            "centerOfMass": list(SYSTEMPARAMETERS.get("centerOfMass", [])),
        },
    }
    result_path = os.path.join(out_dir, "result.yml")
    with open(result_path, "w") as fh:
        yaml.safe_dump(to_native(result), fh)

    # Final convergence plot
    if loss_history:
        try:
            plt.rcParams.update({"text.usetex": False})
            fig, ax = plt.subplots(figsize=(7, 4))
            evals = list(range(1, len(loss_history) + 1))
            ax.plot(evals, loss_history, alpha=0.5, label="Eval")
            ax.plot(evals, np.minimum.accumulate(loss_history), "--", lw=1.5, label="Best")
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Loss")
            ax.set_title(f"FluidID CMA-ES | n_coef=15 | train={n_csvs}")
            ax.grid()
            ax.legend()
            fig.tight_layout()
            fig.savefig(conv_plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  [warn] Convergence plot skipped: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*66}")
    print(f"  Best train loss : {best_loss[0]:.6f}")
    if val_loss is not None:
        print(f"  Val loss (mean) : {val_loss:.6f}")
    print(f"  Saved to        : {result_path}")
    print(f"\n  Best coef (paste into auto_SOFI_params.yml or SYSTEMPARAMETERS):")
    coef = best_coef[0]
    print(f"    fluidCoef:")
    labels = [
        "head blunt", "head slender", "head angular", "head kutta", "head magnus",
        "tail blunt", "tail slender", "tail angular", "tail kutta", "tail magnus",
        "fin blunt",  "fin slender",  "fin angular",  "fin kutta",  "fin magnus",
    ]
    for i, (c, lbl) in enumerate(zip(coef, labels)):
        print(f"      {i:2d}: {c:10.6f}  # {lbl}")
    print(f"\n    # One-liner for fluidCoef list:")
    print(f"    [{', '.join(f'{c:.6g}' for c in coef)}]")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
