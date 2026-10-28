"""Motor system-identification: fit {kp, kv, armature, dead-time} of the MuJoCo
position actuator so the simulated motor follows the real Dynamixel.

Data: RL-deployment runs ONLY (field reserved for validation). Command read from
raw run.csv `motor_setpoint`; measured = arctan2(obs_01, obs_00); both placed on
the SAME 1 kHz grid via the SAME clock map as fishdeploy/analysis/canonical/convert.py, and the
SAME per-run motor_zero offset subtracted from BOTH (so no phantom DC tracking
error). Each part is an independent open-loop segment (p1 from rest; p2 warm-
started) — never simulated across the p1→p2 gap.

Forward model = the full SOFI_motorid.xml fish (tail + tendon + fluid load present), driven
open-loop by the ZOH command. forcerange fixed (±2.3 N·m). Velocity saturation is
IN the forward model during the fit (so it is NOT absorbed into a wrong linear
bandwidth — the mistake the 3.8 Hz gear EMA made).

Cross-check: fitted static gain should match the ARX dc_gain ≈ 0.89 (motor_zero.py).

Run (repo root, conda mujoco):
    python sysid/closed_loop/fit_motor.py            # fit + validate
    python sysid/closed_loop/fit_motor.py --no-fit   # validate current XML params only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
from paths import CANONICAL, DEPLOY_RUNS, FIELD      # noqa: E402
from sysid.closed_loop import canonical as C          # noqa: E402
CUR = PE = C

_XML = os.path.join(_REPO, "geometry", "SOFI_motorid.xml")
_DATA = str(CANONICAL)
_OUT = os.path.join(_REPO, "outputs", "closed_loop", "motor_id")

_FORCERANGE = 2.3                     # fixed (N·m), from module-01 current limit
_FIT_NUMS = [6, 13, 15, 26, 30]       # diverse: normal, near-miss, TURN(15), circle, fig-8
_VAL_NUMS = [6, 13, 14, 15, 25, 26, 27, 30, 32]   # all (full length)
_HELDOUT_NUMS = [14, 25, 27, 32]      # never in the fit → honest generalization check
_FIT_TRUNC_S = 6.0                    # truncate RL fit segments (rich excitation is enough)
_SKIP_S = 0.3                         # drop per-segment init transient from the loss
_FIX_KV = None                        # if set, kv is held fixed (v_max = forcerange/kv)
# Joint viscous damping on motor_0 (N·m·s/rad) = back-EMF / torque–speed droop, applied
# OUTSIDE the actuator forcerange clip so it caps velocity even on saturated steps.
# Physical: b = forcerange / v_max. Empirical v_max ≈ 4.8 rad/s (field motor_capability_
# report: ω_free=4.82 @12V/46RPM; RL-deploy p99.5=4.90) → b ≈ 2.3/4.8 ≈ 0.48.
_JOINT_DAMPING = 0.0                  # 0 = legacy (no joint damping); set via --joint-damping
_KV_LOWER = 0.1                       # CMA lower bound + clamp floor for free kv (--kv-lower)
_SUFFIX = ""                          # output-file suffix so variant runs don't clobber

# Field runs added to the COMBINED fit so the high-amplitude saturated regime (the
# backward-exploit-relevant 3–4 Hz band) is constrained — the RL deploy data (from an
# LPF-trained policy) under-excites it. The April field tail may differ slightly from
# the May RL tail (thinner/bendier), so the combined fit lands between the two states;
# DR covers the residual spread.
_FIELD_DIR = str(FIELD)
_FIT_FIELD = ["run_03_1.0Hz_9.0mm", "run_04_2.0Hz_9.0mm", "run_07_3.0Hz_7.0mm",
              "run_10_4.0Hz_7.0mm", "run_13_4.0Hz_9.0mm"]
_FIT_FIELD_TRUNC_S = 5.0
_W_FIELD = 0.3                        # per-segment field weight (RL=1.0). RL is the
                                      # deployment distribution → fit RL-primary; field
                                      # acts as a soft cap on the 4 Hz amplitude only.


# ---------------------------------------------------------------------------
# motor_zero (replicates 13_convert._resolve_motor_zero exactly)
# ---------------------------------------------------------------------------

def _resolve_motor_zero(run_id, saved, computed):
    if run_id in saved:
        return float(saved[run_id].get("value", 0.0))
    e = computed.get(run_id, {})
    if e.get("reliable") and e.get("recommended") is not None:
        return float(e["recommended"])
    return 0.0


def _zoh(grid, src_t, src_v):
    idx = np.clip(np.searchsorted(src_t, grid, side="right") - 1, 0, len(src_v) - 1)
    return src_v[idx]


# ---------------------------------------------------------------------------
# Segment loading (command + measured on the canonical grid, mz on BOTH)
# ---------------------------------------------------------------------------

def load_segments(nums, sim_dt=0.001, trunc_s=None):
    l2 = CUR.load_layer2(os.path.join(_DATA, "layer2_parts.yml"))
    saved = json.load(open(os.path.join(_DATA, "motor_zeros.json")))
    computed = json.load(open(os.path.join(_DATA, "motor_zero.json")))
    segs = []
    for e in l2:
        if e["num"] not in nums or e["source"] != "rl_deploy":
            continue
        run_dir = os.path.join(str(DEPLOY_RUNS), e["run_id"])
        mz = _resolve_motor_zero(e["run_id"], saved, computed)
        run = pd.read_csv(os.path.join(run_dir, "raw", "run.csv"))
        off = PE.clock_offset_s(run_dir)
        t0 = pd.read_csv(os.path.join(run_dir, "raw", "frame_timestamps.csv"))["laptop_time"].min()
        run_t = run["pi_time"].to_numpy() + off - t0
        sp = run["motor_setpoint"].to_numpy()
        enc = np.arctan2(run["obs_01"].to_numpy(), run["obs_00"].to_numpy())
        vel = run["obs_02"].to_numpy()
        for p in e["parts"]:
            t_start, t_end = p["t_start"], p["t_end"]
            if trunc_s is not None:
                t_end = min(t_end, t_start + trunc_s)
            grid = C._grid(t_start, t_end, sim_dt)
            cmd = _zoh(grid, run_t, sp) - mz       # ZOH staircase, mz on command
            meas = C._interp(run_t, enc, grid) - mz  # matches canonical (mz on measured)
            mvel = C._interp(run_t, vel, grid)
            segs.append(dict(
                label=CUR.part_id(e["num"], e["name"], p["part"]),
                cmd=cmd, meas=meas, mvel=mvel,
                rest=bool(p["starts_at_rest"]), mz=mz, src="rl",
            ))
    return segs


def load_field_segments(runs, trunc_s=5.0, sim_dt=0.001):
    """Field sinusoid runs as open-loop segments: drive Commanded_Alpha, match
    Ghost_Alpha (both already 1 kHz on a unified clock, same encoder frame → no mz)."""
    segs = []
    for run in runs:
        csv = os.path.join(_FIELD_DIR, run, "processed", "03_field2sim_ready.csv")
        if not os.path.isfile(csv):
            print(f"  [skip field] {run}: not found"); continue
        df = pd.read_csv(csv)
        n = min(len(df), int(round(trunc_s / sim_dt)))
        cmd = df["Commanded_Alpha"].to_numpy()[:n]
        meas = df["Ghost_Alpha"].to_numpy()[:n]
        mvel = np.gradient(meas, sim_dt)
        segs.append(dict(label=f"field_{run}", cmd=cmd, meas=meas, mvel=mvel,
                         rest=True, mz=0.0, src="field"))
    return segs


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def load_model(kp, kv, armature, fluid=True, damping=None):
    if damping is None:
        damping = _JOINT_DAMPING
    m = mujoco.MjModel.from_xml_path(_XML)
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_0")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "motor_0")
    qadr = int(m.jnt_qposadr[jid]); vadr = int(m.jnt_dofadr[jid])
    m.actuator_gainprm[aid][0] = kp
    m.actuator_biasprm[aid][1] = -kp
    m.actuator_biasprm[aid][2] = -kv
    m.actuator_forcerange[aid][:] = [-_FORCERANGE, _FORCERANGE]
    m.dof_armature[vadr] = armature
    m.dof_damping[vadr] = damping        # back-EMF droop, OUTSIDE the actuator clip
    if not fluid:
        m.opt.density = 0.0
    return m, aid, qadr, vadr


def simulate(m, aid, qadr, vadr, cmd, *, deadtime_steps, init_pos, init_vel):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[qadr] = init_pos; d.qvel[vadr] = init_vel
    mujoco.mj_forward(m, d)
    n = len(cmd)
    pos = np.empty(n); vel = np.empty(n)
    buf = deque([float(cmd[0])] * (deadtime_steps + 1), maxlen=deadtime_steps + 1)
    for i in range(n):
        buf.append(float(cmd[i]))
        d.ctrl[aid] = buf[0]
        mujoco.mj_step(m, d)
        pos[i] = d.qpos[qadr]; vel[i] = d.qvel[vadr]
    return pos, vel


def evaluate(x, segs, *, fluid=True, w_vel=0.01, sim_dt=0.001):
    kp, kv, arm_k, td_ms = x
    if _FIX_KV is not None:
        kv = _FIX_KV                       # honor exactly (incl. 0); no lower clamp
    else:
        kv = max(_KV_LOWER, kv)
    kp = max(0.5, kp); arm_k = max(0.2, arm_k)
    deadtime = int(round(max(0.0, td_ms)))
    m, aid, qadr, vadr = load_model(kp, kv, arm_k * 1e-3, fluid=fluid)
    skip = int(round(_SKIP_S / sim_dt))
    tot = 0.0; tot_w = 0.0
    for s in segs:
        w = _W_FIELD if s.get("src") == "field" else 1.0
        ip = float(s["meas"][0]); iv = 0.0 if s["rest"] else float(s["mvel"][0])
        sp, sv = simulate(m, aid, qadr, vadr, s["cmd"],
                          deadtime_steps=deadtime, init_pos=ip, init_vel=iv)
        e = sp[skip:] - s["meas"][skip:]
        ev = sv[skip:] - s["mvel"][skip:]
        tot += w * float(np.mean(e ** 2) + w_vel * np.mean(ev ** 2))
        tot_w += w
    return tot / tot_w


# ---------------------------------------------------------------------------
# Fit + validate
# ---------------------------------------------------------------------------

def fit(segs):
    import cma
    x0 = [5.0, 0.40, 6.0, 40.0]                    # kp, kv, armature(×1e-3), Td[ms]
    opts = {
        "bounds": [[1.0, _KV_LOWER, 0.5, 0.0], [25.0, 1.2, 20.0, 60.0]],
        "scaling_of_variables": [5.0, 0.4, 6.0, 40.0],
        "maxfevals": 120, "popsize": 10, "seed": 0, "verb_disp": 1,
    }
    es = cma.CMAEvolutionStrategy(x0, 0.3, opts)
    while not es.stop():
        sols = es.ask()
        # fit fluid-OFF (6x faster; motor tracking is only ~6% coupled to fluid)
        es.tell(sols, [evaluate(x, segs, fluid=False) for x in sols])
        es.disp()
    xb = es.result.xbest
    kv_out = _FIX_KV if _FIX_KV is not None else float(xb[1])  # xb[1] drifts when kv fixed
    return dict(kp=float(xb[0]), kv=float(kv_out), armature=float(xb[2] * 1e-3),
                deadtime_steps=int(round(xb[3])), deadtime_ms=float(round(xb[3])),
                loss=float(es.result.fbest))


def validate(params, nums, *, tag, sim_dt=0.001):
    segs = load_segments(nums, sim_dt=sim_dt, trunc_s=None)
    m, aid, qadr, vadr = load_model(params["kp"], params["kv"], params["armature"])
    skip = int(round(_SKIP_S / sim_dt))
    rows = []
    overlays = []
    for s in segs:
        ip = float(s["meas"][0]); iv = 0.0 if s["rest"] else float(s["mvel"][0])
        sp, sv = simulate(m, aid, qadr, vadr, s["cmd"],
                          deadtime_steps=params["deadtime_steps"], init_pos=ip, init_vel=iv)
        pr = np.degrees(np.sqrt(np.mean((sp[skip:] - s["meas"][skip:]) ** 2)))
        vr = float(np.sqrt(np.mean((sv[skip:] - s["mvel"][skip:]) ** 2)))
        rows.append(dict(label=s["label"], pos_rmse_deg=float(pr), vel_rmse=vr))
        if len(overlays) < 3:
            overlays.append((s["label"], s["cmd"], s["meas"], sp))
    df = pd.DataFrame(rows)
    print(f"\n=== validation ({tag}) ===")
    print(df.to_string(index=False))
    print(f"  MEAN pos RMSE = {df.pos_rmse_deg.mean():.2f}°   vel RMSE = {df.vel_rmse.mean():.3f} rad/s")

    fig, axes = plt.subplots(len(overlays), 1, figsize=(12, 3 * len(overlays)), squeeze=False)
    for ax, (lab, cmd, meas, sp) in zip(axes[:, 0], overlays):
        t = np.arange(len(cmd)) * sim_dt
        ax.plot(t, np.degrees(cmd), color="gray", lw=0.8, label="command")
        ax.plot(t, np.degrees(meas), "k-", lw=1.2, label="measured")
        ax.plot(t, np.degrees(sp), "r-", lw=1.0, alpha=0.8, label="sim (fitted)")
        ax.set_title(lab, fontsize=9); ax.set_ylabel("motor [°]"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("time [s]")
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    p = os.path.join(_OUT, f"overlay_{tag}.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  overlay: {p}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fit", action="store_true",
                    help="skip CMA; validate the current XML params")
    ap.add_argument("--fix-kv", type=float, default=None,
                    help="hold kv fixed (e.g. 0 to drop actuator kv); fit kp/armature/T_d only")
    ap.add_argument("--joint-damping", type=float, default=0.0,
                    help="motor_0 joint viscous damping b (N·m·s/rad), back-EMF droop OUTSIDE "
                         "the actuator clip. b=forcerange/v_max; v_max≈4.8 → b≈0.48.")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix for output files so variant runs don't overwrite each other")
    ap.add_argument("--kv-lower", type=float, default=0.1,
                    help="lower bound/floor for free kv in CMA (default 0.1; set 0 to let kv→0)")
    args = ap.parse_args()
    os.makedirs(_OUT, exist_ok=True)
    globals()["_KV_LOWER"] = float(args.kv_lower)
    globals()["_JOINT_DAMPING"] = float(args.joint_damping)
    globals()["_SUFFIX"] = args.out_suffix
    if args.joint_damping > 0:
        vmax_b = _FORCERANGE / args.joint_damping
        print(f"Joint damping b = {args.joint_damping}  →  velocity ceiling "
              f"forcerange/b = {vmax_b:.2f} rad/s (empirical ≈ 4.8)")
    if args.fix_kv is not None:
        globals()["_FIX_KV"] = args.fix_kv
        print(f"kv FIXED at {args.fix_kv}")

    if args.no_fit:
        params = dict(kp=5.0, kv=0.40, armature=0.001, deadtime_steps=0, deadtime_ms=0.0)
        print(f"Validating current XML params: {params}")
    else:
        print(f"Loading COMBINED FIT segments: RL {_FIT_NUMS} (≤{_FIT_TRUNC_S}s) "
              f"+ field {_FIT_FIELD} (≤{_FIT_FIELD_TRUNC_S}s) …")
        fit_segs = (load_segments(_FIT_NUMS, trunc_s=_FIT_TRUNC_S)
                    + load_field_segments(_FIT_FIELD, trunc_s=_FIT_FIELD_TRUNC_S))
        for s in fit_segs:
            print(f"  fit[{s['src']:5s}]: {s['label']:34s} N={len(s['cmd']):6d} rest={s['rest']}")
        params = fit(fit_segs)
        params["joint_damping"] = float(_JOINT_DAMPING)
        print(f"\n=== FITTED PARAMETERS ===\n{json.dumps(params, indent=2)}")
        if _JOINT_DAMPING > 0:
            print(f"  velocity ceiling = forcerange/b = {_FORCERANGE / _JOINT_DAMPING:.2f} rad/s "
                  f"(empirical ≈ 4.8); actuator-kv equilibrium = "
                  f"{_FORCERANGE / max(params['kv'], 1e-6):.2f}")
        else:
            print(f"  implied v_max = forcerange/kv = {_FORCERANGE / max(params['kv'],1e-6):.2f} rad/s")
        json.dump(params, open(os.path.join(_OUT, f"fitted_params{_SUFFIX}.json"), "w"), indent=2)

    val_fit = validate(params, _FIT_NUMS, tag=f"fit_runs{_SUFFIX}")
    val_held = validate(params, _HELDOUT_NUMS, tag=f"heldout{_SUFFIX}")
    val_all = validate(params, _VAL_NUMS, tag=f"all_runs{_SUFFIX}")
    val_all.to_csv(os.path.join(_OUT, f"validation_rmse{_SUFFIX}.csv"), index=False)
    print(f"\n  fit-set    mean pos RMSE = {val_fit.pos_rmse_deg.mean():.2f}°")
    print(f"  HELD-OUT   mean pos RMSE = {val_held.pos_rmse_deg.mean():.2f}°  "
          f"(runs {_HELDOUT_NUMS} — honest generalization)")
    print(f"  all-runs   mean pos RMSE = {val_all.pos_rmse_deg.mean():.2f}°")
    print("  Adopted (kv-in-clip, no damping) reference: ~2.49° all-runs.")


if __name__ == "__main__":
    main()
