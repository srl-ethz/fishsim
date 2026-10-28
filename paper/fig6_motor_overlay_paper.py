#!/usr/bin/env python3
"""Fig 6 (motor tracking overlay) at the PAPER's column width.

Derived from the report generator; only the output folder and the set_size reference
width are changed, so the report asset (387 pt, printing at 4.4 pt) is untouched.
"""
import glob, sys, os, numpy as np, pandas as pd
from pathlib import Path
from collections import deque
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
from _paths import GEOMETRY as _GEOMETRY, DEPLOY_RUNS as _DEPLOY_RUNS, OUT as _OUT
from paper_style import use_paper_style, size, save, ROLE, LW, EMPH_LW
use_paper_style()

FC = _GEOMETRY
RUNS = _DEPLOY_RUNS
DEST = _OUT
DEST.mkdir(parents=True, exist_ok=True)
RUN = sys.argv[1] if len(sys.argv) > 1 else "full_success_2"
WIN  = (2.0, 12.0)   # window over which RMSE is computed (whole-run metric)
ZOOM = (3.0, 5.0)    # x-axis view: a representative 2 s excerpt, re-zeroed so the axis reads 0-2 s

MODELS = {
    "old": dict(xml=str(FC/"SOFI_cpg.xml"), dt_sub=0,  c="#D55E00", ls="-",
                lab="old motor model + E13a fluid", short="Idealized model"),
    "new": dict(xml=str(FC/"SOFI_da.xml"),  dt_sub=14, c="#0072B2", ls="-",
                lab="new motor model + refitB fluid", short="Identified model"),
}


def replay(cfg, cmd, dt):
    m = mujoco.MjModel.from_xml_path(cfg["xml"])
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_0")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "motor_0")
    qadr = int(m.jnt_qposadr[jid])
    fs = max(1, int(round(dt / m.opt.timestep)))
    buf = deque([cmd[0]] * (cfg["dt_sub"] + 1), maxlen=cfg["dt_sub"] + 1)
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
    out = np.empty(len(cmd))
    for i, c in enumerate(cmd):
        for _ in range(fs):
            buf.append(float(c)); d.ctrl[aid] = buf[0]; mujoco.mj_step(m, d)
        out[i] = d.qpos[qadr]
    return out


folder = sorted(glob.glob(str(RUNS / f"*{RUN}")))[0]
df = pd.read_csv(f"{folder}/raw/run.csv")
t = df["pi_time"].to_numpy(); t -= t[0]
cmd = df["motor_setpoint"].to_numpy()
meas = np.arctan2(df["obs_01"].to_numpy(), df["obs_00"].to_numpy())
dt = float(np.median(np.diff(t)))
sims = {k: replay(cfg, cmd, dt) for k, cfg in MODELS.items()}

deg = np.degrees
mask = (t >= WIN[0]) & (t <= WIN[1])
tp = t - ZOOM[0]                       # plotted time: the excerpt starts at 0
fig, ax = plt.subplots(figsize=size(1.0, aspect=0.48), layout="constrained")
# commanded setpoint: dashed grey behind, so it is separable from the old model
# (which over-tracks it); the message is old-model->setpoint, new-model->encoder.
ax.plot(tp[mask], deg(cmd[mask]), color=ROLE["setpoint"], lw=LW, ls=(0, (5, 2)),
        alpha=0.9, label="Commanded", zorder=1)
ax.plot(tp[mask], deg(meas[mask]), color=ROLE["measured"], lw=EMPH_LW,
        label="Measured", zorder=4)
rmse = {}
for k, cfg in MODELS.items():
    r = np.sqrt(np.mean(deg(sims[k][mask] - meas[mask]) ** 2)); rmse[k] = r
    ax.plot(tp[mask], deg(sims[k][mask]), color=cfg["c"], lw=LW, alpha=0.9,
            label=cfg["short"],
            zorder=(2 if k == "old" else 3))
ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
ax.set_xlim(0, ZOOM[1] - ZOOM[0])
ax.set_xticks(range(0, int(ZOOM[1] - ZOOM[0]) + 1))
# "outside" legend: constrained layout reserves the strip for it, so the figure keeps
# its exact column width and the 7 pt type is never rescaled by LaTeX
fig.legend(ncol=2, loc="outside lower center", handlelength=1.6, columnspacing=1.2)
print(f"run={RUN}  RMSE old={rmse['old']:.2f} new={rmse['new']:.2f}")
save(fig, "fig6_motor_overlay", expect="col")
