"""Open-loop motor tracking comparison: hardware vs sim (Kp=800 vs Kp=1.3).

Run from the fishsim root:
    python sysid/motor/motor_tracking_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

FIELD_CSV = (
    Path(__file__).parent.parent
    / "field_test/data/20260401_first_swim/run_3_1.0Hz_9.0mm/processed/03_field2sim_ready.csv"
)
XML_PATH  = str(Path(__file__).resolve().parent.parent.parent / "geometry/SOFI_cpg.xml")
OUT_DIR   = Path(__file__).parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

FREQ      = 1.0   # Hz
PERIOD    = 1.0 / FREQ
N_GRID    = 300   # points per averaged period
SKIP_PERIODS = 2  # discard transient
USE_PERIODS  = 8  # periods to average

phase_grid = np.linspace(0, 1, N_GRID, endpoint=False)
t_grid     = phase_grid * PERIOD   # relative time within one period


# ─────────────────────────────────────────────────────────────────────────────
# 1. Field data
# ─────────────────────────────────────────────────────────────────────────────

def load_field():
    df  = pd.read_csv(FIELD_CSV)
    t   = df["Time"].values
    cmd = df["Commanded_Alpha"].values

    meas_mask = df["Meas_Alpha"].notna()
    t_m   = df.loc[meas_mask, "Time"].values
    meas  = df.loc[meas_mask, "Meas_Alpha"].values

    amp = float(np.abs(cmd).max())
    t_start = SKIP_PERIODS * PERIOD

    cmd_acc  = np.zeros(N_GRID)
    meas_acc = np.zeros(N_GRID)
    cnt      = np.zeros(N_GRID)

    for i in range(USE_PERIODS):
        t0, t1 = t_start + i * PERIOD, t_start + (i + 1) * PERIOD

        # commanded (dense 1 kHz)
        sel_c = (t >= t0) & (t < t1)
        cmd_acc += np.interp(t_grid, t[sel_c] - t0, cmd[sel_c])

        # measured (sparse ~52 Hz, filter zeros = placeholders)
        sel_m = (t_m >= t0) & (t_m < t1) & (meas != 0.0)
        if sel_m.sum() < 4:
            continue
        vi = np.interp(t_grid, t_m[sel_m] - t0, meas[sel_m])
        meas_acc += vi
        cnt      += 1

    cmd_avg  = cmd_acc / USE_PERIODS
    meas_avg = np.where(cnt > 0, meas_acc / cnt, np.nan)
    return cmd_avg, meas_avg, amp


# ─────────────────────────────────────────────────────────────────────────────
# 2. MuJoCo open-loop simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(kp, kv, amp, dt=0.001):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    motor_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_0")
    model.actuator_gainprm[motor_idx, 0]  =  kp
    model.actuator_biasprm[motor_idx, 1]  = -kp
    model.actuator_biasprm[motor_idx, 2]  = -kv

    mujoco.mj_resetData(model, data)

    n_total = int((SKIP_PERIODS + USE_PERIODS) * PERIOD / dt)
    t_skip  = SKIP_PERIODS * PERIOD

    t_list, cmd_list, qpos_list = [], [], []
    for step in range(n_total):
        t = step * dt
        cmd_rad = amp * np.sin(2 * np.pi * FREQ * t)
        data.ctrl[motor_idx] = cmd_rad
        mujoco.mj_step(model, data)

        if t >= t_skip:
            t_list.append(t - t_skip)
            cmd_list.append(cmd_rad)
            qpos_list.append(float(data.joint("motor_0").qpos[0]))

    t_arr, cmd_arr, qpos_arr = map(np.array, (t_list, cmd_list, qpos_list))

    cmd_acc  = np.zeros(N_GRID)
    meas_acc = np.zeros(N_GRID)
    for i in range(USE_PERIODS):
        t0 = i * PERIOD
        sel = (t_arr >= t0) & (t_arr < t0 + PERIOD)
        cmd_acc  += np.interp(t_grid, t_arr[sel] - t0, cmd_arr[sel])
        meas_acc += np.interp(t_grid, t_arr[sel] - t0, qpos_arr[sel])

    return cmd_acc / USE_PERIODS, meas_acc / USE_PERIODS


# ─────────────────────────────────────────────────────────────────────────────
# 3. Run everything
# ─────────────────────────────────────────────────────────────────────────────

print("Loading field data …")
cmd_hw, meas_hw, amp = load_field()
print(f"  amplitude from field: {amp:.4f} rad")

print("Running sim  Kp=800, Kv=20 …")
cmd_800, meas_800 = run_sim(kp=800.0, kv=20.0, amp=amp)

print("Running sim  Kp=1.3, Kv=0  …")
cmd_13,  meas_13  = run_sim(kp=1.3,   kv=0.0,  amp=amp)

print("Plotting …")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figures
# ─────────────────────────────────────────────────────────────────────────────

BLUE = "#1f77b4"
RED  = "#d62728"
LW   = 2.0
YLIM = (-0.85, 0.85)

# --- Panel figure: three stacked subplots ---
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, sharey=True)

titles = [
    "Hardware  (1 Hz, 9 mm field run  —  averaged over 8 periods)",
    "Sim  Kp = 800 Nm/rad,  Kv = 20 Nm·s/rad  (current sim params)",
    "Sim  Kp = 1.3 Nm/rad,  Kv = 0  (empirical / field-matched)",
]
data_pairs = [
    (cmd_hw,  meas_hw),
    (cmd_800, meas_800),
    (cmd_13,  meas_13),
]

for ax, title, (cmd, meas) in zip(axes, titles, data_pairs):
    ax.plot(t_grid * 1000, cmd,  color=BLUE, lw=LW, label="Commanded")
    ax.plot(t_grid * 1000, meas, color=RED,  lw=LW, label="Measured",
            linestyle="--" if "800" in title else "-")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Motor position (rad)")
    ax.set_ylim(*YLIM)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # annotation: peak error
    peak_cmd  = float(np.nanmax(np.abs(cmd)))
    peak_meas = float(np.nanmax(np.abs(meas)))
    shortfall = (peak_cmd - peak_meas) / peak_cmd * 100 if peak_cmd > 0 else 0
    ax.annotate(
        f"peak shortfall  {shortfall:.1f}%",
        xy=(0.97, 0.06), xycoords="axes fraction",
        ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

axes[-1].set_xlabel("Time within period (ms)")
fig.suptitle("Open-loop motor tracking: hardware vs sim", fontsize=13, y=1.01)
fig.tight_layout()
out_panel = OUT_DIR / "motor_tracking_panel.png"
fig.savefig(out_panel, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {out_panel}")


# --- Overlay figure: all three measured on one plot ---
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(t_grid * 1000, cmd_hw,  color=BLUE,  lw=LW,   label="Commanded (all same)")
ax2.plot(t_grid * 1000, meas_hw, color="black", lw=LW,  label="Measured — Hardware",  ls="-")
ax2.plot(t_grid * 1000, meas_800,color=RED,    lw=LW,   label="Measured — Sim Kp=800", ls="--")
ax2.plot(t_grid * 1000, meas_13, color="green", lw=LW,  label="Measured — Sim Kp=1.3", ls=":")

ax2.set_xlabel("Time within period (ms)")
ax2.set_ylabel("Motor position (rad)")
ax2.set_title("Overlay: commanded vs measured — hardware, Sim Kp=800, Sim Kp=1.3")
ax2.set_ylim(*YLIM)
ax2.axhline(0, color="gray", lw=0.5, ls=":")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)
fig2.tight_layout()
out_overlay = OUT_DIR / "motor_tracking_overlay.png"
fig2.savefig(out_overlay, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  → {out_overlay}")

# --- Summary numbers ---
print("\n=== Summary ===")
peak_cmd = float(np.nanmax(np.abs(cmd_hw)))
for label, meas in [("Hardware", meas_hw), ("Sim Kp=800", meas_800), ("Sim Kp=1.3", meas_13)]:
    peak = float(np.nanmax(np.abs(meas)))
    shortfall = (peak_cmd - peak) / peak_cmd * 100
    # phase lag estimate: time of first peak
    idx_cmd  = int(np.argmax(cmd_hw))
    idx_meas = int(np.argmax(np.nan_to_num(meas, nan=0.0)))
    lag_ms   = (idx_meas - idx_cmd) / N_GRID * PERIOD * 1000
    print(f"  {label:15s}  peak={peak:.4f} rad  shortfall={shortfall:.1f}%  phase_lag≈{lag_ms:.0f} ms")
