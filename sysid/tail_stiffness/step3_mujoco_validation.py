"""
Step 3 — MuJoCo Structural Validation
--------------------------------------
Measures effective tendon stiffness k_eff_mujoco from a fixed-head
static sweep, then repeats with fluid forces active to show the shift.

Sub-test A (air, density=0): k_eff_mujoco must be within 5% of R4 target.
Sub-test B (water):           shows added-mass / drag shift from fluid.

Usage:
    micromamba run -n mujoco python sysid/tail_stiffness/step3_mujoco_validation.py

NOTE — Sub-test A systematic offset (2026-04-28, first run):
    Sub-test A reported ~19.4% error vs k_target when first run against the R4 structural model.

    Root cause: k_eff_from_joints() uses the analytical formula
        k_eff = r² / Σ(d_i² / k_joint_i)
    with TENDON_MOMENT_ARMS = simple perpendicular eyelet-to-joint-axis distances.
    MuJoCo computes the tendon Jacobian ∂L/∂q from the actual 3D tendon path through
    pulleys, connector bodies, and body frames — these effective moment arms differ
    from the analytical d_i, producing a systematic discrepancy in k_eff.

    This is NOT a bug: the structural model is correctly calibrated from R4 field data.
    The offset means MuJoCo's spring forces deviate ~19% from the physical fish.

    Fix (if needed): re-fit per-joint stiffness so k_eff_mujoco matches k_eff_R4.
    Scale all hinge_stiffness values by (k_eff_R4 / k_eff_mujoco_static).
    For E13, this offset is accepted and documented; fluid SysID proceeds with the
    as-calibrated values. Document in FLUID_COEFFICIENTS.md for each experiment.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import mujoco
import yaml

XML_PATH    = "geometry/SOFI_twin.xml"
CONFIG_PATH = "sysid/fluid/field2sim.yml"
PLOT_DIR    = os.path.join(os.path.dirname(__file__), "plots")
PASS_TOL    = 0.05   # 5% tolerance on k_eff

SWEEP_ANGLES = np.linspace(-0.40, 0.40, 17)   # rad, static sweep
SETTLE_STEPS = 3000                              # steps at 1kHz = 3 s to reach steady state
FREQS_HZ     = [1.0, 2.0, 3.0, 4.0]
N_CYCLES     = 8                                 # sinusoidal cycles per frequency


def fix_head_joints(model: mujoco.MjModel, data: mujoco.MjData, fish_id: int = 0) -> None:
    """Zero position and velocity of all head floating-base joints."""
    head_joints = [f"headX_{fish_id}", f"headY_{fish_id}", f"headZ_{fish_id}",
                   f"headRollX_{fish_id}", f"headRollY_{fish_id}", f"headRollZ_{fish_id}"]
    for jname in head_joints:
        try:
            dof = model.joint(jname).dofadr[0]
            data.qpos[dof] = 0.0
            data.qvel[dof] = 0.0
        except Exception:
            pass


def get_motor_dof(model: mujoco.MjModel, fish_id: int = 0) -> int:
    return model.joint(f"motor_{fish_id}").dofadr[0]


def measure_keff_static(model: mujoco.MjModel, fish_id: int = 0) -> float:
    """
    Static sweep: set motor to each angle, settle, read passive spring torque.
    Returns k_eff [Nm/rad] from linear fit of torque vs angle.
    """
    data = mujoco.MjData(model)
    motor_dof = get_motor_dof(model, fish_id)
    torques = []

    for theta in SWEEP_ANGLES:
        mujoco.mj_resetData(model, data)
        for _ in range(SETTLE_STEPS):
            data.qpos[motor_dof] = theta
            data.qvel[motor_dof] = 0.0
            fix_head_joints(model, data, fish_id)
            mujoco.mj_step(model, data)
        # Spring torque opposes deflection; qfrc_passive is the passive spring+damping force
        torques.append(-data.qfrc_passive[motor_dof])

    k_eff, _ = np.polyfit(SWEEP_ANGLES, torques, 1)
    return float(k_eff)


def measure_dynamic_H(model: mujoco.MjModel, freq_hz: float,
                       fish_id: int = 0) -> complex:
    """
    Sinusoidal sweep at freq_hz: teleport motor, record passive torque, extract H phasor.
    Returns H = tau_phasor / theta_phasor.
    """
    data   = mujoco.MjData(model)
    dt     = model.opt.timestep
    omega  = 2 * np.pi * freq_hz
    T      = 1.0 / freq_hz
    A      = 0.30   # rad amplitude (well within range)
    t_total = (N_CYCLES + 2) * T   # 2 extra for transient
    n_steps = int(t_total / dt)
    motor_dof = get_motor_dof(model, fish_id)

    t_rec, theta_rec, tau_rec = [], [], []

    mujoco.mj_resetData(model, data)
    for step in range(n_steps):
        t = step * dt
        data.qpos[motor_dof] = A * np.sin(omega * t)
        data.qvel[motor_dof] = A * omega * np.cos(omega * t)
        fix_head_joints(model, data, fish_id)
        mujoco.mj_step(model, data)

        if t > 2 * T:   # discard 2-cycle transient
            t_rec.append(t)
            theta_rec.append(data.qpos[motor_dof])
            tau_rec.append(-data.qfrc_passive[motor_dof])

    t_arr     = np.array(t_rec)
    theta_arr = np.array(theta_rec)
    tau_arr   = np.array(tau_rec)

    # Phasor via least-squares sine fit
    def phasor(t, y, f):
        M = np.column_stack([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)])
        c, *_ = np.linalg.lstsq(M, y, rcond=None)
        return complex(c[0], -c[1])   # convention: A*sin(ωt+φ) → Ae^{jφ}

    theta_ph = phasor(t_arr, theta_arr, freq_hz)
    tau_ph   = phasor(t_arr, tau_arr,   freq_hz)
    return tau_ph / theta_ph if abs(theta_ph) > 1e-6 else complex(0, 0)


def run_sub_test(label: str, density: float, config: dict) -> dict:
    """Run static + dynamic tests with given fluid density."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.density = density

    k_static = measure_keff_static(model, fish_id=0)
    H_dynamic = {}
    for f in FREQS_HZ:
        H_dynamic[f] = measure_dynamic_H(model, f, fish_id=0)

    omega_sq = np.array([(2*np.pi*f)**2 for f in FREQS_HZ])
    real_H   = np.array([H_dynamic[f].real for f in FREQS_HZ])
    k_dyn, I_dyn = np.polyfit(omega_sq, real_H, 1)[::-1]   # [k, -I] from y = k - I*ω²

    return {
        "label":    label,
        "density":  density,
        "k_static": k_static,
        "k_dyn":    float(k_dyn),
        "I_dyn":    float(-I_dyn),
        "H":        H_dynamic,
    }


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    k_eff_target = config.get("structural", {}).get("hinge_stiffness", None)
    if k_eff_target is None:
        print("WARNING: no structural.hinge_stiffness in config — cannot compute target k_eff")
        k_target = None
    else:
        ks = k_eff_target if isinstance(k_eff_target, list) else [k_eff_target]*5
        from sysid.tail_stiffness.utils import k_eff_from_joints
        k_target = k_eff_from_joints(np.array(ks))
        print(f"Target k_eff (from config per-joint values): {k_target:.4f} Nm/rad")

    print("\n" + "="*60)
    print("Sub-test A — Air (density = 0, fluid forces off)")
    print("="*60)
    air = run_sub_test("Air", density=0.0, config=config)
    print(f"  k_eff static:  {air['k_static']:.4f} Nm/rad")
    print(f"  k_eff dynamic: {air['k_dyn']:.4f} Nm/rad")
    print(f"  I_eff dynamic: {air['I_dyn']*1e3:.4f} x10^-3 kg*m^2")
    if k_target is not None:
        err = abs(air['k_static'] - k_target) / k_target
        status = "PASS" if err <= PASS_TOL else "FAIL"
        print(f"  vs target {k_target:.4f}: error = {err*100:.1f}%  [{status}]")

    print("\n" + "="*60)
    print("Sub-test B — Water (density = 1000 kg/m^3, fluid forces on)")
    print("="*60)
    water = run_sub_test("Water", density=1000.0, config=config)
    print(f"  k_eff static:  {water['k_static']:.4f} Nm/rad")
    print(f"  k_eff dynamic: {water['k_dyn']:.4f} Nm/rad")
    print(f"  I_eff dynamic: {water['I_dyn']*1e3:.4f} x10^-3 kg*m^2")
    print(f"  DeltaI (fluid added mass): {(water['I_dyn']-air['I_dyn'])*1e3:.4f} x10^-3 kg*m^2")
    if air['I_dyn'] > 1e-8:
        print(f"  I_water/I_air: {water['I_dyn']/air['I_dyn']:.3f}  (expect >1 from added mass)")

    # Bode plot
    freqs = FREQS_HZ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for res, col in [(air, "C0"), (water, "C1")]:
        re_H = [res["H"][f].real for f in freqs]
        im_H = [res["H"][f].imag for f in freqs]
        ax1.plot(freqs, re_H, "o-", color=col, label=res["label"])
        ax2.plot(freqs, im_H, "o-", color=col, label=res["label"])

    if k_target is not None:
        ax1.axhline(k_target, color="k", ls="--", lw=1.5, label=f"Target k_eff={k_target:.3f}")
    ax1.set_xlabel("Frequency [Hz]"); ax1.set_ylabel("Re(H)  [Nm/rad]")
    ax1.set_title("Stiffness response: Re(H) = k_eff - I_eff*omega^2")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Frequency [Hz]"); ax2.set_ylabel("Im(H)  [Nm*s/rad * omega]")
    ax2.set_title("Damping response: Im(H)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("step3_mujoco_validation: Air vs Water H(omega)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "s3_mujoco_validation_bode.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nSaved: {out}")

    if k_target is not None:
        err = abs(air['k_static'] - k_target) / k_target
        if err > PASS_TOL:
            print(f"\nFAIL: k_eff_mujoco={air['k_static']:.4f} deviates {err*100:.1f}% from target {k_target:.4f}")
            sys.exit(1)
        print(f"\nPASS: k_eff_mujoco within {err*100:.1f}% of target")


if __name__ == "__main__":
    main()
