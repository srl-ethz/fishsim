#!/usr/bin/env python3
"""
SysID Post-Hoc Evaluator
========================
Loads a saved opt_result_*.yml and produces:

  1. trajectories_train_real.png  — training runs: real fish vs twin (real Ghost_Alpha)
  2. trajectories_train_sin.png   — training runs: real fish vs twin (fitted sinusoid)
  3. trajectories_test_real.png   — test runs (unseen): real fish vs twin (real Ghost_Alpha)
  4. trajectories_test_sin.png    — test runs: real fish vs twin (fitted sinusoid)
  5. per_run_loss_bar.png         — grouped bar chart: real vs sinusoid vs test loss
  6. convergence_train_val.png    — training loss vs sinusoid-validation loss over evaluations
  7. trajectories_interpolated.png — interpolated free-swim (no real fish; twin only)

Usage:
    python sysid/fluid/eval_sysid.py                        # most recent result
    python sysid/fluid/eval_sysid.py outputs/sysid/opt_result_20260423_074659.yml
    python sysid/fluid/eval_sysid.py --skip_val_curve       # skip expensive recon
    python sysid/fluid/eval_sysid.py --legacy_mass          # use old mass model (E1-E8)
"""

import argparse, copy, glob, os, subprocess, sys, yaml
import numpy as np, pandas as pd, mujoco
import matplotlib.pyplot as plt
import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
try:
    from report_style import use_report_style, CB, savefig as _rsave, TEXTWIDTH_PT
    use_report_style(collage=True)
    _RS=True
except Exception:
    _RS=False
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from geometry.auto_SOFI_field2sim import generate_xml as _generate_xml_current
from geometry.auto_SOFI_field2sim import SYSTEMPARAMETERS as _SP_CURRENT

# Active generator — overridden by --legacy_mass at runtime
_generate_xml = _generate_xml_current
_SP_TEMPLATE  = _SP_CURRENT

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TMP_CFG   = os.path.join(_REPO_ROOT, "sysid", "fluid", "_eval_tmp.yml")
_TMP_XML   = os.path.join(_REPO_ROOT, "geometry", "_eval_tmp.xml")
_DATA_GLOB = os.path.join(os.environ.get("FISHDEPLOY_ROOT", os.path.join(_REPO_ROOT, "..", "fishdeploy")), "field_test", "data",
                           "20260401_first_swim", "run_*", "processed", "03_field2sim_ready.csv")
_START_WEIGHT = 4.0  # must match the value used during optimisation
_PULLEY_R       = 0.0175  # m — for mm→rad conversion in interpolated specs
_PULLEY_DISPLAY = 17.5    # mm — real pulley radius, used to convert rad → mm for display

# Interpolated free-swim conditions — NOT in training or test datasets.
# Amplitudes / offsets converted: rad = mm / _PULLEY_R
# Straight runs: frequencies and amplitudes between known training values
_INTERP_STRAIGHT = [
    {'freq': 1.75, 'amp': 9.0/17.5, 'offset': 0.0, 'label': '1.75Hz / 9mm (interp)'},
    {'freq': 2.5,  'amp': 7.0/17.5, 'offset': 0.0, 'label': '2.5Hz / 7mm (interp)'},
    {'freq': 3.5,  'amp': 6.0/17.5, 'offset': 0.0, 'label': '3.5Hz / 6mm (interp)'},
    {'freq': 2.0,  'amp': 5.0/17.5, 'offset': 0.0, 'label': '2Hz / 5mm (interp)'},
]
# Turning runs: offset ≠ 0, also at in-between frequencies/amplitudes
_INTERP_TURNING = [
    {'freq': 1.5,  'amp': 8.0/17.5, 'offset': -1.75/17.5, 'label': '1.5Hz / 8mm / −1.75mm L (interp)'},
    {'freq': 2.5,  'amp': 7.0/17.5, 'offset': -2.5/17.5,  'label': '2.5Hz / 7mm / −2.5mm L (interp)'},
    {'freq': 3.0,  'amp': 8.0/17.5, 'offset': -2.0/17.5,  'label': '3Hz / 8mm / −2mm L (interp)'},
    {'freq': 2.0,  'amp': 6.5/17.5, 'offset':  2.0/17.5,  'label': '2Hz / 6.5mm / +2mm R (interp)'},
]


# ── sinusoid fitting ──────────────────────────────────────────────────────────

def _sin_model(t, amp, freq, phase, offset):
    return amp * np.sin(2 * np.pi * freq * t + phase) + offset

def fit_sinusoid(signal, dt):
    N = len(signal)
    t = np.arange(N) * dt
    spec = np.abs(rfft(signal)); spec[0] = 0
    freqs = rfftfreq(N, d=dt)
    fb = int(np.argmax(spec))
    fg = float(freqs[fb]) if fb > 0 else 1.0
    try:
        popt, _ = curve_fit(
            _sin_model, t, signal,
            p0=[float(np.std(signal) * np.sqrt(2)), fg, 0.0, float(np.mean(signal))],
            bounds=([0, 0.2, -np.pi, -1.5], [np.pi, 10.0, np.pi, 1.5]), maxfev=8000,
        )
        a, f, p, o = popt
        return {'amp': float(abs(a)), 'freq': float(abs(f)), 'phase': float(p), 'offset': float(o)}
    except Exception:
        return {'amp': float(np.std(signal) * np.sqrt(2)), 'freq': fg, 'phase': 0.0,
                'offset': float(np.mean(signal))}


# ── simulation ────────────────────────────────────────────────────────────────

def _get_joint_indices(model, fish_id):
    qpos, qvel = [], []
    for jname in [f"headX_{fish_id}", f"headY_{fish_id}", f"headZ_{fish_id}",
                   f"headRollX_{fish_id}", f"headRollY_{fish_id}", f"headRollZ_{fish_id}",
                   f"motor_{fish_id}",
                   f"joint0_{fish_id}", f"joint1_{fish_id}", f"joint2_{fish_id}",
                   f"joint3_{fish_id}", f"joint4_{fish_id}"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1: continue
        qa, va = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        jt = model.jnt_type[jid]
        nq, nv = (7,6) if jt==mujoco.mjtJoint.mjJNT_FREE else (4,3) if jt==mujoco.mjtJoint.mjJNT_BALL else (1,1)
        qpos.extend(range(qa, qa+nq)); qvel.extend(range(va, va+nv))
    return qpos, qvel

def _make_xml(coef, base_config):
    cfg = copy.deepcopy(base_config)
    cfg['fluid_tuning']['coefficients'] = [float(c) for c in coef]
    with open(_TMP_CFG, 'w') as f: yaml.dump(cfg, f)
    sp = copy.deepcopy(_SP_TEMPLATE)
    sp["filename"] = _TMP_XML
    _generate_xml(sp, config_path=_TMP_CFG)


def _load_legacy_generator():
    """
    Exec git HEAD:geometry/auto_SOFI_field2sim.py in an isolated namespace
    to recover the old generate_xml and SYSTEMPARAMETERS (head=0.001g,
    bodyMass=0.6g, COM=[0.1284, 0, 0.0035]). No working-tree changes needed.
    """
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:geometry/auto_SOFI_field2sim.py"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        raise RuntimeError("git show failed — are you in a git repo?")
    ns = {"__file__": os.path.join(_REPO_ROOT, "geometry", "auto_SOFI_field2sim.py")}
    exec(compile(src, "auto_SOFI_field2sim.py@HEAD", "exec"), ns)
    return ns["generate_xml"], ns["SYSTEMPARAMETERS"]

def collect_trajectories(csv_path, base_config, coef, sinusoid_override=None):
    """
    Simulate one run. Returns (times, ghost_xy, twin_xy, wpos_loss).

    SysID Step A: teleport actuation, zero coupling (no EKF needed).
    ghost_xy: always real CSV positions (the real fish).
    twin_xy: simulated physics.
    sinusoid_override: if provided, both ghost motor AND twin motor are driven by the
        fitted sinusoid instead of the recorded Ghost_Alpha.  Ghost X/Y/Yaw are always real.
    """
    _make_xml(coef, base_config)
    df = pd.read_csv(csv_path)
    model = mujoco.MjModel.from_xml_path(_TMP_XML)
    data  = mujoco.MjData(model)
    f0_qp, f0_qv = _get_joint_indices(model, 0)
    f1_qp, f1_qv = _get_joint_indices(model, 1)
    mujoco.mj_forward(model, data)

    N, dt = len(df), model.opt.timestep
    times = np.arange(N) * dt

    if sinusoid_override is not None:
        so = sinusoid_override
        _sa  = so['amp'] * np.sin(2*np.pi*so['freq']*times + so['phase']) + so['offset']
        _sda = so['amp'] * 2*np.pi*so['freq'] * np.cos(2*np.pi*so['freq']*times + so['phase'])
    else:
        _sa = _sda = None

    ghost_xy = np.empty((N, 2))
    twin_xy  = np.empty((N, 2))

    for i, row in df.iterrows():
        g_alpha  = _sa[i]  if _sa  is not None else row['Ghost_Alpha']
        g_dalpha = _sda[i] if _sda is not None else row['Ghost_dAlpha']

        # Teleport ghost to real position, sinusoid/real motor
        data.qpos[f0_qp[0]], data.qpos[f0_qp[1]] = row['Ghost_X'],    row['Ghost_Y']
        data.qpos[f0_qp[5]], data.qpos[f0_qp[6]] = row['Ghost_Yaw'],  g_alpha
        data.qvel[f0_qv[0]], data.qvel[f0_qv[1]] = row['Ghost_vX'],   row['Ghost_vY']
        data.qvel[f0_qv[5]], data.qvel[f0_qv[6]] = row['Ghost_Omega'], g_dalpha

        # Twin: teleport motor, let physics run freely (stiffness=0)
        data.qpos[f1_qp[6]] = g_alpha
        data.qvel[f1_qv[6]] = g_dalpha
        data.ctrl[0] = g_alpha
        data.ctrl[1] = g_alpha

        mujoco.mj_step(model, data)
        ghost_xy[i] = [row['Ghost_X'], row['Ghost_Y']]
        twin_xy[i]  = [data.qpos[f1_qp[0]], data.qpos[f1_qp[1]]]

        if not np.isfinite(twin_xy[i]).all():
            return times[:i], ghost_xy[:i], twin_xy[:i], 1e6

    # Time-weighted position MSE (same formula as optimizer)
    w = np.exp(np.log(_START_WEIGHT) * (1.0 - times / max(times[-1], 1e-9)))
    w /= w.mean()
    wpos = float(np.mean(w * np.sum((ghost_xy - twin_xy)**2, axis=1)))
    return times, ghost_xy, twin_xy, wpos


def simulate_free_twin(base_config, coef, amp, freq, offset, duration=10.0):
    """
    Free-swim a single twin fish with sinusoidal motor teleport.  No CSV needed.
    Fish 0 (ghost) is pinned at origin; fish 1 (twin) swims freely.
    Returns (times, twin_xy).
    """
    _make_xml(coef, base_config)
    model = mujoco.MjModel.from_xml_path(_TMP_XML)
    data  = mujoco.MjData(model)
    f0_qp, f0_qv = _get_joint_indices(model, 0)
    f1_qp, f1_qv = _get_joint_indices(model, 1)
    mujoco.mj_forward(model, data)

    dt   = model.opt.timestep
    N    = int(duration / dt)
    times   = np.arange(N) * dt
    twin_xy = np.empty((N, 2))

    for i in range(N):
        t = i * dt
        alpha  = amp * np.sin(2 * np.pi * freq * t) + offset
        dalpha = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)

        # Ghost pinned at origin
        data.qpos[f0_qp[0]] = 0.0
        data.qpos[f0_qp[1]] = 0.0
        data.qpos[f0_qp[5]] = 0.0   # yaw
        data.qpos[f0_qp[6]] = alpha
        data.qvel[f0_qv[0]] = 0.0
        data.qvel[f0_qv[1]] = 0.0
        data.qvel[f0_qv[6]] = dalpha

        # Twin motor teleport
        data.qpos[f1_qp[6]] = alpha
        data.qvel[f1_qv[6]] = dalpha
        data.ctrl[0] = alpha
        data.ctrl[1] = alpha

        mujoco.mj_step(model, data)
        twin_xy[i] = [data.qpos[f1_qp[0]], data.qpos[f1_qp[1]]]

        if not np.isfinite(twin_xy[i]).all():
            return times[:i], twin_xy[:i]

    return times, twin_xy


# ── plotting ──────────────────────────────────────────────────────────────────

def _run_label(csv_path):
    for part in csv_path.split(os.sep):
        if part.startswith('run_'):
            return part
    return os.path.basename(os.path.dirname(os.path.dirname(csv_path)))

_PULLEY_MM = 12.5  # commanded tail angle [rad] = tendon travel [mm] / pulley radius [mm]

def _clean_run_title(label, loss):
    """Readable, usetex-safe panel title: '1.0 Hz, 0.72 rad  (wpos 0.013)'.

    The run's commanded drive amplitude is stored as tendon travel (mm); the
    physically meaningful drive is the tail-base joint angle, amp_rad = mm / 12.5.
    """
    import re
    fm = re.search(r"([\d.]+)\s*Hz", label)
    am = re.search(r"([\d.]+)\s*mm", label)
    freq = fm.group(1) if fm else "?"
    amp_rad = (float(am.group(1)) / _PULLEY_MM) if am else float("nan")
    return r"%s Hz, %.2f rad  ($\mathrm{wpos}=%.3f$)" % (freq, amp_rad, loss)


def plot_trajectory_grid(results, title, out_path):
    """results: list of dict with keys label, ghost_xy, twin_xy, loss, [sin_label]"""
    n = len(results)
    if n == 0:
        print(f"  (no runs to plot for: {title})"); return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    if _RS:
        # Report variant: build near final width (no heavy LaTeX downscaling) with
        # the small-panel ("dense") font tier so ticks/titles stay legible on the page.
        use_report_style(dense=True)
        fw = TEXTWIDTH_PT / 72.27 * 0.99
        fh = fw * (nrows / ncols) * 0.92
        data_lw, mk = 1.2, 3.2
    else:
        fw, fh = 5.5 * ncols, 4.5 * nrows
        data_lw, mk = 1.8, 5.0
    fig, axs = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)

    for idx, r in enumerate(results):
        ax = axs[idx // ncols][idx % ncols]
        g, tw = r['ghost_xy'], r['twin_xy']
        ax.plot(g[:,0],  g[:,1],  color=(CB['blue'] if _RS else 'steelblue'), lw=data_lw, label='Real fish', zorder=3)
        ax.plot(tw[:,0], tw[:,1], color=(CB['orange'] if _RS else 'tomato'), lw=data_lw, label='Twin (sim)', zorder=3, ls='--')
        ax.plot(g[0,0],  g[0,1],  'o', color=(CB['blue'] if _RS else 'steelblue'), ms=mk, zorder=5)
        ax.plot(tw[0,0], tw[0,1], 'o', color=(CB['orange'] if _RS else 'tomato'), ms=mk, zorder=5)
        loss = r.get('loss', float('nan'))
        amp_str = ("  [%.1fmm actual]" % r['achieved_mm']) if 'achieved_mm' in r else ""
        if _RS:
            ax.set_title(_clean_run_title(r['label'], loss))
        else:
            ax.set_title("%s%s\nwpos=%.4f" % (r['label'].replace("_"," "), amp_str, loss), fontsize=15)
        if 'times' in r:
            t8 = int(np.searchsorted(r['times'], 8.0))
            t8 = min(t8, len(r['ghost_xy']) - 1)
            if t8 > 0:
                ax.plot(r['ghost_xy'][t8, 0], r['ghost_xy'][t8, 1],
                        's', color=(CB['green'] if _RS else 'green'), ms=mk+2, zorder=6, label=r'$t=8$\,s' if _RS else 't=8s')
                ax.plot(r['twin_xy'][t8, 0], r['twin_xy'][t8, 1],
                        's', color=(CB['green'] if _RS else 'green'), ms=mk+2, zorder=6)
        ax.set_xlabel(r"$x$ [m]" if _RS else "X (m)"); ax.set_ylabel(r"$y$ [m]" if _RS else "Y (m)")
        ax.axis('equal'); ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(loc='best', fontsize=(None if _RS else 8))

    for idx in range(n, nrows * ncols):
        axs[idx // ncols][idx % ncols].set_visible(False)

    if _RS:
        fig.set_layout_engine("constrained")
        _rsave(fig, out_path[:-4] if out_path.endswith('.png') else out_path)
    else:
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_convergence_train_val(train_curve, val_curve, out_path, subtitle=""):
    fig, ax = plt.subplots(figsize=(11, 5))
    evals = list(range(1, len(train_curve) + 1))
    ax.plot(evals, train_curve, color='steelblue', lw=1.8, label='Training loss (best so far)')
    ax.plot(evals, val_curve,   color='tomato',    lw=1.8, ls='--', label='Sinusoid validation loss (best so far)')
    # Mark divergence points (val > train) with shading
    t = np.array(train_curve); v = np.array(val_curve)
    ax.fill_between(evals, t, v, where=(v > t), alpha=0.12, color='tomato', label='Generalisation gap')
    ax.set_xlabel("Cumulative evaluations"); ax.set_ylabel("Weighted position MSE")
    ax.set_title(f"Training vs Sinusoid Validation Loss\n{subtitle}", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_run_loss_bar(train_results, sin_results, test_results, out_path):
    """Bar chart comparing real-actuation vs sinusoid-actuation loss per run."""
    labels, real_losses, sin_losses = [], [], []
    for r_real, r_sin in zip(train_results, sin_results):
        mm_str = ("\n%.1fmm" % r_real['achieved_mm']) if 'achieved_mm' in r_real else ""
        labels.append(r_real['label'] + mm_str)
        real_losses.append(r_real['loss'])
        sin_losses.append(r_sin['loss'])

    test_labels, test_losses = [], []
    for r in test_results:
        mm_str = ("\n%.1fmm" % r['achieved_mm']) if 'achieved_mm' in r else ""
        test_labels.append(r['label'] + mm_str)
        test_losses.append(r['loss'])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 2.5*len(labels)), 5))
    w = 0.35
    bars1 = ax.bar(x - w/2, real_losses, w, label='Real Ghost_Alpha', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + w/2, sin_losses,  w, label='Fitted sinusoid',  color='tomato',    alpha=0.85)
    if test_losses:
        x2 = np.arange(len(labels), len(labels) + len(test_labels))
        ax.bar(x2, test_losses, w*1.2, label='Test runs (real)', color='mediumseagreen', alpha=0.85)
        ax.set_xticks(np.concatenate([x, x2]))
        ax.set_xticklabels(labels + test_labels, rotation=30, ha='right', fontsize=8)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)

    if len(labels) > 0:
        ax.axvline(len(labels) - 0.5, color='gray', ls='--', lw=1, alpha=0.5, label='train / test split')

    ax.set_ylabel("Weighted position MSE"); ax.set_title("Per-Run Loss: Train vs Sinusoid vs Test")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_freeswim_grid(results, title, out_path):
    """Twin-only free-swim trajectories (no ghost reference)."""
    n = len(results)
    if n == 0:
        print(f"  (no results for: {title})"); return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 4.0*nrows), squeeze=False)

    for idx, r in enumerate(results):
        ax = axs[idx // ncols][idx % ncols]
        tw = r['twin_xy']
        ax.plot(tw[:,0], tw[:,1], color='tomato', lw=1.8, label='Twin (sim)')
        ax.plot(tw[0,0], tw[0,1], 'o', color='tomato', ms=7, label='Start')
        ax.plot(tw[-1,0], tw[-1,1], 'x', color='darkred', ms=8,
                markeredgewidth=2, zorder=6, label='End (15s)')
        if 'times' in r and 'mark_t' in r:
            idx_t = int(np.searchsorted(r['times'], r['mark_t']))
            idx_t = min(idx_t, len(tw) - 1)
            ax.plot(tw[idx_t, 0], tw[idx_t, 1], 's', color='green',
                    ms=8, zorder=5, label=f"t={r['mark_t']:.0f}s")
        mm_str = ("\n%.1fmm commanded" % r['cmd_mm']) if 'cmd_mm' in r else ""
        ax.set_title("%s%s" % (r['label'], mm_str), fontsize=8)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.axis('equal'); ax.grid(True, alpha=0.3)
        if idx == 0: ax.legend(fontsize=7, loc='best')

    for idx in range(n, nrows * ncols):
        axs[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    if _RS:
        _rsave(fig, out_path[:-4] if out_path.endswith('.png') else out_path)
    else:
        fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_yaml", nargs='?', default=None,
                        help="Path to opt_result_*.yml. Omit to use most recent.")
    parser.add_argument("--config", default="sysid/fluid/field2sim.yml")
    parser.add_argument("--output_dir", default="outputs/sysid/eval")
    parser.add_argument("--skip_val_curve", action='store_true',
                        help="Skip convergence reconstruction (skips re-running sims, fast).")
    parser.add_argument("--legacy_mass", action='store_true',
                        help="Use git HEAD mass model (head=0.001g, bodyMass=0.6g, COM=[0.1284,0,0.0035]). "
                             "Required for evaluating E1-E8 results which were optimised with the old model.")
    args = parser.parse_args()

    global _generate_xml, _SP_TEMPLATE
    if args.legacy_mass:
        print("  [legacy_mass] Loading old mass model from git HEAD...")
        _generate_xml, _SP_TEMPLATE = _load_legacy_generator()
        print(f"  [legacy_mass] COM = {_SP_TEMPLATE['centerOfMass']}")
        print(f"  [legacy_mass] bodyMass = {_SP_TEMPLATE.get('bodyMass', '?')}")

    # ── find result YAML ──────────────────────────────────────────────────────
    if args.result_yaml:
        result_path = args.result_yaml
    else:
        candidates = sorted(glob.glob("outputs/sysid/opt_result_*.yml"))
        if not candidates:
            print("No opt_result_*.yml found in outputs/sysid/. Pass a path explicitly."); return
        result_path = candidates[-1]
    print(f"\nLoading: {result_path}")

    with open(result_path) as f:
        result = yaml.unsafe_load(f)
    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    best_coef  = result['best_coef']
    n_coef     = result.get('n_coef', len(best_coef))
    all_evals  = result.get('all_evaluations', [])
    train_dirs = result.get('run_dirs', [])

    train_csvs = [os.path.realpath(os.path.join(d, "processed", "03_field2sim_ready.csv"))
                  for d in train_dirs]
    train_csvs = [c for c in train_csvs if os.path.exists(c)]
    if not train_csvs:
        print("No training CSVs found (check run_dirs in result YAML)."); return

    # Find test runs = all available CSVs not used in training.
    # os.path.realpath on both sides so relative vs absolute paths match correctly.
    all_csvs  = [os.path.realpath(c) for c in sorted(glob.glob(_DATA_GLOB))]
    test_csvs = [c for c in all_csvs if c not in train_csvs]

    os.makedirs(args.output_dir, exist_ok=True)
    dt = base_config.get('environment', {}).get('dt', 0.001)

    print(f"\n  Training runs : {len(train_csvs)}")
    print(f"  Test runs     : {len(test_csvs)}")
    print(f"  Best coef     : {[f'{c:.4g}' for c in best_coef]}")
    print(f"  Best loss     : {result.get('best_loss', '?')}")

    # ── fit sinusoids for all CSVs ────────────────────────────────────────────
    print("\n[1/7] Fitting sinusoids...")
    sin_params = {}
    run_durations = {}
    for csv in train_csvs + test_csvs:
        df = pd.read_csv(csv)
        sp = fit_sinusoid(df['Ghost_Alpha'].values, dt)
        sin_params[csv] = sp
        run_durations[csv] = len(df) * dt
        print(f"  {_run_label(csv):40s}  A={sp['amp']:.3f}  f={sp['freq']:.2f}Hz  off={sp['offset']:+.3f}")

    # ── [1] training runs, real actuation ─────────────────────────────────────
    print("\n[2/7] Training runs — real Ghost_Alpha actuation...")
    train_real = []
    for csv in train_csvs:
        times, ghost, twin, loss = collect_trajectories(csv, base_config, best_coef)
        label = _run_label(csv)
        achieved_mm = sin_params[csv]['amp'] * _PULLEY_DISPLAY
        dur = run_durations[csv]
        print(f"  {label:40s}  wpos={loss:.5f}  actual={achieved_mm:.1f}mm  dur={dur:.1f}s")
        train_real.append({'label': label, 'ghost_xy': ghost, 'twin_xy': twin, 'loss': loss,
                           'achieved_mm': achieved_mm, 'times': times, 'duration': dur})

    plot_trajectory_grid(
        train_real,
        "Training Runs — Real Ghost_Alpha Actuation",
        os.path.join(args.output_dir, "trajectories_train_real.png"),
    )

    # ── [2] training runs, sinusoid actuation ─────────────────────────────────
    print("\n[3/7] Training runs — fitted sinusoid actuation...")
    train_sin = []
    for csv in train_csvs:
        sp = sin_params[csv]
        times, ghost, twin, loss = collect_trajectories(csv, base_config, best_coef, sinusoid_override=sp)
        label = _run_label(csv)
        achieved_mm = sp['amp'] * _PULLEY_DISPLAY
        dur = run_durations[csv]
        print(f"  {label:40s}  wpos={loss:.5f}  (A={sp['amp']:.3f} f={sp['freq']:.2f}Hz off={sp['offset']:+.3f}  actual={achieved_mm:.1f}mm  dur={dur:.1f}s)")
        train_sin.append({'label': f"{label}\n[sin A={sp['amp']:.3f} f={sp['freq']:.2f}Hz]",
                          'ghost_xy': ghost, 'twin_xy': twin, 'loss': loss,
                          'achieved_mm': achieved_mm, 'times': times, 'duration': dur})

    plot_trajectory_grid(
        train_sin,
        "Training Runs — Fitted Sinusoid Actuation  (Sinusoid Validation)",
        os.path.join(args.output_dir, "trajectories_train_sin.png"),
    )

    # ── [3] test runs, real actuation ─────────────────────────────────────────
    print(f"\n[4/7] Test runs ({len(test_csvs)}) — real Ghost_Alpha actuation...")
    test_real = []
    for csv in test_csvs:
        times, ghost, twin, loss = collect_trajectories(csv, base_config, best_coef)
        label = _run_label(csv)
        achieved_mm = sin_params[csv]['amp'] * _PULLEY_DISPLAY
        dur = run_durations[csv]
        print(f"  {label:40s}  wpos={loss:.5f}  actual={achieved_mm:.1f}mm  dur={dur:.1f}s")
        test_real.append({'label': label, 'ghost_xy': ghost, 'twin_xy': twin, 'loss': loss,
                          'achieved_mm': achieved_mm, 'times': times, 'duration': dur})

    if test_real:
        plot_trajectory_grid(
            test_real,
            "Test Runs (Unseen) — Real Ghost_Alpha Actuation",
            os.path.join(args.output_dir, "trajectories_test_real.png"),
        )

    # ── [4] test runs, sinusoid actuation ─────────────────────────────────────
    print(f"\n[5/7] Test runs ({len(test_csvs)}) — fitted sinusoid actuation...")
    test_sin = []
    for csv in test_csvs:
        sp = sin_params[csv]
        times, ghost, twin, loss = collect_trajectories(csv, base_config, best_coef, sinusoid_override=sp)
        label = _run_label(csv)
        achieved_mm = sp['amp'] * _PULLEY_DISPLAY
        dur = run_durations[csv]
        print(f"  {label:40s}  wpos={loss:.5f}  (A={sp['amp']:.3f} f={sp['freq']:.2f}Hz off={sp['offset']:+.3f}  actual={achieved_mm:.1f}mm  dur={dur:.1f}s)")
        test_sin.append({'label': f"{label}\n[sin A={sp['amp']:.3f} f={sp['freq']:.2f}Hz]",
                         'ghost_xy': ghost, 'twin_xy': twin, 'loss': loss,
                         'achieved_mm': achieved_mm, 'times': times, 'duration': dur})

    if test_sin:
        plot_trajectory_grid(
            test_sin,
            "Test Runs (Unseen) — Fitted Sinusoid Actuation",
            os.path.join(args.output_dir, "trajectories_test_sin.png"),
        )

    # ── [5] per-run loss bar chart ─────────────────────────────────────────────
    print("\n[6/7] Per-run loss bar chart...")
    plot_per_run_loss_bar(
        train_real, train_sin, test_real,
        os.path.join(args.output_dir, "per_run_loss_bar.png"),
    )

    # ── [6] interpolated free-swim ────────────────────────────────────────────
    interp_specs = _INTERP_STRAIGHT + _INTERP_TURNING
    print(f"\n[7/7] Interpolated free-swim — {len(interp_specs)} synthetic conditions...")
    interp_results = []
    freeswim_duration = 15.0  # fixed 15s; real runs range from 8–21s
    _mark_t = 8.0             # annotate position at this time on each trajectory
    for spec in interp_specs:
        label = spec['label']
        times_fs, twin_xy = simulate_free_twin(
            base_config, best_coef,
            amp=spec['amp'], freq=spec['freq'], offset=spec['offset'],
            duration=freeswim_duration,
        )
        cmd_mm = spec['amp'] * _PULLEY_DISPLAY
        dist15 = float(np.linalg.norm(twin_xy[-1] - twin_xy[0])) if len(twin_xy) > 1 else 0.0
        idx8  = int(np.searchsorted(times_fs, _mark_t))
        dist8 = float(np.linalg.norm(twin_xy[min(idx8, len(twin_xy)-1)] - twin_xy[0])) if len(twin_xy) > 1 else 0.0
        print(f"  {label:45s}  cmd={cmd_mm:.1f}mm  dist@8s={dist8:.3f}m  dist@15s={dist15:.3f}m")
        interp_results.append({'label': label, 'twin_xy': twin_xy, 'cmd_mm': cmd_mm,
                               'times': times_fs, 'mark_t': _mark_t})

    all_durs   = list(run_durations.values())
    train_durs = [run_durations[c] for c in train_csvs]
    test_durs  = [run_durations[c] for c in test_csvs if c in run_durations]
    dur_note   = (f"Real runs — train: {min(train_durs):.1f}–{max(train_durs):.1f}s"
                  + (f"  |  test: {min(test_durs):.1f}–{max(test_durs):.1f}s" if test_durs else ""))

    plot_freeswim_grid(
        interp_results,
        f"Interpolated Conditions — Twin Free-Swim (no real data)\n{dur_note}",
        os.path.join(args.output_dir, "trajectories_interpolated.png"),
    )

    # ── [Bonus] convergence: train loss vs sinusoid validation ────────────────
    if args.skip_val_curve or not all_evals:
        if not all_evals:
            print("\n  (no all_evaluations in result YAML — skipping convergence reconstruction)")
    else:
        print(f"\n[Bonus] Reconstructing train/val convergence from {len(all_evals)} evaluations...")
        print(        "        (re-simulates each new-best with sinusoid inputs — ~1–2 min)")
        train_curve, val_curve = [], []
        best_train, current_val = np.inf, np.inf
        new_bests = 0

        for ev_i, ev in enumerate(all_evals):
            params = ev['params']
            coef   = [params[f'c{i}'] for i in range(n_coef)]
            loss   = float(ev['loss'])

            if loss < best_train:
                best_train = loss
                new_bests += 1
                # Re-evaluate with pure sinusoid (no blending) → true validation signal
                val_losses = []
                for csv in train_csvs:
                    sp = sin_params.get(csv)
                    if sp:
                        _, _, _, wl = collect_trajectories(csv, base_config, coef, sinusoid_override=sp)
                        if wl < 1e5:
                            val_losses.append(wl)
                if val_losses:
                    current_val = float(np.mean(val_losses))
                print(f"  eval {ev_i+1:4d}: train={best_train:.5f}  val_sin={current_val:.5f}"
                      f"  gap={current_val - best_train:+.5f}")

            train_curve.append(best_train)
            val_curve.append(current_val)

        print(f"\n  {new_bests} new-best re-evaluations completed.")
        print(f"  Final train:  {train_curve[-1]:.5f}")
        print(f"  Final val:    {val_curve[-1]:.5f}")
        print(f"  Final gap:    {val_curve[-1] - train_curve[-1]:+.5f}")

        subtitle = f"n_coef={n_coef} | {os.path.basename(result_path)} | {len(train_csvs)} training runs"
        plot_convergence_train_val(
            train_curve, val_curve,
            os.path.join(args.output_dir, "convergence_train_val.png"),
            subtitle=subtitle,
        )

    # ── cleanup ───────────────────────────────────────────────────────────────
    for p in [_TMP_CFG, _TMP_XML]:
        if os.path.exists(p):
            os.remove(p)

    print(f"\nAll outputs saved to: {args.output_dir}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'Run':<42} {'Real wpos':>10} {'Sin wpos':>10}  note")
    print("="*80)
    for r_real, r_sin in zip(train_real, train_sin):
        gap = r_sin['loss'] - r_real['loss']
        print(f"  {r_real['label']:<40} {r_real['loss']:>10.5f} {r_sin['loss']:>10.5f}  (gap {gap:+.5f})  [train]")
    if test_real:
        print("-"*80)
        for r_real, r_sin in zip(test_real, test_sin):
            gap = r_sin['loss'] - r_real['loss']
            print(f"  {r_real['label']:<40} {r_real['loss']:>10.5f} {r_sin['loss']:>10.5f}  (gap {gap:+.5f})  [test]")
    print("="*80)


if __name__ == "__main__":
    main()
