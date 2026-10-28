#!/usr/bin/env python3
"""Fig 4(c) (Re H vs omega^2 intercept) at the PAPER's column width.

Derived from fishCREATE/digital_twin/tail_sysid/plot_keff_presentation.py --light.
Only the set_size reference width and the output path differ, so the report asset
(325 pt, printing at 4.5 pt in our column) is untouched.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from the existing tail_sysid framework
from _paths import REPO as _REPO, OUT as _OUT
sys.path.insert(0, str(_REPO / "sysid" / "tail_stiffness"))
from step1_stiffness import prescreen_condition, process_file, ALL_AMP_FRACS
from utils import fit_spring_inertia
from paper_style import use_paper_style, size, save, CB, LW, EMPH_LW

PLOT_DIR = str(_OUT)
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true",
                    help="light theme (report); default is the dark presentation theme")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    txt = "black" if args.light else "white"
    print("Gathering data for presentation plot...")
    stiff_records = []
    
    # We'll use all three tendon travel settings to get the global k_eff
    for d_mm in [5.0, 7.0, 9.0]:
        for freq_hz in [1.0, 2.0, 3.0, 4.0]:
            for amp_pct in ALL_AMP_FRACS:
                pk, pc, flags, m = prescreen_condition("air", d_mm, freq_hz, amp_pct, slack_mm=2.5)
                if pk: # only stiffness-passing (no slack)
                    r = process_file("air", d_mm, freq_hz, amp_pct, use_cycle_avg=True)
                    if r: stiff_records.append(r)

    if not stiff_records:
        print("No valid stiffness records found!")
        return

    # Regress to find k_eff (y-intercept) and I_eff (slope)
    omega_sq = np.array([r["omega_sq"] for r in stiff_records])
    real_H   = np.array([r["H"].real    for r in stiff_records])
    kd, Id = fit_spring_inertia(omega_sq, real_H)

    # Styling: light (report) or dark (presentation)
    if args.light:
        use_paper_style()
    else:
        plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(size(1.0, aspect=0.62) if args.light else (8, 5)),
                           layout="constrained")
    
    # data points coloured by drive amplitude (sequential) -> colourbar, not a discrete legend
    allx = np.array([r["omega_sq"] for r in stiff_records])
    ally = np.array([r["H"].real   for r in stiff_records])
    allamp = np.array([r["amp_pct"] for r in stiff_records])
    sc = ax.scatter(allx, ally, c=allamp, cmap="viridis", s=24, alpha=0.95,
                    edgecolor="white", linewidth=0.25, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"drive amplitude (\%)"); cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)

    # regression fit (Re(H) = k_eff - I_eff w^2)
    w2 = np.linspace(0, max(omega_sq) * 1.1, 60)
    ax.plot(w2, kd - Id * w2, color=CB["orange"], lw=EMPH_LW, zorder=2)

    # y-intercept = k_eff: subtle marker + guide line to the axis + a neat label (no arrow)
    ax.plot([0, 0], [0, 1.34], color=CB["orange"], ls=":", lw=LW, zorder=1)
    ax.plot([0], [kd], "o", color=CB["orange"], ms=4.5, mec="white", mew=0.8, zorder=5)
    ax.text(30, 1.41, r"$k_\mathrm{eff}=%.2f$" % kd,
            color=CB["orange"], ha="left", va="center")

    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlabel(r"frequency squared $\omega^2$  [rad$^2$/s$^2$]")
    ax.set_ylabel(r"$\mathrm{Re}(H)$  [N\,m/rad]")
    ax.set_xlim(-12, max(omega_sq) * 1.12)
    ax.set_ylim(-1.05, 1.54)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    # suptitle (not fig.text) so constrained layout reserves room for it, and x=0
    # so the panel letter starts at the column's left edge like (a) and (b)
    fig.suptitle(r"\textbf{(c)} intercept test: $\mathrm{Re}\,H$ against $\omega^2$",
                 x=0.0, ha="left")

    default_name = ("s1_real_H_vs_omega2_light.png" if args.light
                    else "s1_real_H_vs_omega2_presentation_dark.png")
    if args.light:
        save(fig, "fig4c_keff_intercept_paper", expect="col")
    else:
        plt.savefig(args.out or os.path.join(PLOT_DIR, default_name), dpi=200); print("Saved dark")

if __name__ == "__main__":
    main()
