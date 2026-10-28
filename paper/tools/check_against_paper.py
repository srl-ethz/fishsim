#!/usr/bin/env python3
"""Verify every regenerated figure matches the asset in the paper's LaTeX project.

    python paper/tools/check_against_paper.py path/to/paper/Images   (or set PAPER_IMAGES)

Compares extracted text and byte size. Exits non-zero on any mismatch.
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
import os
PAPER = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_IMAGES", ROOT / "reference"))
# The paper's figures, numbered as in the report. Every stem is out/fig<N>_<name>.pdf, built by
# `make figures`. Keep this list in step with the paper: an entry the paper has dropped is worse
# than no entry, because it reports ALL MATCH while the figures actually in the paper go unchecked.
PAIRS = ["fig1_overview",            # fig1_overview.py --variant compact (SVG -> PDF via Chrome/Edge)
         "fig2_training_pipeline",   # fig2_training_pipeline.py
         "fig3_experimental_setup",  # fig3_experimental_setup.py (SVG -> PDF via Chrome/Edge)
         "fig4_tailstiff",           # fig_tailstiff_paper.py
         "fig5_fluidfit",            # fig5_fluidfit_paper.py --pair
         "fig6_motor_overlay",       # fig6_motor_overlay_paper.py
         "fig7_deploy_collage",      # fig_deploy_collage_paper.py (divider variant, the default)
         "fig8_reach_collage",       # fig_reach_collage_paper.py
         "fig9_velspectrum",         # fig_velspectrum.py (with the deployed runs; --no-hardware drops them)
         "fig10_gait_traces"]        # fig_gaits_traces_paper.py

# Used by the paper but hand-made: nothing to regenerate. (Empty since Fig. 3 got a generator.)
STATIC = []


def text(p):
    try: return subprocess.run(["pdftotext", str(p), "-"], capture_output=True).stdout
    except FileNotFoundError: return None


bad = 0
for stem in PAIRS + STATIC:
    mine = ROOT / "out" / f"{stem}.pdf"
    ref = next((c for c in (PAPER / "final" / f"{stem}.pdf", PAPER / "draft" / f"{stem}.pdf", PAPER / f"{stem}.pdf") if c.exists()), None)
    tag = "STATIC" if stem in STATIC else ""
    if not mine.exists(): print(f"  MISSING   {stem}"); bad += 1; continue
    if ref is None:       print(f"  no ref    {stem} (regenerated {mine.stat().st_size} B) {tag}"); continue
    same = mine.stat().st_size == ref.stat().st_size and text(mine) == text(ref)
    print(f"  {'IDENTICAL' if same else 'MISMATCH ':<10} {stem} {tag}")
    bad += 0 if same else 1
print(f"\n{'ALL MATCH' if not bad else f'{bad} PROBLEM(S)'}")
sys.exit(1 if bad else 0)
