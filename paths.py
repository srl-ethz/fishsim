"""Where fishsim finds its own files and the measured data.

Measured data lives in the fishdeploy repository (github.com/Ramchi99/fishdeploy), expected next to this
one or at $FISHDEPLOY_ROOT, under its `data/` tree (which mirrors the public archive on OneDrive).
Importing this module puts the repository root on sys.path.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
FISHDEPLOY = Path(os.environ.get("FISHDEPLOY_ROOT", REPO.parent / "fishdeploy")).resolve()

GEOMETRY = REPO / "geometry"                  # sofi.py → SOFI.xml, sofi_motorid.py → SOFI_motorid.xml, sofi_twin.py, meshes/
POLICIES = REPO / "policies"

DATA = FISHDEPLOY / "data"
FIELD = DATA / "01_field_openloop_20260401"                 # the open-loop field runs, run_01 … run_13
DEPLOY_RUNS = DATA / "02_deploy_20260520"                    # the closed-loop deployment runs
ANALYSIS = DEPLOY_RUNS / "analysis"                          # tracked trajectories, reach geometry, strobes
CANONICAL = DEPLOY_RUNS / "canonical"                        # curated 1 kHz segments + manifest of the closed-loop fits
TAIL_BENCH = DATA / "03_sysid" / "tail_benchmark" / "air"    # the 100 r17.5 air sweeps of the tail-stiffness fit

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def paper_runs() -> list:
    """The 43 deployment runs of the paper, in recording order, from fishdeploy's paper_runs.txt.
    (The folder holds two more runs whose segments only the closed-loop fits use.)"""
    names = [l.strip() for l in (DEPLOY_RUNS / "paper_runs.txt").read_text().splitlines() if l.strip()]
    return [DEPLOY_RUNS / n for n in names]


def require_fishdeploy() -> Path:
    """Fail early, with the fix, when the measured data is not where we expect it."""
    if not FIELD.is_dir() or not DEPLOY_RUNS.is_dir():
        raise SystemExit(
            f"fishdeploy data not found under {FISHDEPLOY}.\n"
            "Clone https://github.com/Ramchi99/fishdeploy next to this repository, "
            "or point FISHDEPLOY_ROOT at your clone.")
    return FISHDEPLOY
