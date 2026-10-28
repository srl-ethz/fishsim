"""Where the paper scripts find their inputs.

Simulation-side inputs (model, meshes, policies, cached rollouts, static images) live in this
repository. Measured data lives in the fishdeploy repository (github.com/Ramchi99/fishdeploy),
expected next to this one or at $FISHDEPLOY_ROOT.

Importing this module also puts paper/ and the repository root on sys.path, so a figure script can
`import paper_style` and `from geometry.auto_SOFI_cpg import ...` wherever it is started from.
"""
import os
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent                 # fishsim/paper
REPO = PAPER.parent                                     # fishsim
FISHDEPLOY = Path(os.environ.get("FISHDEPLOY_ROOT", REPO.parent / "fishdeploy")).resolve()

OUT = PAPER / "out"                                     # generated figures (gitignored)
STATIC = PAPER / "static"                               # photographs, renders, stills
DATA = PAPER / "data"                                   # cached simulation inputs (npz)
GEOMETRY = REPO / "geometry"                            # SOFI_cpg.xml, SOFI_da.xml, meshes/
POLICIES = REPO / "policies"

FIELD = FISHDEPLOY / "field_test" / "data" / "20260401_first_swim"   # the 12 open-loop runs
DEPLOY_RUNS = FISHDEPLOY / "deploy" / "runs"                         # the 43 deployment runs
ANALYSIS = FISHDEPLOY / "analysis"                                   # caches, strobes, geometry
TAIL_BENCH = FISHDEPLOY / "sysid" / "tail" / "data"                  # the air/water sweeps

for _p in (str(PAPER), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def require_fishdeploy() -> Path:
    """Fail early, with the fix, when the measured data is not where we expect it."""
    if not FIELD.is_dir() or not DEPLOY_RUNS.is_dir():
        raise SystemExit(
            f"fishdeploy data not found under {FISHDEPLOY}.\n"
            "Clone https://github.com/Ramchi99/fishdeploy next to this repository, "
            "or point FISHDEPLOY_ROOT at your clone.")
    return FISHDEPLOY
