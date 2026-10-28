"""Paths for the paper scripts: the repository's `paths.py` plus the figure folders.

Importing this module puts paper/ and the repository root on sys.path, so a figure script can
`import paper_style` and `from geometry.sofi import ...` wherever it is started from.
"""
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent                 # fishsim/paper
_root = PAPER.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from paths import (REPO, FISHDEPLOY, GEOMETRY, POLICIES, FIELD, DEPLOY_RUNS,  # noqa: E402,F401
                   ANALYSIS, CANONICAL, TAIL_BENCH, require_fishdeploy, paper_runs)

OUT = PAPER / "out"                                     # generated figures (gitignored)
STATIC = PAPER / "static"                               # photographs, renders, stills
DATA = PAPER / "data"                                   # cached simulation inputs (npz)
