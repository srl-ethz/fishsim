"""SOFI RL environments — two canonical curriculum directions in cpg/ and da/.

cpg/ — hybrid CPG-position, 6-D action, hardware-proven (20/05/2026).
da/  — Direct Action, 1-D motor setpoint, latest, simulation only.

Each subpackage holds its full inheritance chain ending in the trainable
`env_sofi_full_curriculum_{cpg,da}`. Deprecated experiments are parked in
environments/_archive/ (not registered). See CANONICAL_POLICIES.md.
"""

from . import cpg
from . import da

__all__ = ["cpg", "da"]
