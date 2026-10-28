"""DA full-curriculum env, E13a-fluid variant.

Identical to env_sofi_full_curriculum_da (same re-identified motor model,
same 10-stage curriculum, same LPF support) EXCEPT the fluid model: this one
uses the **E13a** 15-coef vector (the field-SysID fluid the CPG/SDA used)
instead of the default refitB_fwdonly. Used to A/B the fluid model under the
4.5 Hz setpoint LPF.

The fluid swap is done via the FLUID_COEF class attribute, which the base
phase1_1 env applies when generating the MuJoCo XML.
"""

from gymnasium import utils

from .env_sofi_full_curriculum_da import FishEnv as _DAFull

# E13a (verified = the CPG/SDA training fluid; see CANONICAL_POLICIES.md).
_E13A = [0.005822, 0.044600, 0.812577, 0.521112, 0.902770,
         0.039474, 0.551234, 5.427544, 0.000956, 1.581176,
         0.170727, 0.003855, 18.310332, 10.148229, 5.951302]


class FishEnv(_DAFull, utils.EzPickle):
    """DA full curriculum with the E13a fluid model (everything else identical)."""
    FLUID_COEF = _E13A
