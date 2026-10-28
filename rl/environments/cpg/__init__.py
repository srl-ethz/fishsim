"""CPG direction — hybrid CPG-position envs (6-D action, 24-D obs). Hardware-proven.

Chain: env_sofi (BaseFishEnv) + cpg_oscillator
  → env_sofi_phase1_1_noiseAndDelay → env_sofi_phase1_2_physicsDR
  → env_sofi_phase2_physicsDR_polarTarget → env_sofi_full_curriculum_cpg (trainable).
"""

from . import env_sofi
from . import cpg_oscillator
from . import env_sofi_phase1_1_noiseAndDelay
from . import env_sofi_phase1_2_physicsDR
from . import env_sofi_phase2_physicsDR_polarTarget
from . import env_sofi_full_curriculum_cpg
