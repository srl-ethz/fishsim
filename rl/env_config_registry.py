import gymnasium as gym

# CPG chain (hybrid CPG-position, 6-D action) — hardware-proven
from .environments.cpg import env_sofi_phase1_1_noiseAndDelay
from .environments.cpg import env_sofi_phase1_2_physicsDR
from .environments.cpg import env_sofi_phase2_physicsDR_polarTarget
from .environments.cpg import env_sofi_full_curriculum_cpg

# Direct-Action chain (1-D motor setpoint) — latest, simulation only
from .environments.da import env_sofi_da_phase1_1_noiseAndDelay as env_sofi_da_p1_1
from .environments.da import env_sofi_da_phase1_2_physicsDR as env_sofi_da_p1_2
from .environments.da import env_sofi_full_curriculum_da as env_sofi_da_motorid_v1
from .environments.da import env_sofi_full_curriculum_da_e13a as env_sofi_da_motorid_v1_e13a

# Physics models — one frozen generator per direction (see CANONICAL_POLICIES.md)
from geometry.auto_SOFI_cpg import SYSTEMPARAMETERS as SOFI_CPG_PARAMS
from geometry.auto_SOFI_da import SYSTEMPARAMETERS as SOFI_DA_PARAMS

ENVIRONMENT_REGISTRY = {
    # ── CPG direction ────────────────────────────────────────────────────────
    "sofi_phase1_1_noiseAndDelay": {
        "entry_point": env_sofi_phase1_1_noiseAndDelay.FishEnv,
        "system_parameters": SOFI_CPG_PARAMS,
    },
    "sofi_phase1_2_physicsDR": {
        "entry_point": env_sofi_phase1_2_physicsDR.FishEnv,
        "system_parameters": SOFI_CPG_PARAMS,
    },
    "sofi_phase2_physicsDR_polarTarget": {
        "entry_point": env_sofi_phase2_physicsDR_polarTarget.FishEnv,
        "system_parameters": SOFI_CPG_PARAMS,
    },
    "sofi_full_curriculum_cpg": {  # CANONICAL CPG — produces stage8_lateral policy
        "entry_point": env_sofi_full_curriculum_cpg.FishEnv,
        "system_parameters": SOFI_CPG_PARAMS,
    },
    # ── Direct-Action direction ──────────────────────────────────────────────
    "sofi_da_phase1_1_noiseAndDelay": {
        "entry_point": env_sofi_da_p1_1.FishEnv,
        "system_parameters": SOFI_DA_PARAMS,
    },
    "sofi_da_phase1_2_physicsDR": {
        "entry_point": env_sofi_da_p1_2.FishEnv,
        "system_parameters": SOFI_DA_PARAMS,
    },
    "sofi_full_curriculum_da": {  # CANONICAL DA — produces sofi_da_rl_policy
        "entry_point": env_sofi_da_motorid_v1.FishEnv,
        "system_parameters": SOFI_DA_PARAMS,
    },
    "sofi_full_curriculum_da_e13a": {  # same motor/curriculum, E13a fluid (A/B vs refitB)
        "entry_point": env_sofi_da_motorid_v1_e13a.FishEnv,
        "system_parameters": SOFI_DA_PARAMS,
    },
}

def register_environments(max_episode_steps=4000, reward_threshold=1000000.0):
    """Registers all environments in the registry with Gymnasium."""
    for env_id, config in ENVIRONMENT_REGISTRY.items():
        full_id = f"gymnasium_env/{env_id}"
        if full_id in gym.registry:
            continue
        gym.register(
            id=full_id,
            entry_point=config["entry_point"],
            max_episode_steps=max_episode_steps,
            reward_threshold=reward_threshold,
        )

def get_system_parameters(env_id):
    """Returns a copy of the system parameters for a given environment ID."""
    if env_id in ENVIRONMENT_REGISTRY:
        return ENVIRONMENT_REGISTRY[env_id]["system_parameters"].copy()
    raise ValueError(f"Environment ID '{env_id}' not found in registry.")
