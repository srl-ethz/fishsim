################################################################################
# Run hyperparameter optimization for the force mapping model. Configurations 
# should be defined in train_force_mapping.py, with hyperparameters to optimize
# being defined in the config_sweep_force.yaml file.
################################################################################

import os, sys, yaml, traceback, argparse

def _log_base_dir():
    if os.environ.get("SLURM_JOB_ID"):
        return f"/cluster/scratch/{os.environ.get('USER', 'lmaloney')}/sofiRL/outputs"
    return "outputs"
import wandb
import numpy as np

# Allow running this script directly (e.g. `python rl/opt_hyperRL.py`) by
# making the project root importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.env_config_registry import get_system_parameters
from rl import train_rl, train_rl_curriculum


# Set wandb environment variable
os.environ["WANDB_MODE"] = "online"


def run_sweep():
    try:
        # 1. Initialize WandB FIRST so it grabs the sweep parameters (like 'env')
        wandb.init()

        # 2. Check which environment WandB decided to use for this specific run
        env_name = wandb.config.get("env")

        # 3. Get system parameters from the centralized registry
        SYSTEMPARAMETERS = get_system_parameters(env_name)

        # 4. Prepare the base configs
        configs = SYSTEMPARAMETERS.copy()
        configs['eval'] = False
        configs['load'] = False

        # 5. Update the active WandB config with these base parameters
        # (This correctly populates the WandB dashboard with the right fluid stats)
        wandb.config.update(configs)

        # 6. Read network architecture and ensure they are integers
        hiddenDim = wandb.config.get("hiddenDim")
        numLayers = wandb.config.get("numLayers")
        if hiddenDim is None or numLayers is None:
            raise ValueError(
                f"Sweep config is missing required fields: "
                f"hiddenDim={hiddenDim}, numLayers={numLayers}. "
                "Add them to the sweep YAML."
            )
        hiddenDim = int(hiddenDim)
        numLayers = int(numLayers)
        
        wandb.config.update({
            "net_arch": [hiddenDim] * numLayers,
            "logDir": f"{_log_base_dir()}/{wandb.run.id}"
        })

        ### Run training and testing
        if "curriculum" in env_name or "phase2" in env_name or "sofi_da" in env_name:
            log = train_rl_curriculum.main(wandb.config)
        else:
            log = train_rl.main(wandb.config)

        # Some training entrypoints (e.g., the curriculum runner) may return None.
        # Recover rewards/figures from the log directory when possible so sweeps
        # can still record summary metrics.
        if log is None:
            try:
                from stable_baselines3.common.results_plotter import load_results, ts2xy
                log_dir = wandb.config.get("logDir", f"outputs/{wandb.run.id}")
                x, y = ts2xy(load_results(log_dir), "timesteps")
                rewards = np.array(y) if len(y) > 0 else np.array([])
                figs = []
                dists = np.array([])
                log = {"rewards": rewards, "dists": dists, "figs": figs}
            except Exception:
                # Fallback to empty results
                log = {"rewards": np.array([]), "dists": np.array([]), "figs": []}

        # Safe logging: handle empty arrays
        reward_mean = float(np.mean(log['rewards'])) if getattr(log['rewards'], 'size', 0) else float('nan')
        reward_std = float(np.std(log['rewards'])) if getattr(log['rewards'], 'size', 0) else float('nan')
        dists_mean = float(np.mean(log['dists'])) if getattr(log['dists'], 'size', 0) else float('nan')
        dists_std = float(np.std(log['dists'])) if getattr(log['dists'], 'size', 0) else float('nan')

        wandb.log({
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "dists_mean": dists_mean,
            "dists_std": dists_std,
            "Test Trajectories": log['figs'],
        })

    except:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for fish swimming.")
    parser.add_argument('--sweepId', type=str, default=None, help='Sweep ID to continue running previous WandB sweep')
    parser.add_argument('--sweepConfig', type=str, default="rl/sweepconfigs/sweep_cpg.yml", help='Config file for hyperparameter sweep')
    parser.add_argument('--count', type=int, default=None, help='Number of runs to execute before agent exits. Default is infinite.')
    args = parser.parse_args()

    # Continue running existing sweeps
    if args.sweepId is not None:
        sweepId = args.sweepId
    else:
        with open(args.sweepConfig, 'r', encoding='utf-8') as stream:
            sweepConfig = yaml.safe_load(stream)
            print(sweepConfig)
        env_val    = sweepConfig['parameters']['env']['value']
        policy_val = sweepConfig['parameters'].get('policy', {}).get('value', 'SAC')
        sweepConfig['name'] = f"sweep_{env_val}_{policy_val}"
        sweepId = wandb.sweep(sweepConfig, entity=sweepConfig.get("entity", "srl_ethz"))
    # Multiple machines can run the same sweep
    print(f"### Sweep ID: {sweepId}")
    ### The configs from wandb are updated with sweep config!
    wandb.agent(sweepId, function=run_sweep, count=args.count)

