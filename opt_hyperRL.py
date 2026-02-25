################################################################################
# Run hyperparameter optimization for the force mapping model. Configurations 
# should be defined in train_force_mapping.py, with hyperparameters to optimize
# being defined in the config_sweep_force.yaml file.
################################################################################

import os, yaml, traceback, argparse
import wandb
import numpy as np

from Geometry.auto_tendonFish import SYSTEMPARAMETERS
import train_rl


# Set wandb environment variable
os.environ["WANDB_MODE"] = "online"


def run_sweep ():
    configs = SYSTEMPARAMETERS.copy()
    try:
        configs['eval'] = False
        configs['load'] = False

        wandb.init(config=configs)

        # Read current network architecture, and create list for train.
        hiddenDim = wandb.config.get("hiddenDim")
        numLayers = wandb.config.get("numLayers")
        wandb.config["net_arch"] = [hiddenDim] * numLayers
        wandb.config["logDir"] = f"Outputs/{wandb.run.id}"

        ### Run training and testing
        log = train_rl.main(wandb.config)

        wandb.log({
            "reward_mean": np.mean(log['rewards']),
            "reward_std": np.std(log['rewards']),
            "dists_mean": np.mean(log['dists']),
            "dists_std": np.std(log['dists']),
            "Test Trajectories": log['figs'],
        })

    except:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for fish swimming.")
    parser.add_argument('--sweepId', type=str, default=None, help='Sweep ID to continue running previous WandB sweep')
    parser.add_argument('--sweepConfig', type=str, default="Environments/sweepconfig_SAC.yml", help='Config file for hyperparameter sweep')
    args = parser.parse_args()

    with open(args.sweepConfig, 'r', encoding='utf-8') as stream:
        try:
            sweepConfig = yaml.safe_load(stream)
            print(sweepConfig)
        except yaml.YAMLError as exc:
            print(exc)

    sweepConfig['name'] = f"sweep_{sweepConfig['parameters']['env']['value']}_{sweepConfig['parameters']['policy']['value']}"

    # Continue running existing sweeps
    if args.sweepId is not None:
        sweepId = args.sweepId
    else:
        sweepId = wandb.sweep(sweepConfig, entity='srl_ethz')
    # Multiple machines can run the same sweep
    print(f"### Sweep ID: {sweepId}")
    ### The configs from wandb are updated with sweep config!
    wandb.agent(sweepId, function=run_sweep)