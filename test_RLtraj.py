################################################################################
# Test trained RL agent on tracking waypoints.
################################################################################

import os
import time

import scipy.optimize
from bayes_opt import BayesianOptimization

import cv2
import numpy as np
import gymnasium as gym

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

os.environ["WANDB_MODE"] = "offline"

import Environments.env_fish_target_relative as env_fish_target_relative

SEED = 42

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})

mm = 1 / 25.4



def main(args, visualize=False, numWaypoints=20, period=400):
    # Create log dir
    log_dir = "Outputs"
    os.makedirs(log_dir, exist_ok=True)
    if "logDir" in args:
        log_dir = args.logDir
        os.makedirs(log_dir, exist_ok=True)

    print(args)

    # Register environment
    if args.env == "fish_target_relative-v0":
        gym.register(
            id="gymnasium_env/fish_target_relative-v0",
            entry_point=env_fish_target_relative.FishEnv,
            max_episode_steps=args.maxEpisodeSteps,
            reward_threshold=1000000.0,
        )
    else:
        raise NotImplementedError(f"Environment {args.env} not implemented.")

    env = gym.make(f"gymnasium_env/{args.env}", reset_noise_scale=0.0, frame_skip=20, render_mode="rgb_array", motorVelRange=args.motorVelRange, actionMultiplier=args.actionMultiplier, rewardFailure=args.rewardFailure, seed=SEED, noterminate=True, width=800, height=800)
    check_env(env)
    env = Monitor(env, log_dir)


    # Load model
    print(f"Using policy {args.policy} for environment {args.env}.")

    model = None
    if args.policy == "SAC":
        model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": args.net_arch}, train_freq=args.train_freq, learning_starts=args.learningStarts, learning_rate=args.learningRate, batch_size=args.batchSize, gamma=args.gamma, tau=args.tau, device="cpu", verbose=0, seed=SEED) 
    elif args.policy == "PPO":
        model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": args.net_arch}, learning_rate=args.learningRate, batch_size=args.batchSize, vf_coef=args.vf_coef, ent_coef=args.ent_coef, gamma=args.gamma, device="cpu", verbose=0, seed=SEED)

    # Print number of parameters
    numParams = 0
    for key, value in model.get_parameters()['policy'].items():
        numParams += value.numel()
    print(f"\033[95mNumber of parameters in the model: {numParams:,}\033[0m")

    try:
        loadFile = f"{log_dir}/{args.policy}_{args.env}"
        model.set_parameters(loadFile)
        print(f"Pre-trained model {loadFile} loaded successfully.")
    except:
        print("No pre-trained model found, starting from scratch.")
        return

    vec_env = model.get_env()

    rolloutFigs = []
    imgs = []
    dists = []
    rewards = []
    vec_env.seed(SEED)
    for iEnv in range(1):
        obs = vec_env.reset()

        actions = []
        ctrls = []
        motorPos = []
        motorVel = []
        coms = []
        waypoints = []

        totalSteps = int(2*np.pi*period)
        waypointResets = 1 #totalSteps // numWaypoints
        for t in range(totalSteps-waypointResets-1+0):
            if (t % waypointResets == 0) and t < totalSteps-waypointResets-1:
                ### Track target in front of fish
                # com = vec_env.envs[0].unwrapped.data.site("COM_0").xpos.copy()
                # headAngle = obs[0,5]
                # rot = np.array([
                #     [np.cos(headAngle), -np.sin(headAngle)],
                #     [np.sin(headAngle), np.cos(headAngle)]
                # ])
                # target = com[:2] + rot @ np.array([
                #     np.random.uniform(low=-2.0, high=-0.5),
                #     np.random.uniform(low=-1.0, high=1.0)
                # ])

                ### Track circle
                x = (t+waypointResets)/period
                radius = 1.0
                target = radius * np.array([
                    -np.sin(x),
                    1-np.cos(x)
                ])

                ### Track sinusoidal
                # x = (t+waypointResets)/period
                # target = np.array([
                #     -0.2 - 15*(t+waypointResets)/totalSteps,
                #     1.0 * np.sin(2.0*np.pi*x*0.5)
                # ])

                ### Track square
                # idx = int((t+waypointResets)//(totalSteps/4))
                # points = np.array([
                #     [-2, 0.],
                #     [-2, 2.],
                #     [0, 2.],
                #     [0, 0.],
                # ])
                # target = points[idx%4]

                waypoints.append(target.copy())
                vec_env.envs[0].unwrapped.target = target
                vec_env.envs[0].unwrapped.data.site("target").xpos[:2] = target
                vec_env.envs[0].unwrapped.model.site("target").pos[:2] = target
                # print(f"New target: {target}")

            action, _states = model.predict(obs)
            obs, reward, dones, info = vec_env.step(action)

            # compute distance for circle
            com = vec_env.envs[0].unwrapped.data.site("COM_0").xpos.copy()
            # dist = obs[0][vec_env.envs[0].unwrapped.observationIndices["distanceTarget"]]
            dist = abs(np.linalg.norm(com[:2] - np.array([0.0, radius])) - radius)

            actions.append(action[0])
            ctrls.append(obs[0][env.unwrapped.observationIndices["motorJointVel"]])
            # dists.append(obs[0][env.unwrapped.observationIndices["distanceTarget"]])
            dists.append(dist)
            rewards.append(reward[0])
            motorPos.append(obs[0,0])
            motorVel.append(obs[0,1])
            coms.append(vec_env.envs[0].unwrapped.data.site("COM_0").xpos.copy())

            if visualize:
                print(f"Timestep {t:04d}: \tAction: {action} and \tReward: {reward}")
                vec_env.render("human")
                img = vec_env.render("rgb_array")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                imgs.append(img)

            
            if dones.any():
                print(f"Episode finished after {t+1} timesteps")
                break

        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        axs[0].plot(np.array(actions), label="Actions")
        axs[0].set_xlabel("Timestep")
        axs[0].set_ylabel("Action Value")
        axs[0].grid()

        axs[1].plot(np.array(ctrls), label="Controls", color='orange')
        axs[1].set_xlabel("Timestep")
        axs[1].set_ylabel("Control Value")
        axs[1].grid()

        axs[2].plot(np.array(dists), label="Distance", color='green')
        axs[2].set_xlabel("Timestep")
        axs[2].set_ylabel("Distance to target (m)")
        axs[2].grid()

        fig.savefig("Outputs/test.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(9, 3))
        fig.subplots_adjust(wspace=0.35)
        axs[0].plot(np.array(motorPos), label="Motor Position")
        axs[0].set_xlabel("Timestep")
        axs[0].set_ylabel("Motor Position (rad)")
        axs[0].grid()
        axs[1].plot(np.array(motorVel), label="Motor Velocity", color='orange')
        axs[1].set_xlabel("Timestep")
        axs[1].set_ylabel("Motor Velocity (rad/s)")
        axs[1].grid()
        fig.savefig("Outputs/motor.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


        fig, ax = plt.subplots(figsize=(40*mm, 40*mm))
        coms = np.array(coms)
        waypoints = np.array(waypoints)
        ax.plot(coms[:, 0], coms[:, 1], linewidth=1, label="COM", zorder=1)
        ax.scatter(coms[0, 0], coms[0, 1], s=5, label="Start", color='tab:green', zorder=10)
        ax.scatter(coms[-1, 0], coms[-1, 1], s=5, label="End", color='tab:red', zorder=10)
        # ax.plot(waypoints[:, 0], waypoints[:, 1], label="Trajectory", color=(0.1, 0.8, 0.1, 1.0), linestyle='--', linewidth=0.5, zorder=8)
        ax.plot(waypoints[:, 0], waypoints[:, 1], label="Trajectory", color="black", linestyle='--', linewidth=0.5, zorder=2)
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.grid()
        # ax.legend()
        fig.savefig("Outputs/com_trajectory.png", dpi=300, bbox_inches="tight")
        fig.savefig("Outputs/com_trajectory.pdf", bbox_inches="tight")
        plt.close(fig)


    if visualize:
        ### Save Video
        video = cv2.VideoWriter("_tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 50, (img.shape[0], img.shape[1]))
        for img in imgs:
            video.write(img)
        video.release()

        os.system("ffmpeg -i _tmp.mp4 -y -hide_banner -loglevel error -c:v libx264 -crf 28 -c:a aac -strict 2 Outputs/movie.mp4")
        os.remove("_tmp.mp4")
        print("\033[94mStored video in Outputs/movie.mp4.\033[0m")

    log = {
        "actions": np.array(actions),
        "ctrls": np.array(ctrls),
        "dists": np.array(dists),
        "rewards": np.array(rewards),
        "figs": rolloutFigs,
    }

    return log



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a fish model with RL.")
    parser.add_argument("--env", type=str, default="fish_target_relative-v0", choices=["fish_target_simple-v0", "fish_target-v0", "fish_target_3daction-v0", "fish_target_2daction-v0", "fish_target_freqperiod-v0", "fish_target_torque-v0", "fish_target_history-v0", "fish_target_relative-v0"], help="Environment to use (default: fish_target_torque-v0).")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model instead of training.")
    parser.add_argument("--load", action="store_true", help="Load a pre-trained model instead of training from scratch.")
    parser.add_argument("--nSteps", type=int, default=5_000_000, help="Number of environment steps.")
    parser.add_argument("--policy", type=str, default="SAC", choices=["PPO", "SAC"], help="RL policy to use (default: SAC).")

    parser.add_argument("--net_arch", type=list, default=[128, 128], help="Network architecture for the policy.")
    parser.add_argument("--learningRate", type=float, default=2e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--batchSize", type=int, default=256, help="Batch size for training.")

    parser.add_argument("--maxEpisodeSteps", type=int, default=4000, help="Maximum number of steps per episode.")
    parser.add_argument("--actionMultiplier", type=float, default=400, help="Multiplier for the action space.")
    parser.add_argument("--rewardFailure", type=float, default=-1e3, help="Reward for failure (default: -1e3).")
    parser.add_argument("--motorVelRange", type=float, default=30, help="Random sample range for motor velocities (default: 30).")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")

    ### PPO specific arguments
    parser.add_argument("--vf_coef", type=float, default=1.0, help="Value function coefficient (default: 1.0).")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient (default: 0.01).")
    
    ### SAC specific arguments
    parser.add_argument("--tau", type=float, default=0.1, help="Soft update coefficient.")
    parser.add_argument("--learningStarts", type=int, default=50_000, help="Number of steps to explore before starting training.")
    parser.add_argument("--train_freq", type=int, default=200, help="Training frequency.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the environment and save a video.")

    args = parser.parse_args()


    # args.net_arch = [659, 659, 659, 659]
    # args.actionMultiplier = 423.3954337821436

    args.net_arch = [1123, 1123, 1123, 1123]
    args.actionMultiplier = 451.6423199890723


    log = main(args, visualize=args.visualize, period=97, numWaypoints=50)

    print(f"Mean Reward: {np.mean(log['rewards']):.4f} \t- Std: {np.std(log['rewards']):.4f}")
    print(f"Mean Distance: {np.mean(log['dists']):.4f} \t- Std: {np.std(log['dists']):.4f} \t- Min: {np.min(log['dists']):.4f} \t- Max: {np.max(log['dists']):.4f}")
