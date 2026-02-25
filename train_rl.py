
import os
import time

import cv2
import numpy as np
import gymnasium as gym

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

import wandb
# Set wandb environment variable
os.environ["WANDB_MODE"] = "offline"

import Environments.env_fish_target_relative as env_fish_target_relative

SEED = 42

plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})# Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"ps.fonttype": 42}) # Prevents type 3 fonts (deprecated in paper submissions)
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})#, "font.serif": ['Computer Modern']})

mm = 1 / 25.4


class PlotCallback (BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, nEnv: int = 1, rewardAverageWindow: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.nEnv = nEnv
        self.rewardAverageWindow = rewardAverageWindow

        self.startTime = time.time()
        self.prevTime = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % (self.check_freq//self.nEnv) == 0:
            print(f"Step {self.n_calls:06d} \t- Total Time elapsed: {time.time() - self.startTime:.2f}s \t- Time since last check: {time.time() - self.prevTime:.2f}s")
            self.prevTime = time.time()

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Plot rewards
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                ax.scatter(x, y, s=1, alpha=0.9)
                # Running average plot
                if len(y) > self.rewardAverageWindow:
                    running_avg = np.convolve(y, np.ones(self.rewardAverageWindow)/self.rewardAverageWindow, mode='valid')
                    ax.plot(x[self.rewardAverageWindow-1:], running_avg, color='red', label=f'Running average ({self.rewardAverageWindow} steps)')
                    # ax.legend()
                ax.set_xlabel("Environment Steps (-)")
                ax.set_ylabel("Reward")
                # ax.set_yscale('symlog')
                ax.grid()
                ax.ticklabel_format(style="sci", axis='both', scilimits=(0,0))
                fig.savefig(os.path.join(self.log_dir, "training_rewards.png"), dpi=300, bbox_inches="tight")
                plt.close(fig)

        return True


class CheckCallback (BaseCallback):
    def __init__(self, save_freq: int, log_dir: str, filename: str, nEnv: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.nEnv = nEnv
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.filename = filename

    def _on_step(self) -> bool:
        if self.n_calls % (self.save_freq//self.nEnv) == 0:
            print(f"Step {self.n_calls:06d}: Saving model to {self.filename}")
            self.model.save(f"{self.log_dir}/{self.filename}")

        return True




def main(args):
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



    # Parallel environments
    env = gym.make(f"gymnasium_env/{args.env}", reset_noise_scale=0.0, frame_skip=20, render_mode="rgb_array", motorVelRange=args.motorVelRange, actionMultiplier=args.actionMultiplier, rewardFailure=args.rewardFailure, seed=SEED, width=800, height=800)
    check_env(env)
    if not args.eval:
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
    if os.environ["WANDB_MODE"] == "online":
        wandb.log({"Number of Trainable Parameters": numParams})


    if args.load or args.eval:
        try:
            loadFile = f"{log_dir}/{args.policy}_{args.env}"
            model.set_parameters(loadFile)
            print(f"Pre-trained model {loadFile} loaded successfully.")
        except:
            print("No pre-trained model found, starting from scratch.")

    # Train the model
    if not args.eval:
        # Create callbacks for storing results
        callback = PlotCallback(check_freq=10_000, log_dir=log_dir, rewardAverageWindow=100)

        checkpoint_callback = CheckCallback(
            save_freq=100_000,
            log_dir=log_dir,
            filename=f"{args.policy}_{args.env}",
        )
        
        try:
            model.learn(total_timesteps=args.nSteps, callback=[callback, checkpoint_callback])
        finally:
            model.save(f"{log_dir}/{args.policy}_{args.env}")
            print(f"Model saved to {log_dir}/ppo_{args.env}")

            x, y = ts2xy(load_results(log_dir), "timesteps")
            # Plot rewards
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.scatter(x, y, s=1, alpha=0.9)
            # Running average plot
            if len(y) > callback.rewardAverageWindow:
                running_avg = np.convolve(y, np.ones(callback.rewardAverageWindow)/callback.rewardAverageWindow, mode='valid')
                ax.plot(x[callback.rewardAverageWindow-1:], running_avg, color='red', label=f'Running average ({callback.rewardAverageWindow} steps)')
                # ax.legend()
            ax.set_xlabel("Environment Steps (-)")
            ax.set_ylabel("Reward")
            ax.set_yscale('symlog')
            ax.grid()
            ax.ticklabel_format(style="sci", axis='x', scilimits=(0,0))
            fig.savefig(os.path.join(log_dir, "training_rewards.png"), dpi=300, bbox_inches="tight")
            if os.environ["WANDB_MODE"] == "online":
                wandb.log({"Training Rewards": wandb.Image(fig, caption="Training Rewards")})
            plt.close(fig)

    vec_env = model.get_env()


    print(f"Training completed")
    print(f"Action space: {vec_env.action_space}")

    rolloutFigs = []
    imgs = []
    dists = []
    rewards = []
    coms = []
    targets = []
    successes = 0
    vec_env.seed(SEED)
    for iEnv in range(20):
        obs = vec_env.reset()
        targets.append(vec_env.envs[0].unwrapped.data.site("target").xpos.copy())


        actions = []
        ctrls = []
        motorPos = []
        motorVel = []
        currComs = []
        for t in range(args.maxEpisodeSteps-1):
            action, _states = model.predict(obs)
            obs, reward, dones, info = vec_env.step(action)

            actions.append(action[0])
            ctrls.append(obs[0][env.unwrapped.observationIndices["motorJointVel"]])
            dists.append(obs[0][env.unwrapped.observationIndices["distanceTarget"]])
            # dists.append(np.linalg.norm(obs[0][env.unwrapped.observationIndices["vecTarget"]]))
            rewards.append(reward[0])
            motorPos.append(obs[0,0])
            motorVel.append(obs[0,1])

            if dones.any():
                print(f"Episode finished after {t+1} timesteps")
                successes += 1
                break
            currComs.append(vec_env.envs[0].unwrapped.data.site("COM_0").xpos.copy())

            if args.eval:
                print(f"Timestep {t:04d}: \tAction: {action} and \tReward: {reward} and \tDistance: {obs[0][env.unwrapped.observationIndices['distanceTarget']][0]:.4f}")
                # vec_env.render("human")
                # img = vec_env.render("rgb_array")
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # imgs.append(img)
            
        coms.append(np.array(currComs))

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

        if os.environ["WANDB_MODE"] == "online":
            rolloutFigs.append(wandb.Image(fig, caption=f"Environment {iEnv}"))

        if args.eval:
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
    for i, (cs, ts) in enumerate(zip(coms, targets)):
        ax.plot(cs[:, 0], cs[:, 1], linewidth=1, color="tab:blue", label="COM Trajectory")
        ax.scatter(ts[0], ts[1], label="Target", color='tab:red', s=5, zorder=5)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_xlim(-4, 0)
    ax.set_ylim(-2, 2)
    ax.grid()
    # ax.legend()
    fig.savefig("Outputs/com_trajectory.png", dpi=300, bbox_inches="tight")
    fig.savefig("Outputs/com_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\033[95mSuccessful targets reached: {successes}/{len(targets)}\033[0m")


    if args.eval:
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
    parser.add_argument("--env", type=str, default="fish_target_relative-v0", choices=["fish_target_relative-v0"], help="Environment to use (default: fish_target_relative-v0).")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model instead of training.")
    parser.add_argument("--load", action="store_true", help="Load a pre-trained model instead of training from scratch.")
    parser.add_argument("--nSteps", type=int, default=5_000_000, help="Number of environment steps.")
    parser.add_argument("--policy", type=str, default="SAC", choices=["PPO", "SAC"], help="RL policy to use (default: SAC).")

    parser.add_argument("--net_arch", type=list, default=[128, 128], help="Network architecture for the policy.")
    parser.add_argument("--learningRate", type=float, default=2e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--batchSize", type=int, default=256, help="Batch size for training.")

    parser.add_argument("--maxEpisodeSteps", type=int, default=400, help="Maximum number of steps per episode.")
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

    args = parser.parse_args()

    # args.net_arch = [659, 659, 659, 659]
    # args.actionMultiplier = 423.3954337821436
    # args.batchSize = 664
    # args.gamma = 0.9936627804628952
    # args.learningRate = 0.002691036439521784
    # args.learningStarts = 82221
    # args.maxEpisodeSteps = 1000
    # args.train_freq = 31
    # args.nSteps = 3_000_000

    args.net_arch = [1123, 1123, 1123, 1123]
    args.actionMultiplier = 451.6423199890723
    args.maxEpisodeSteps = 1000


    log = main(args)

    print(f"Mean Reward: {np.mean(log['rewards']):.4f} \t- Std: {np.std(log['rewards']):.4f}")
    print(f"Mean Distance: {np.mean(log['dists']):.4f} \t- Std: {np.std(log['dists']):.4f}")