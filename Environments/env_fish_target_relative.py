
__credits__ = ["Mike Y. Michelis"]

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import wandb

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from Geometry.auto_tendonFish import generate_xml, SYSTEMPARAMETERS



class FishEnv (MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "./Geometry/tendonFish.xml",
        frame_skip: int = 20,
        default_camera_config: dict[str, float | int] = {},
        reset_noise_scale: float = 0.0,
        motorVelRange: float = 1.0,
        actionMultiplier: float = 1.0,
        rewardFailure: float = -1000.0,
        noterminate=False,
        seed: int = 42,
        **kwargs,
    ):
        ### Generate and initialize XML environment
        systemParameters = SYSTEMPARAMETERS.copy()
        generate_xml(systemParameters, xml_file)

        self._reset_noise_scale = reset_noise_scale
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            camera_name="tracking_0",
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }


        self.observation_structure = {
            "cos(motorJointPos)": 1,
            "sin(motorJointPos)": 1,
            "motorJointVel": 1,
            "headJointVel": 3,
            "tailJointPos": 5,
            "tailJointVel": 5,
            "distanceTarget": 1,
            "vecTarget": 2,
            "velVecTarget": 2,
            "prevAction": 1,
        }
        observationKeys = list(self.observation_structure.keys())
        observationValues = list(self.observation_structure.values())
        self.observationIndices = {}
        for i in range(len(observationKeys)):
            self.observationIndices[observationKeys[i]] = range(sum(observationValues[:i]), sum(observationValues[:i+1]))
        obs_shape = sum(self.observation_structure.values())
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.prevCom = None
        self.prevComVel = None
        self.prevVecTarget = None
        self.prevAction = None
        self.target = np.array([0.0, 0.0])  # Target position for the fish to swim towards

        self.nStep = 0
        self.currentEpisodeReward = 0.0 # Total reward for the current episode
        self.episodeRewards = [] # List of rewards for each episode
        self.noterminate = noterminate

        self.motorVelRange = motorVelRange
        self.actionMultiplier = actionMultiplier
        self.rewardFailure = rewardFailure

        self.plotHistogram = False  # Plot histograms of observations and actions
        self.obs = []
        self.act = []

        self.startTime = time.time()
        self.seed = seed
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.reset(seed=seed)


    def step (self, action):
        self.nStep += 1

        # Action as acceleration
        action_velocity = self.data.qvel[6:7].copy() + self.actionMultiplier * action * self.dt
        self.do_simulation(action_velocity, self.frame_skip)

        observation = self._get_obs()
        if self.plotHistogram:
            self.obs.append(observation.copy())
            self.act.append(action.copy())

        # Termination condition
        com = self.data.site("COM_0").xpos
        terminated = (not self.noterminate) and bool(float(observation[self.observationIndices["distanceTarget"]]) < 0.05)

        rewardDistance = -float(observation[self.observationIndices["distanceTarget"]])
        rewardAction = -np.linalg.norm(action)
        rewardSuccess = 300.0 if terminated else 0.0

        reward = float(rewardDistance + 1.0 * rewardAction + rewardSuccess)

        self.currentEpisodeReward += reward
        self.prevAction = action.copy()

        ### Logging
        if self.nStep % 100 == 0:
            print(f"\
Step {self.nStep:,} in {time.time()-self.startTime:.2f}s: \
 |  COM: [{self.data.site('COM_0').xpos[0]:+.2f}, {self.data.site('COM_0').xpos[1]:+.2f}, {self.data.site('COM_0').xpos[2]:+.2f}] \
 |  Target: [{self.target[0]:+.2f}, {self.target[1]:+.2f}] \
 |  Distance: {observation[self.observationIndices['distanceTarget']][0]:.4f} \
 |  Ctrl: [{self.data.ctrl[0]:+.2f}] \
 |  Action: [{action[0]:+.4f}] \
 |  Reward: {reward:.4f}\
            ")

            if self.plotHistogram:
                nCols = 7
                fig, axs = plt.subplots(5, nCols, figsize=(nCols*2.5, 12))
                fig.subplots_adjust(wspace=0.6, hspace=0.6)
                for i in range(observation.shape[0]):
                    axs[i//nCols, i%nCols].hist(np.array(self.obs)[:, i], bins=50, density=True)
                    axs[i//nCols, i%nCols].set_xlabel(f"Obs {i}")
                    axs[i//nCols, i%nCols].grid()
                for i in range(1, action.shape[0]+1):
                    axs[-1, -i].hist(np.array(self.act)[:, i-1], bins=50, density=True)
                    axs[-1, -i].set_xlabel(f"Act {i-1}")
                    axs[-1, -i].grid()
                fig.savefig("Outputs/testo.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

        info = {}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info


    def reset_model (self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # motor vel
        # qvel[6] += self.np_random.uniform(low=-self.motorVelRange, high=self.motorVelRange)
        # self.data.ctrl = qvel[6]

        # Target position
        self.target = np.array([
            self.np_random.uniform(low=-4.0, high=0.0),
            self.np_random.uniform(low=-2.0, high=2.0)
        ])
        self.data.site("target").xpos[:2] = self.target
        self.model.site("target").pos[:2] = self.target

        self.set_state(qpos, qvel)

        self.prevCom = self.data.site("COM_0").xpos.copy()
        self.prevComVel = np.zeros(3)
        self.prevVecTarget = self.target - self.prevCom[:2]
        self.prevAction = np.zeros(self.action_space.shape)

        print(f"Step {self.nStep:,}: Episode Reward: {self.currentEpisodeReward:.4f}")
        self.episodeRewards.append(self.currentEpisodeReward)
        self.currentEpisodeReward = 0.0

        if os.environ["WANDB_MODE"] == "online" and self.nStep > 0:
            wandb.log({"Episode Reward": self.episodeRewards[-1], "Environment Step": self.nStep, "Episode Number": len(self.episodeRewards)})

        return self._get_obs()


    def _get_obs (self):
        headJointPos = self.data.qpos[:6].copy()
        headJointVel = self.data.qvel[:6].copy()
        motorJointPos = self.data.qpos[6:7].copy()
        motorJointVel = self.data.qvel[6:7].copy() #/ self.motorVelRange
        tailJointPos = self.data.qpos[7:12].copy()
        tailJointVel = self.data.qvel[7:12].copy() #/ self.motorVelRange

        com = self.data.site("COM_0").xpos.copy()
        comVel = (com - self.prevCom) / self.dt
        comAcc = (comVel - self.prevComVel) / self.dt
        self.prevCom = com
        self.prevComVel = comVel
        
        vecTarget = self.target - com[:2]
        velVecTarget = (vecTarget - self.prevVecTarget) / self.dt
        # rotate vecTarget and velVecTarget into fish frame
        angle = -headJointPos[5]
        rotationMatrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        vecTarget = rotationMatrix @ vecTarget
        velVecTarget = rotationMatrix @ velVecTarget
        distanceTarget = np.linalg.norm(vecTarget).reshape(1,)
        # directionTarget = vecTarget / (distanceTarget + 1e-6)
        self.prevVecTarget = vecTarget

        return np.concatenate([
            np.cos(motorJointPos), np.sin(motorJointPos), motorJointVel,
            headJointVel[3:],
            tailJointPos, tailJointVel,
            distanceTarget, vecTarget, velVecTarget,
            self.prevAction
        ])
    

