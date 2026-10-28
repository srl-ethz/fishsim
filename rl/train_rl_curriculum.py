"""SAC curriculum training — stage definitions live in each env class.

Each FishEnv subclass carries a CURRICULUM_STAGES class attribute that
defines its canonical progression.  Phase-2 clean variants also expose
CURRICULUM_STAGES_CLEAN.  This runner reads those attributes at startup,
so adding a new curriculum is just adding stages to the env file.

Stage advancement: the rolling success rate over SUCCESS_WINDOW episodes
from the current stage exceeds SUCCESS_THRESHOLD.  An episode counts as a
success when it terminates naturally (target reached, distance < 0.05 m)
rather than timing out via the episode step limit.

Replay buffer is NOT flushed at transitions — LR is scaled down instead
so updates stay small while the buffer turns over naturally.
"""

import os
import sys
import time

import numpy as np
import gymnasium as gym

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.env_config_registry import register_environments
from rl.mirrored_replay_buffer import MirroredReplayBuffer

SEED = 42

# ── Curriculum hyper-parameters ──────────────────────────────────────────────
# Stage advancement: advance when the rolling episode success rate over the
# last SUCCESS_WINDOW episodes in the current stage exceeds SUCCESS_THRESHOLD.
SUCCESS_THRESHOLD = 0.9     # fraction of episodes that must reach the target
SUCCESS_WINDOW    = 100     # rolling window size (episodes); must complete this many in the
                            # current stage before advancement is allowed

# Learning rate scaling applied at each stage transition.
LR_SCALE = 0.3

# How often (in steps) to check the success rate and log to WandB.
CHECK_FREQ = 2_000

# Stage definitions live in each env class as CURRICULUM_STAGES (and
# CURRICULUM_STAGES_CLEAN for the phase-2 clean variant).  The runner reads
# them from the instantiated env after gym.make() so no imports are needed here.


# ── Callbacks ────────────────────────────────────────────────────────────────

class StageStopCallback(BaseCallback):
    """Stops model.learn() early when rolling episode success rate hits the threshold.

    An episode is a success when it terminates naturally (target reached) rather
    than being cut off by the time limit.  Returns False from _on_step to signal
    SB3 to stop the current learn() call; stage advancement is handled by main().
    """

    def __init__(self, threshold, window, stage_idx,
                 min_stage_steps=0, check_freq=CHECK_FREQ, verbose=1):
        super().__init__(verbose)
        self.threshold       = threshold
        self.window          = window
        self.stage_idx       = stage_idx
        self.min_stage_steps = min_stage_steps
        self.check_freq      = check_freq
        self._advanced       = False
        self._success_buf    = []   # 1 per episode end in this stage: 1=success, 0=timeout

    def _on_step(self) -> bool:
        # Record outcome for every episode that ends this step.
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self._success_buf.append(1 if info.get("is_success", False) else 0)

        if self.n_calls % self.check_freq != 0:
            return True

        if os.environ.get("WANDB_MODE", "offline") == "online":
            wandb.log({"curriculum/stage": self.stage_idx}, step=self.num_timesteps)

        if not self._success_buf:
            return True

        # Always log success rate from current-stage episodes only.
        # During the first <window episodes this is a partial rate over however
        # many episodes have completed — useful for early monitoring.
        success_rate = float(np.mean(self._success_buf[-self.window:]))

        if os.environ.get("WANDB_MODE", "offline") == "online":
            wandb.log({"curriculum/stage_success_rate": success_rate}, step=self.num_timesteps)

        # Advancement requires a full window of current-stage episodes AND the
        # minimum stage-step budget (prevents premature promotion on a lucky streak
        # before the buffer has flushed to the new physics distribution).
        if (len(self._success_buf) >= self.window
                and success_rate >= self.threshold
                and self.n_calls >= self.min_stage_steps):
            print(f"\nSuccess rate {success_rate:.1%} >= {self.threshold:.1%} "
                  f"at step {self.num_timesteps:,} — advancing to next stage.")
            self._advanced = True
            # Save at this exact moment: model.save() after model.learn() returns
            # is ~500 gradient steps later (check_freq / train_freq), which can
            # degrade the policy via Q-overestimated updates.
            if self.model is not None and hasattr(self.model, "logger"):
                _at_thresh = os.path.join(
                    self.model.logger.dir,
                    f"stage{self.stage_idx}_at_threshold"
                )
                self.model.save(_at_thresh)
                print(f"  Checkpoint at threshold → {_at_thresh}.zip")
            return False

        return True


class PlotCallback(BaseCallback):
    """Periodic reward plot and W&B scalar logging."""

    def __init__(self, check_freq, log_dir, reward_window=200, verbose=1):
        super().__init__(verbose)
        self.check_freq    = check_freq
        self.log_dir       = log_dir
        self.reward_window = reward_window
        self.start_time    = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        x, y = ts2xy(load_results(self.log_dir), "timesteps")
        if len(x) == 0:
            return True

        recent = y[-self.reward_window:] if len(y) >= self.reward_window else y

        if os.environ.get("WANDB_MODE", "offline") == "online":
            log_dict = {
                "train/episode_reward":    float(y[-1]),
                "train/reward_mean":       float(np.mean(recent)),
                "train/reward_std":        float(np.std(recent)),
                "train/environment_steps": int(x[-1]),
            }
            if self.model is not None:
                if hasattr(self.model, "log_ent_coef"):
                    import torch
                    log_dict["train/ent_coef"] = float(torch.exp(self.model.log_ent_coef).item())
                elif hasattr(self.model, "ent_coef_tensor"):
                    log_dict["train/ent_coef"] = float(self.model.ent_coef_tensor.item())
                log_dict["train/lr_actor"]  = float(self.model.actor.optimizer.param_groups[0]["lr"])
                log_dict["train/lr_critic"] = float(self.model.critic.optimizer.param_groups[0]["lr"])
            wandb.log(log_dict, step=self.num_timesteps)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, s=1, alpha=0.9)
        if len(y) > self.reward_window:
            running = np.convolve(
                y, np.ones(self.reward_window) / self.reward_window, mode='valid'
            )
            ax.plot(x[self.reward_window - 1:], running, color='red',
                    label=f'Rolling mean ({self.reward_window} eps)')
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Reward")
        ax.grid()
        fig.savefig(os.path.join(self.log_dir, "training_rewards.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        elapsed = time.time() - self.start_time
        print(f"Step {self.num_timesteps:,} | "
              f"reward_mean={np.mean(recent):.1f} | {elapsed:.0f}s elapsed")
        return True


class EvalSuccessCallback(BaseCallback):
    """Adaptive stochastic-only evaluation.

    Checks every eval_freq steps. Two triggers decide whether to run:
    1. Last eval stochastic >= high_threshold: run now with n_eval_fast episodes.
    2. eval_freq_slow steps have elapsed since last eval (fallback): run with n_eval_slow.

    Before abs_stage reaches early_stage_threshold, only the slow fallback fires
    and it uses eval_freq_early instead of eval_freq_slow — avoiding pointless
    frequent evals during easy early stages where the policy either bootstraps
    quickly or doesn't.  Above the threshold, normal fast/slow behaviour resumes.

    Initialized so the fallback fires at the very first check (step eval_freq),
    giving an early baseline even before any threshold can be met.

    Rationale: at low success rates, eval episodes almost always hit the max-step
    timeout (2000 steps), so 200 eval episodes per 50k training steps wastes ~86%
    of wall-clock.  This scheme runs cheap fallback evals every 100k when struggling,
    and switches to denser, larger evals only once the policy is performing well.
    """

    def __init__(self, train_env, eval_env,
                 eval_freq=50_000, eval_freq_slow=100_000,
                 n_eval_fast=100, n_eval_slow=50,
                 high_threshold=0.80,
                 stage_ref=None, early_stage_threshold=4, eval_freq_early=200_000,
                 log_dir=None, verbose=1):
        super().__init__(verbose)
        self._train_raw            = train_env.unwrapped
        self._eval_env             = eval_env
        self._eval_raw             = eval_env.unwrapped
        self.eval_freq             = eval_freq
        self.eval_freq_slow        = eval_freq_slow
        self.n_eval_fast           = n_eval_fast
        self.n_eval_slow           = n_eval_slow
        self.high_threshold        = high_threshold
        self._stage_ref            = stage_ref if stage_ref is not None else [0]
        self.early_stage_threshold = early_stage_threshold
        self.eval_freq_early       = eval_freq_early
        self.log_dir               = log_dir
        self.best_stoch            = -np.inf
        self.best_stoch_per_stage  = {}        # {stage_idx: best success rate seen}
        self._last_eval_stoch      = -np.inf  # forces fallback on first late-stage check
        self._last_eval_step       = 0        # early: first eval at eval_freq_early (200k)
                                              # late:  first slow eval at eval_freq_slow (100k)
        self._last_eval_stage      = self._stage_ref[0]  # fires on stage change in early stages

    def _sync_eval_env(self):
        src, dst = self._train_raw, self._eval_raw
        sm_s, sm_d = src._sensor_model, dst._sensor_model
        for attr in ("enable_imu", "enable_encoder", "enable_vision",
                     "enable_vision_noise", "enable_latency", "enable_fps_dr",
                     "latency_min", "latency_max", "frame_drop_prob"):
            if hasattr(sm_s, attr):
                setattr(sm_d, attr, getattr(sm_s, attr))
        for attr in ("r_min", "r_max", "theta_max",
                     "_motor_dr_active", "_extra_frames_active"):
            if hasattr(src, attr):
                setattr(dst, attr, getattr(src, attr))
        pm_s = getattr(src, "_physics_model", None)
        pm_d = getattr(dst, "_physics_model", None)
        if pm_s is not None and pm_d is not None:
            for attr in ("enable_init_vel", "enable_inertial", "enable_mechanics",
                         "enable_fluid", "fluid_std", "enable_motor",
                         "motor_kp_range", "motor_kv_range", "motor_deadtime_range",
                         "motor_armature_range", "motor_forcerange_range",
                         "motor_zero_offset_std"):
                if hasattr(pm_s, attr):
                    setattr(pm_d, attr, getattr(pm_s, attr))

    def _run_episodes(self, n_episodes: int) -> float:
        n_success = 0
        for _ in range(n_episodes):
            obs, _ = self._eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated
                if info.get("is_success", False):
                    n_success += 1
        return n_success / n_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        steps_since_last = self.num_timesteps - self._last_eval_step
        current_stage    = self._stage_ref[0]

        if current_stage < self.early_stage_threshold:
            # Early stages: fire on stage change OR every eval_freq_early steps (200k).
            # Stage-change trigger catches fast-bootstrapping runs that promote before
            # the time-based fallback would fire.
            stage_changed = current_stage != self._last_eval_stage
            if not stage_changed and steps_since_last < self.eval_freq_early:
                return True
            n_ep = self.n_eval_slow
            mode = "slow-early"
        else:
            above_threshold = self._last_eval_stoch >= self.high_threshold
            if above_threshold:
                n_ep = self.n_eval_fast
                mode = "fast"
            elif steps_since_last >= self.eval_freq_slow:
                n_ep = self.n_eval_slow
                mode = "slow"
            else:
                return True  # below threshold and not yet at slow checkpoint

        self._sync_eval_env()
        stoch = self._run_episodes(n_ep)
        self._last_eval_stoch = stoch
        self._last_eval_step  = self.num_timesteps
        self._last_eval_stage = current_stage
        if self.verbose:
            print(f"\n[Eval/{mode}] step={self.num_timesteps:,} | "
                  f"stochastic={stoch:.1%} (n={n_ep})")

        if os.environ.get("WANDB_MODE", "offline") == "online":
            wandb.log({
                "eval/success_rate_stochastic": stoch,
                "eval/n_episodes":              n_ep,
            }, step=self.num_timesteps)

        if stoch > self.best_stoch and self.log_dir:
            self.best_stoch = stoch
            self.model.save(os.path.join(self.log_dir, "best_model"))
            print(f"  → best_model saved (stochastic={stoch:.1%})")

        if self.log_dir and stoch > self.best_stoch_per_stage.get(current_stage, -np.inf):
            self.best_stoch_per_stage[current_stage] = stoch
            path = os.path.join(self.log_dir, f"best_model_stage{current_stage}")
            self.model.save(path)
            print(f"  → best_model_stage{current_stage} saved (stochastic={stoch:.1%})")
            if os.environ.get("WANDB_MODE", "offline") == "online":
                wandb.log({
                    f"eval/best_stage{current_stage}": stoch,
                }, step=self.num_timesteps)

        return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    log_dir = getattr(args, 'logDir', 'outputs') or 'outputs'
    os.makedirs(log_dir, exist_ok=True)

    register_environments(max_episode_steps=args.maxEpisodeSteps)

    env_name    = getattr(args, 'env', 'sofi_phase1_1_noiseAndDelay')
    base_env_id = f"gymnasium_env/{env_name}"
    seed        = int(getattr(args, "seed", SEED))
    lr_scale    = float(getattr(args, "lr_scale", LR_SCALE))

    heading_reward_weight  = float(getattr(args, "heading_reward_weight",  0.0))
    progress_reward_weight = float(getattr(args, "progress_reward_weight", 0.0))
    distance_reward_weight = float(getattr(args, "distance_reward_weight", 1.0))
    time_penalty_per_step  = float(getattr(args, "time_penalty_per_step",  0.0))
    alpha_jerk             = float(getattr(args, "alpha_jerk",             0.0))
    alpha_heading          = float(getattr(args, "alpha_heading",          0.0))
    alpha_dist             = float(getattr(args, "alpha_dist",             0.0))
    alpha_fwd              = float(getattr(args, "alpha_fwd",              0.0))
    alpha_backward         = float(getattr(args, "alpha_backward",         0.0))
    backward_vel_limit     = float(getattr(args, "backward_vel_limit",     0.0))
    success_radius         = float(getattr(args, "success_radius",         0.1))
    reward_success         = float(getattr(args, "rewardSuccess",          1000.0))
    motor_ctrlrange        = float(getattr(args, "motor_ctrlrange",        1.0))
    camera_fps             = float(getattr(args, "camera_fps",             30.0))
    use_mirror             = bool(getattr(args, "mirror_replay_buffer",    False))
    lpf_cutoff_hz          = float(getattr(args, "lpf_cutoff_hz",          0.0))
    motor_deadtime_steps   = int(getattr(args,   "motor_deadtime_steps",    0))
    use_distance_reward    = bool(getattr(args, "use_distance_reward",     False))
    reward_kwargs = {}
    if heading_reward_weight  > 0.0:  reward_kwargs["heading_reward_weight"]  = heading_reward_weight
    if progress_reward_weight > 0.0:  reward_kwargs["progress_reward_weight"] = progress_reward_weight
    if distance_reward_weight != 1.0: reward_kwargs["distance_reward_weight"] = distance_reward_weight
    if time_penalty_per_step  > 0.0:  reward_kwargs["time_penalty_per_step"]  = time_penalty_per_step
    if alpha_jerk             > 0.0:  reward_kwargs["alpha_jerk"]             = alpha_jerk
    if alpha_heading          > 0.0:  reward_kwargs["alpha_heading"]          = alpha_heading
    if alpha_dist             > 0.0:  reward_kwargs["alpha_dist"]             = alpha_dist
    if alpha_fwd              > 0.0:  reward_kwargs["alpha_fwd"]              = alpha_fwd
    if alpha_backward         > 0.0:  reward_kwargs["alpha_backward"]         = alpha_backward
    if backward_vel_limit     > 0.0:  reward_kwargs["backward_vel_limit"]     = backward_vel_limit
    if lpf_cutoff_hz          > 0.0:  reward_kwargs["lpf_cutoff_hz"]          = lpf_cutoff_hz
    if motor_deadtime_steps   > 0:    reward_kwargs["motor_deadtime_steps"]   = motor_deadtime_steps
    if use_distance_reward:           reward_kwargs["use_distance_reward"]     = True
    reward_kwargs["success_radius"] = success_radius
    reward_kwargs["rewardSuccess"] = reward_success
    reward_kwargs["camera_fps"]    = camera_fps
    env = gym.make(
        base_env_id,
        frame_skip=20,
        render_mode="rgb_array",
        motorVelRange=1,
        actionMultiplier=1,
        rewardFailure=-1.0,
        seed=seed,
        width=800,
        height=800,
        **reward_kwargs,
    )
    env.unwrapped._motor_ctrlrange = motor_ctrlrange
    check_env(env)

    # Stages are defined on the env class; clean variant uses CURRICULUM_STAGES_CLEAN.
    if "clean" in env_name:
        stages = env.unwrapped.CURRICULUM_STAGES_CLEAN
    else:
        stages = env.unwrapped.CURRICULUM_STAGES

    start_stage = int(getattr(args, "start_stage", 0))
    if start_stage > 0:
        print(f"Skipping stages 0–{start_stage - 1} (start_stage={start_stage}).")
        stages = stages[start_stage:]

    env = Monitor(env, log_dir)

    model_file_path = getattr(args, 'modelFilePath', None)
    if model_file_path:
        # Load checkpoint directly — SAC.load reads architecture from the file so
        # hiddenDim/numLayers in wandb.config don't need to match the checkpoint.
        print(f"Loading pretrained model from {model_file_path}")
        model = SAC.load(model_file_path, env=env, device="cpu")
        model.set_random_seed(seed)
        # SAC.load() creates the default ReplayBuffer, not MirroredReplayBuffer.
        # Replace it here so mirror_replay_buffer=True takes effect for finetune runs.
        _buf_size = 2_000_000 if use_mirror else 1_000_000
        model.replay_buffer = (MirroredReplayBuffer if use_mirror else
                                model.replay_buffer.__class__)(
            buffer_size=_buf_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=model.device,
            n_envs=1,
            optimize_memory_usage=False,
        )
        model.buffer_size = _buf_size
        # learning_starts=0: SB3 samples random (not checkpoint) actions when
        # num_timesteps < learning_starts.  With learning_starts=0 the policy
        # is used from step 0.  The actual warm-up is the pre-fill below.
        model.learning_starts = 0
        # Reset LR to the sweep-specified finetune rate. The checkpoint's baked-in LR
        # has been decayed ×0.3 per stage through all of phase 1_1 — it's far too
        # small for a new phase. args.learningRate comes from the finetune sweep YAML.
        _init_lr = args.learningRate
        model.learning_rate = _init_lr
        for _opt in [model.actor.optimizer, model.critic.optimizer]:
            for _pg in _opt.param_groups:
                _pg["lr"] = _init_lr
        if hasattr(model, "ent_coef_optimizer"):
            for _pg in model.ent_coef_optimizer.param_groups:
                _pg["lr"] = _init_lr
        # Apply swept tau — SAC.load() bakes in the checkpoint's tau (from whichever
        # sweep produced it); without this the tau sweep dimension is wasted.
        model.tau = float(args.tau)
        print(f"Fine-tune start: learning_starts=0, LR={_init_lr:.2e}, "
              f"tau={model.tau:.4f}, "
              f"buffer={'Mirrored' if use_mirror else 'Standard'}({_buf_size:,})")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [args.hiddenDim] * args.numLayers},
            learning_rate=args.learningRate,
            batch_size=args.batchSize,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=args.train_freq,
            learning_starts=args.learningStarts,
            buffer_size=2_000_000 if use_mirror else 1_000_000,
            replay_buffer_class=MirroredReplayBuffer if use_mirror else None,
            device="cpu",
            verbose=0,
            seed=seed,
        )

    # Progressive-obs envs (e.g. sofi_da_angular_prog_curriculum) use a 40-dim
    # network but only 24 dims are non-zero at stage 0.  PyTorch kaiming init
    # scales weights for fan_in=obs_dim, so the active-dim signal is
    # sqrt(24/40) ≈ 0.775× too weak → ~2× slower bootstrapping → more runs
    # collapse to the "always swim forward" 60% local optimum before clearing
    # stage 0.  Rescaling the active columns by sqrt(fan_in/n_active) restores
    # the correct pre-activation variance.  Applied once at init; does not need
    # to be repeated at LR resets (weights have been trained by then).
    _obs_dim = env.observation_space.shape[0]
    _n_active = getattr(env.unwrapped, 'N_ACTIVE_AT_INIT', 24)
    if not model_file_path and _obs_dim > 24 and _n_active < _obs_dim:
        # encoder+IMU (0:4), all 3 recent frames (12:24), motor setpoints.
        # Motor history is interleaved [setpoint, actual] pairs starting at [24],
        # so setpoints sit at even indices [24,26,...,38] — range(24, obs_dim, 2).
        _active_cols = list(range(0, 4)) + list(range(12, 24)) + list(range(24, _obs_dim, 2))
        _scale = float((_obs_dim / _n_active) ** 0.5)
        with torch.no_grad():
            model.actor.latent_pi[0].weight[:, _active_cols] *= _scale
            model.critic.qf0[0].weight[:, _active_cols] *= _scale
            model.critic.qf1[0].weight[:, _active_cols] *= _scale
        print(f"Init correction: active obs cols ×{_scale:.3f} "
              f"(obs_dim={_obs_dim}, n_active={_n_active})")

    sensor_model  = env.unwrapped._sensor_model
    physics_model = getattr(env.unwrapped, "_physics_model", None)

    # Separate eval env — same kwargs but no render, different seed.
    # Sensor/physics settings are synced from the training env before each eval.
    eval_env = gym.make(
        base_env_id,
        frame_skip=20,
        render_mode=None,
        motorVelRange=1,
        actionMultiplier=1,
        rewardFailure=-1.0,
        seed=seed + 1,
        **reward_kwargs,
    )
    eval_env.unwrapped._motor_ctrlrange = motor_ctrlrange

    plot_cb = PlotCallback(check_freq=10_000, log_dir=log_dir, reward_window=SUCCESS_WINDOW)
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000, save_path=log_dir, name_prefix="SAC_curriculum"
    )
    eval_freq       = int(getattr(args, "eval_freq",       50_000))
    eval_freq_early = int(getattr(args, "eval_freq_early", 200_000))
    n_eval_ep       = int(getattr(args, "n_eval_episodes",  100))
    stage_ref = [start_stage]  # mutable; updated each stage so eval_cb can read it
    eval_cb = EvalSuccessCallback(
        train_env=env,
        eval_env=eval_env,
        eval_freq=eval_freq,
        eval_freq_slow=eval_freq * 2,
        n_eval_fast=n_eval_ep,
        n_eval_slow=max(50, n_eval_ep // 2),
        stage_ref=stage_ref,
        eval_freq_early=eval_freq_early,
        log_dir=log_dir,
    )

    initial_lr = model.learning_rate if callable(model.learning_rate) else float(model.learning_rate)

    try:
        for stage_idx, stage in enumerate(stages):
            abs_stage_idx = start_stage + stage_idx  # absolute index for logging/filenames
            stage_ref[0] = abs_stage_idx  # keep eval_cb aware of current stage
            # Flip sensor flags for this stage (starts clean at stage 0).
            sensor_model.enable_imu          = stage["enable_imu"]
            sensor_model.enable_encoder      = stage["enable_encoder"]
            sensor_model.enable_vision       = stage["enable_vision"]
            sensor_model.enable_vision_noise = stage["enable_vision_noise"]
            sensor_model.enable_latency      = stage["enable_latency"]
            sensor_model.enable_fps_dr       = stage.get("enable_fps_dr", False)
            # Frame-buffer stages carry per-stage latency/drop overrides.
            if "latency_min" in stage:
                sensor_model.latency_min = stage["latency_min"]
                sensor_model.latency_max = stage["latency_max"]
            if "frame_drop_prob" in stage:
                sensor_model.frame_drop_prob = stage["frame_drop_prob"]
            # Physics DR curriculum (phase 1.2): toggle perturbation groups.
            if physics_model is not None and "enable_fluid" in stage:
                physics_model.enable_init_vel  = stage.get("enable_init_vel",  False)
                physics_model.enable_inertial  = stage.get("enable_inertial",  False)
                physics_model.enable_mechanics = stage.get("enable_mechanics", False)
                physics_model.enable_fluid     = stage.get("enable_fluid",     False)
                physics_model.fluid_std        = stage.get("fluid_std",        0.20)
                physics_model.enable_motor        = stage.get("enable_motor",        False)
                env.unwrapped._motor_dr_active    = stage.get("enable_motor",        False)
                env.unwrapped._extra_frames_active = stage.get("enable_extra_frames", False)
                # Per-stage motor DR range overrides (ADR split — see env_sofi_da_angular_prog_curriculum.py).
                # If absent, physics_model retains its previous range — correct for fluid-DR stages
                # that follow motor_dr_full (range stays at [1.0, 8.0] without needing to repeat it).
                if "motor_kp_range" in stage:
                    physics_model.motor_kp_range = tuple(stage["motor_kp_range"])
                if "motor_kv_range" in stage:
                    physics_model.motor_kv_range = tuple(stage["motor_kv_range"])
                if "motor_forcerange_range" in stage:
                    physics_model.motor_forcerange_range = tuple(stage["motor_forcerange_range"])
                if "motor_deadtime_range" in stage:
                    physics_model.motor_deadtime_range = tuple(stage["motor_deadtime_range"])
                if "motor_armature_range" in stage:
                    physics_model.motor_armature_range = tuple(stage["motor_armature_range"])
                if "motor_zero_offset_std" in stage:
                    physics_model.motor_zero_offset_std = float(stage["motor_zero_offset_std"])

            # Zero-init newly-activated obs dims to prevent Kaiming-noise perturbation
            # of the Q-function at each obs unlock point.  Guard stage_idx>0: at stage 0
            # prev_stage defaults to {} (False for everything), so envs that activate all
            # dims from stage 0 (e.g. N_ACTIVE_AT_INIT=40) would otherwise zero out dims
            # that are already warm.
            prev_stage = stages[stage_idx - 1] if stage_idx > 0 else {}
            if stage_idx > 0 and stage.get("enable_extra_frames", False) and not prev_stage.get("enable_extra_frames", False):
                _frame_cols = list(range(4, 12))
                with torch.no_grad():
                    model.actor.latent_pi[0].weight[:, _frame_cols] = 0.0
                    model.critic.qf0[0].weight[:, _frame_cols] = 0.0
                    model.critic.qf1[0].weight[:, _frame_cols] = 0.0
                print("Zero-init: frame slots 0–1 (obs dims 4–11)")
            if stage_idx > 0 and stage.get("enable_motor", False) and not prev_stage.get("enable_motor", False):
                _actual_cols = list(range(25, 40, 2))
                with torch.no_grad():
                    model.actor.latent_pi[0].weight[:, _actual_cols] = 0.0
                    model.critic.qf0[0].weight[:, _actual_cols] = 0.0
                    model.critic.qf1[0].weight[:, _actual_cols] = 0.0
                print("Zero-init: motor actuals (obs dims 25,27,…,39)")
            # Polar-target curriculum: widen the spawn radius / angle.
            if "r_min" in stage:
                env.unwrapped.r_min = stage["r_min"]
                env.unwrapped.r_max = stage["r_max"]
                env.unwrapped.theta_max = float(np.deg2rad(stage["theta_max_deg"]))

            if stage_idx == 0:
                new_lr = model.learning_rate
            elif stage.get("reset_lr", False):
                # Phase boundary: reset to the original learning rate so the
                # policy can adapt quickly to the new challenge introduced by
                # this phase (mirrors manually resetting LR between phases).
                new_lr = initial_lr
                model.learning_rate = new_lr
                for opt in [model.actor.optimizer, model.critic.optimizer]:
                    for pg in opt.param_groups:
                        pg["lr"] = new_lr
                if hasattr(model, "ent_coef_optimizer"):
                    for pg in model.ent_coef_optimizer.param_groups:
                        pg["lr"] = new_lr
            else:
                new_lr = model.learning_rate * lr_scale
                model.learning_rate = new_lr
                # SAC has separate actor/critic optimizers — no single policy.optimizer
                for opt in [model.actor.optimizer, model.critic.optimizer]:
                    for pg in opt.param_groups:
                        pg["lr"] = new_lr
                if hasattr(model, "ent_coef_optimizer"):
                    for pg in model.ent_coef_optimizer.param_groups:
                        pg["lr"] = new_lr

            print(f"\n{'='*60}")
            print(f"  Curriculum stage {abs_stage_idx}: {stage['name']}")
            print(f"  LR → {new_lr:.2e}")
            print(f"{'='*60}\n")

            if os.environ.get("WANDB_MODE", "offline") == "online":
                # stage_idx==0 uses reset_num_timesteps=True, so model.num_timesteps
                # still reflects the loaded checkpoint (e.g. 750k). Log at 0
                # to match where model.learn() will actually start.
                log_step = 0 if stage_idx == 0 else model.num_timesteps
                wandb.log({
                    "curriculum/stage":      abs_stage_idx,
                    "curriculum/stage_name": stage["name"],
                    "curriculum/lr":         new_lr,
                }, step=log_step)

            # Pre-fill replay buffer with checkpoint-policy transitions before the
            # first gradient update.  Without this, training starts with 4–256 buffer
            # entries and the Q-function overfits immediately, causing collapse.
            # Sensor and physics flags above are already set to this stage's config,
            # so pre-fill data comes from the correct distribution.
            # Only done at stage_idx==0 (first stage of finetune) and only when a
            # checkpoint was loaded (otherwise buffer starts fresh from scratch).
            if stage_idx == 0 and model_file_path:
                _prefill = int(getattr(args, 'prefillSteps', 50_000))
                print(f"Pre-filling replay buffer: {_prefill:,} steps with checkpoint "
                      f"policy at {stage['name']} physics (no gradient updates)...")
                _pf_obs, _ = env.reset()
                for _pf_i in range(_prefill):
                    with torch.no_grad():
                        _pf_action, _ = model.predict(_pf_obs, deterministic=False)
                    _pf_next, _pf_rew, _pf_done, _pf_trunc, _pf_info = env.step(_pf_action)
                    model.replay_buffer.add(
                        np.array([_pf_obs]),
                        np.array([_pf_next]),
                        np.array([_pf_action]),
                        np.array([_pf_rew]),
                        np.array([_pf_done or _pf_trunc]),
                        [_pf_info],
                    )
                    _pf_obs = _pf_next if not (_pf_done or _pf_trunc) else env.reset()[0]
                print(f"Buffer pre-filled: {model.replay_buffer.size():,} transitions "
                      f"({model.replay_buffer.size() / model.buffer_size * 100:.1f}% full)")

            stop_cb = StageStopCallback(
                threshold=float(getattr(args, "success_threshold", SUCCESS_THRESHOLD)),
                window=int(stage.get("success_window", SUCCESS_WINDOW)),
                stage_idx=abs_stage_idx,
                min_stage_steps=int(stage.get("min_stage_steps", 0)),
                check_freq=CHECK_FREQ,
            )

            # Each stage gets its own nSteps budget. reset_num_timesteps=False
            # keeps the WandB step counter monotonically increasing across stages.
            model.learn(
                total_timesteps=args.nSteps,
                callback=[stop_cb, plot_cb, checkpoint_cb, eval_cb],
                reset_num_timesteps=(stage_idx == 0),
            )

            stage_path = f"{log_dir}/SAC_curriculum_stage{abs_stage_idx}_{stage['name']}"
            model.save(stage_path)
            model.save_replay_buffer(stage_path + "_replay_buffer")
            print(f"Stage {abs_stage_idx} ({stage['name']}) saved → {stage_path} (+ replay buffer)")

            if not stop_cb._advanced:
                print(f"\nStage {abs_stage_idx} ({stage['name']}) did not reach "
                      f"{stop_cb.threshold:.1%} success rate within {args.nSteps:,} steps "
                      f"— stopping run.")
                return

    finally:
        model.save(f"{log_dir}/SAC_curriculum_final")
        model.save_replay_buffer(f"{log_dir}/SAC_curriculum_final_replay_buffer")
        print(f"Model saved → {log_dir}/SAC_curriculum_final (+ replay buffer)")
        if os.environ.get("WANDB_MODE", "offline") == "online":
            wandb.finish()

    # Return final logs for hyperparameter optimization script.
    # 'dists' and 'figs' are empty: curriculum training doesn't run test trajectories.
    _, y = ts2xy(load_results(log_dir), "timesteps")
    if len(y) > 0:
        return {"rewards": np.array(y), "dists": np.array([]), "figs": []}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nSteps",          type=int,   default=3_000_000)
    parser.add_argument("--maxEpisodeSteps", type=int,   default=500)
    parser.add_argument("--hiddenDim",       type=int,   default=256)
    parser.add_argument("--numLayers",       type=int,   default=3)
    parser.add_argument("--learningRate",    type=float, default=3e-4)
    parser.add_argument("--batchSize",       type=int,   default=512)
    parser.add_argument("--gamma",           type=float, default=0.995)
    parser.add_argument("--tau",             type=float, default=0.005)
    parser.add_argument("--train_freq",      type=int,   default=4)
    parser.add_argument("--learningStarts",  type=int,   default=5_000)
    parser.add_argument("--prefillSteps",    type=int,   default=0,
                        help="When loading a checkpoint: collect this many steps with the "
                             "checkpoint policy (no gradient updates) to warm up the replay "
                             "buffer before training starts.")
    parser.add_argument("--lr_scale",        type=float, default=LR_SCALE)
    parser.add_argument("--seed",            type=int,   default=SEED)
    parser.add_argument("--logDir",          type=str,   default="outputs")
    parser.add_argument("--env",             type=str,   default="sofi_phase1_1_noiseAndDelay",
                        help="Curriculum variant: 'sofi_phase1_1_noiseAndDelay', "
                             "'sofi_phase2_polarTarget', or 'sofi_phase2_polarTarget_clean'")
    parser.add_argument("--load", action="store_true", help="Load a pretrained model before training")
    parser.add_argument("--modelFile", type=str, default=None, help="Explicit path to pretrained model parameters (set model.set_parameters path)")
    parser.add_argument("--start_stage", type=int, default=0, help="Skip the first N curriculum stages (useful when finetuning from a checkpoint that already mastered them)")
    parser.add_argument("--alpha_jerk",     type=float, default=0.0,
                        help="3rd-order jerk penalty weight for DA envs")
    parser.add_argument("--alpha_fwd",      type=float, default=0.0,
                        help="Forward-swimming bonus weight (fires when fish moves toward nose)")
    parser.add_argument("--alpha_backward", type=float, default=0.0,
                        help="Backward-swimming penalty weight (fires when fish moves toward tail)")
    parser.add_argument("--backward_vel_limit", type=float, default=0.0,
                        help="Terminate episode if backward velocity exceeds this (m/s). 0=disabled.")
    parser.add_argument("--lpf_cutoff_hz",  type=float, default=0.0,
                        help="First-order low-pass filter on the motor setpoint (Hz). 0 = off. The DA-FOF policy of the paper used 4.5 Hz.")
    parser.add_argument("--motor_deadtime_steps", type=int, default=0,
                        help="Motor transport delay in physics steps (1ms each). Identified ~42 (≈2 policy steps). 0=disabled.")
    parser.add_argument("--use_distance_reward", action="store_true", default=False,
                        help="Use CPG-style reward: r = -dist + rewardSuccess (SDA default)")
    parser.add_argument("--success_radius", type=float, default=0.1,
                        help="Distance threshold (m) to declare success (default: 0.1)")
    parser.add_argument("--eval_freq",       type=int, default=50_000,
                        help="Eval frequency (fast/slow) once past early_stage_threshold")
    parser.add_argument("--eval_freq_early", type=int, default=200_000,
                        help="Slow-only eval frequency before early_stage_threshold (stage 4)")
    parser.add_argument("--n_eval_episodes", type=int, default=100,
                        help="Episodes per eval (stochastic + deterministic each)")

    args = parser.parse_args()
    main(args)
