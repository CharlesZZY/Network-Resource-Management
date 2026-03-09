"""M3-PPO: Proximal Policy Optimisation baseline using Stable-Baselines3."""

import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.environment.slicing_env import NetworkSlicingEnv


# ---------------------------------------------------------------------------
# Training environment wrapper
# ---------------------------------------------------------------------------

class MixedScenarioEnv(gym.Wrapper):
    """Randomly switches between steady / burst on each episode reset."""

    def __init__(self, config: dict, burst_prob: float = 0.4):
        env = NetworkSlicingEnv(config, scenario="steady")
        super().__init__(env)
        self.burst_prob = burst_prob

    def reset(self, **kwargs):
        self.env.scenario = (
            "burst" if np.random.random() < self.burst_prob else "steady"
        )
        return self.env.reset(**kwargs)


# ---------------------------------------------------------------------------
# Training callback – episode logging + early stopping
# ---------------------------------------------------------------------------

class _TrainingCallback(BaseCallback):
    """Tracks per-episode rewards and triggers early stopping."""

    def __init__(self, early_stop_window: int = 50,
                 early_stop_threshold: float = 0.01,
                 max_episodes: int = 1000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.early_stop_window = early_stop_window
        self.early_stop_threshold = early_stop_threshold
        self.max_episodes = max_episodes

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_rewards: list[float] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", self.locals.get("done", [False]))
        rewards = self.locals.get("rewards", self.locals.get("reward", [0.0]))

        if not hasattr(dones, "__len__"):
            dones = [dones]
            rewards = [rewards]

        for done, reward in zip(dones, rewards):
            self._current_rewards.append(float(reward))
            if done:
                ep_reward = sum(self._current_rewards)
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(len(self._current_rewards))
                self._current_rewards = []

                n_eps = len(self.episode_rewards)
                if self.verbose >= 1 and n_eps % 50 == 0:
                    recent = self.episode_rewards[-self.early_stop_window:]
                    print(f"  Episode {n_eps}: "
                          f"avg_reward={np.mean(recent):.4f}")

                if n_eps >= self.max_episodes:
                    if self.verbose >= 1:
                        print(f"  Reached max episodes ({self.max_episodes})")
                    return False

                if n_eps >= self.early_stop_window * 2:
                    recent = self.episode_rewards[-self.early_stop_window:]
                    prev = self.episode_rewards[
                        -2 * self.early_stop_window:-self.early_stop_window
                    ]
                    mean_recent = np.mean(recent)
                    mean_prev = np.mean(prev)
                    if abs(mean_prev) > 1e-8:
                        change_rate = abs(
                            (mean_recent - mean_prev) / mean_prev
                        )
                        if change_rate < self.early_stop_threshold:
                            if self.verbose >= 1:
                                print(
                                    f"  Early stopping at episode {n_eps}: "
                                    f"change_rate={change_rate:.4f}"
                                )
                            return False
        return True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Trains a PPO agent on the mixed-scenario slicing environment."""

    def __init__(self, config: dict):
        self.config = config
        ppo_cfg = config.get("ppo", {})
        self.learning_rate = ppo_cfg.get("learning_rate", 3e-4)
        self.n_steps = ppo_cfg.get("n_steps", 2048)
        self.batch_size = ppo_cfg.get("batch_size", 64)
        self.n_epochs = ppo_cfg.get("n_epochs", 10)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.max_episodes = ppo_cfg.get("max_episodes", 1000)
        self.early_stop_window = ppo_cfg.get("early_stop_window", 50)
        self.early_stop_threshold = ppo_cfg.get("early_stop_threshold", 0.01)
        self.burst_prob = ppo_cfg.get("burst_prob", 0.4)

    def train(self, model_path: str = "results/models/ppo_model",
              curve_path: str = "results/data/ppo_training_curve.csv",
              verbose: int = 1) -> PPO:
        env = MixedScenarioEnv(self.config, burst_prob=self.burst_prob)

        total_timesteps = (
            self.max_episodes
            * self.config["experiment"].get("num_cycles", 100)
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            verbose=0,
        )

        callback = _TrainingCallback(
            early_stop_window=self.early_stop_window,
            early_stop_threshold=self.early_stop_threshold,
            max_episodes=self.max_episodes,
            verbose=verbose,
        )

        if verbose >= 1:
            print(f"  PPO training: max_episodes={self.max_episodes}, "
                  f"total_timesteps={total_timesteps}")

        model.learn(total_timesteps=total_timesteps, callback=callback)

        model.save(model_path)
        if verbose >= 1:
            print(f"  Model saved to {model_path}.zip")

        curve_df = pd.DataFrame({
            "episode": range(1, len(callback.episode_rewards) + 1),
            "reward": callback.episode_rewards,
            "length": callback.episode_lengths,
        })
        curve_df.to_csv(curve_path, index=False)
        if verbose >= 1:
            print(f"  Training curve saved to {curve_path}")

        return model


# ---------------------------------------------------------------------------
# Evaluation policy wrapper (used by exp1)
# ---------------------------------------------------------------------------

class PPOPolicy:
    """Wraps a trained PPO model to match the decide(obs, env) interface."""

    def __init__(self, model_path: str = "results/models/ppo_model.zip"):
        self.model = PPO.load(model_path)

    def decide(self, obs, env=None):
        t0 = time.time()
        action, _ = self.model.predict(obs, deterministic=True)
        latency = time.time() - t0
        return action, latency, 0
