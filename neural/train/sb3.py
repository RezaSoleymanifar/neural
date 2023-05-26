import os
from typing import Optional, Tuple
from torch import nn
import torch

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG, HerReplayBuffer

from neural.train.base import AbstractTrainer
from neural.meta.agent import Agent


class StableBaselinesTrainer(AbstractTrainer):

    def __init__(self,
                 agent: Agent,
                 file_path: os.PathLike,
                 dataset_name: str,
                 n_chunks: int = 1,
                 train_ratio: float = 1,
                 n_envs: int = 1,
                 async_envs: bool = True,
                 exclusive_envs: True = False,
                 initial_cash_range: Optional[Tuple[float, float]] = None,
                 initial_assets_range: Optional[Tuple[float, float]] = None,
                 *args,
                 **kwargs) -> None:

        super().__init__(agent=agent,
                         file_path=file_path,
                         dataset_name=dataset_name,
                         n_chunks=n_chunks,
                         train_ratio=train_ratio,
                         n_envs=n_envs,
                         async_envs=async_envs,
                         exclusive_envs=exclusive_envs,
                         initial_cash_range=initial_cash_range,
                         initial_assets_range=initial_assets_range,
                         *args,
                         **kwargs)

        return None

    def test(self, n_episode: int = 1):

        piped_market_env = self._get_piped_env()
        observation = piped_market_env.reset()

        with torch.no_grad(), torch.set_grad_enabled(False):
            for _ in range(n_episode):
                done = False
                while not done:
                    action = self.agent.model(observation)
                    observation, reward, done, info = piped_market_env.step(
                        action)

    def train(self,
              algorithm: OnPolicyAlgorithm,
              steps: int = 1_000_000,
              **kwargs) -> nn.Module:

        piped_market_env = self._get_piped_env()
        model = self.agent.model

        algorithm_ = algorithm(policy=model,
                               env=piped_market_env,
                               verbose=self.verbose)
        algorithm_.learn(total_timesteps=steps, env=piped_market_env, **kwargs)

        return None
