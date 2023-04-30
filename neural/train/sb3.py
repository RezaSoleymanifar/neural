import os
from torch import nn
from stable_baselines3 import PPO

from neural.train.base import AbstractTrainer
from neural.meta.agent import Agent

class StableBaselinesTrainer(AbstractTrainer):

    def __init__(
        self,
        agent: Agent,
        file_path: os.PathLike,
        dataset_name: str,
        verbose: bool,
        n_chunks: int = 1,
        ) -> None:

        super().__init__(agent, file_path, dataset_name, n_chunks=1)
        self.verbose = verbose


    def train(
        self, steps: int = 10000, **kwargs) -> nn.Module:

        piped_market_env = self._get_train_market_env()
        model = self.agent.model
        
        ppo = PPO(policy=model, env=piped_market_env, verbose=self.verbose)
        PPO.learn(total_timesteps=steps, env=piped_market_env)
        model = ppo.policy.to("cpu")

        return model
