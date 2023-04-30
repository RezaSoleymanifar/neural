import os

from abc import ABC, abstractmethod
from torch import nn

from stable_baselines3 import PPO

from neural.env.base import AbstractMarketEnv
from neural.meta.agent import Agent


class AbstractTrainer(ABC):

    def __init__(self, agent = Agent) -> None:
        self.agent = agent

    def construct_market_env(
        file_path: os.PathLike,
        dataset_name: str, 
        *args, 
        **kwargs) -> AbstractMarketEnv:

    @abstractmethod
    def train(
        self,
        market_env: AbstractMarketEnv, 
        *args, 
        **kwargs) -> nn.Module:

        raise NotImplementedError



class StableBaselinesTrainer(AbstractTrainer):

    def __init__(
        self, 
        agent: Agent,
        verbose: bool,
        ) -> None:
        super().__init__(agent)
        self.verbose = verbose


    def train(
        self,
        market_env: AbstractMarketEnv,
        steps: int = 10000,
        *args, 
        **kwargs) -> nn.Module:

        model = self.agent.model
        ppo = PPO(policy=model, env=market_env, verbose=self.verbose)
        PPO.learn(total_timesteps=steps, env=market_env)
        model = ppo.policy.to("cpu")

        return model
        
