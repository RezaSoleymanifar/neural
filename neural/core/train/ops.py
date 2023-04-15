from abc import ABC, abstractmethod
from neural.meta.env.base import AbstractMarketEnv
from typing import Optional

class AbstractTrainer(ABC):

    @abstractmethod
    def train(self, *args, **kwargs):

        raise NotImplementedError

    # uses static dataset and random actions to
    def warmup_pipe(
        self, 
        warmup_env: AbstractMarketEnv, 
        n_episodes: Optional[int] = None):

        for _ in n_episodes:
            piped_env = self.pipe(warmup_env)
            observation = piped_env.reset()

            while True:
                action = piped_env.action_space.sample()
                observation, reward, done, info = piped_env.step(action)
                if done:
                    break
