from typing import List, Type, Dict, Tuple, Optional
from gym import Wrapper
from torch import nn


from neural.meta.env.wrapper import (
    MinTradeSizeActionWrapper,
    RelativeMarginSizingActionWrapper, 
    RelativeShortSizingActionWrapper,
    ContinuousRelativePositionSizingActionWrapper, 
    ConsoleTearsheetRenderWrapper,
    TrainMarketEnvMetadataWrapper)

from neural.meta.env.base import AbstractMarketEnv, TrainMarketEnv, TradeMarketEnv
from dataclasses import dataclass
from abc import abstractmethod, ABC



class AbstractPipe(ABC):   

    @abstractmethod
    def pipe(self, env):

        raise NotImplementedError
    
    def warmup(
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


class NoShortNoMarginPipe(AbstractPipe):

    def __init__(self) -> None:

        self.metadata_wrapper = TrainMarketEnvMetadataWrapper
        self.min_trade = MinTradeSizeActionWrapper
        self.position_sizing = ContinuousRelativePositionSizingActionWrapper
        self.margin_sizing = RelativeMarginSizingActionWrapper
        self.short_sizing = RelativeShortSizingActionWrapper
        self.render = ConsoleTearsheetRenderWrapper

    def pipe(self, env):
        env = self.metadata_wrapper(env)
        env = self.margin_sizing(initial_margin = 1) # equivalent to no margin
        env = self.short_sizing(short_ratio = 0) # equivalent to no short
        env = self.position_sizing(env, trade_ratio=0.02)
        env = self.render(env, verbosity = 20)
        return env
