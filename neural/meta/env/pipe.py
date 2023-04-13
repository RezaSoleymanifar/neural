from typing import List, Type, Dict, Tuple, Optional
from gym import Wrapper
from torch import nn


from neural.meta.env.wrapper import (
    ContinuousRelativeMarginSizingActionWrapper, 
    RelativeShortSizingActionWrapper,
    RelativePositionSizingActionWrapper, 
    ConsoleTearsheetRenderWrapper,
    MarketEnvMetadataWrapper,
    ConstraintCheckerWrapper,
    EnvWarmupWrapperActionWrapper)

from neural.meta.env.base import AbstractMarketEnv, TrainMarketEnv, TradeMarketEnv
from dataclasses import dataclass
from abc import abstractmethod, ABC



class AbstractPipe(ABC):   

    @staticmethod
    def extend(pipe):
        def extended_pipe(env):

            env = pipe(env)
            env = ConstraintCheckerWrapper(env)

            return env
        
        return extended_pipe


    @abstractmethod
    @extend
    def pipe(self):

        raise NotImplementedError




class NoShortNoMarginPipe(AbstractPipe):

    def __init__(self, env: AbstractMarketEnv) -> None:
        super().__init__()

        self.metadata_wrapper = MarketEnvMetadataWrapper
        self.position_sizing = RelativePositionSizingActionWrapper
        self.margin_sizing = ContinuousRelativeMarginSizingActionWrapper
        self.short_sizing = RelativeShortSizingActionWrapper
        self.render = ConsoleTearsheetRenderWrapper

    @AbstractPipe.extend
    def pipe(self):
        env = self.trade_market_env
        env = self.metadata_wrapper(env)
        env = self.position_sizing(env, trade_ratio = 0.3)
        env = self.render(env, verbosity = 20)
        return env
