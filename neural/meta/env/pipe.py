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
        env = self.position_sizing(env, trade_ratio = 0.3)
        env = self.margin_sizing()
        env = self.short_sizing()
        env = self.render(env, verbosity = 20)
        return env
