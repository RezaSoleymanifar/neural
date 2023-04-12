from typing import List, Type, Dict, Tuple
from gym import Wrapper

from neural.meta.env.wrappers import (
    RelativeMarginSizingActionWrapper, 
    RelativeShortSizingActionWrapper,
    RelativePositionSizingActionWrapper, 
    ConsoleTearsheetRenderWrapper,
    MarketEnvMetadataWrapper)

from neural.meta.env.base import TrainMarketEnvWrapper
from dataclasses import dataclass
from abc import abstractmethod, ABC


class AbstractPipe(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.trade = None

    @abstractmethod
    def pipe(self, env):

        raise NotImplementedError
    

class NoShortNoMarginPipe(AbstractPipe):
    def __init__(self) -> None:
        
        self.meta_data = MarketEnvMetadataWrapper
        self.position_sizing = RelativePositionSizingActionWrapper
        self.margin_sizing = RelativeMarginSizingActionWrapper
        self.short_sizing = RelativeShortSizingActionWrapper
        self.render = ConsoleTearsheetRenderWrapper

    def pipe(self, env):
        env = self.meta_data(env)
        env = self.position_sizing(env, trade_ratio = 0.3)
        env = self.render(env, verbosity = 20)
        return env
