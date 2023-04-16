from abc import abstractmethod, ABC
from typing import Optional

from neural.meta.env.wrapper import (
    MinTradeSizeActionWrapper,
    RelativeMarginSizingActionWrapper, 
    RelativeShortSizingActionWrapper,
    ContinuousRelativePositionSizingActionWrapper, 
    ConsoleTearsheetRenderWrapper,
    TrainMarketEnvMetadataWrapper)

from neural.meta.env.base import AbstractMarketEnv



class AbstractPipe(ABC):   


    @abstractmethod
    def pipe(self, env):

        raise NotImplementedError


    def warmup(
            self,
            env: AbstractMarketEnv,
            n_episodes: Optional[int] = None
            ) -> None:
        
        piped_env = self.pipe(env)

        for _ in n_episodes:

            _ = piped_env.reset()

            while True:

                action = piped_env.action_space.sample()
                *_, done, _ = piped_env.step(action)

                if done:
                    break

        return None


class ContinuousShortMarginPipe(AbstractPipe):

    def __init__(
            self,
            min_action: float = 1,
            initial_margin: float = 1,
            short_ratio: float = 0,
            trade_ratio: float = 0.02,
            verbosity: int = 20
            ) -> None:
        
        self.min_action = min_action
        self.initial_margin = initial_margin
        self.short_ratio = short_ratio
        self.trade_ratio = trade_ratio
        self.verbosity = verbosity

        self.metadata_wrapper = TrainMarketEnvMetadataWrapper
        self.min_trade = MinTradeSizeActionWrapper
        self.position_sizing = ContinuousRelativePositionSizingActionWrapper
        self.margin_sizing = RelativeMarginSizingActionWrapper
        self.short_sizing = RelativeShortSizingActionWrapper
        self.render = ConsoleTearsheetRenderWrapper

        return None

    def pipe(self, env):

        env = self.min_trade(env, min_action = self.min_action)
        env = self.metadata_wrapper(env)
        env = self.margin_sizing(env, initial_margin = self.initial_margin)
        env = self.short_sizing(env, short_ratio = self.short_ratio)
        env = self.position_sizing(env, trade_ratio=self.trade_ratio)
        env = self.render(env, verbosity = self.verbosity)

        return env
