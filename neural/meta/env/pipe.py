from abc import abstractmethod, ABC
from typing import Optional

from neural.meta.env.wrapper import (
    MinTradeSizeActionWrapper,
    FixedMarginActionWrapper, 
    NetWorthRelativeMaximumShortSizing,
    NetWorthRelativeUniformPositionSizing, 
    ConsoleTearsheetRenderWrapper,
    MarketEnvMetadataWrapper,
    IntegerAssetQuantityActionWrapper,
    ActionClipperWrapper,
    FlattenDictToNumpyObservationWrapper)

from neural.meta.env.base import AbstractMarketEnv



class AbstractPipe(ABC):   


    @abstractmethod
    def pipe(self, env):

        # abstract method for piping an environment. Wrappers
        # are addedd successively akin to layers in PyTorch.

        raise NotImplementedError


    def warmup(
            self,
            env: AbstractMarketEnv,
            n_episodes: Optional[int] = None
            ) -> None:
        
        # runs env with random actions. Useful for tuning
        # parameters of normalizer wrappers that depend
        # on having seen observations.

        piped_env = self.pipe(env)

        for _ in n_episodes:

            _ = piped_env.reset()

            while True:

                action = piped_env.action_space.sample()
                *_, done, _ = piped_env.step(action)

                if done:
                    break

        return None


class NetWorthRelativeShortMarginPipe(AbstractPipe):

    # A pipe is a stack of market wrappers applied in a non-conflicting way.
    # User wrappers to customize the base market env, manipulate actions
    # and observations, impose trading logic,  etc. according to your
    # specific needs.

    def __init__(
            self,
            min_action: float = 1,
            integer: bool = False,
            initial_margin: float = 1,
            short_ratio: float = 0,
            trade_ratio: float = 0.02,
            verbosity: int = 20,
            ) -> None:
        
        self.min_action = min_action
        self.integer = integer
        self.initial_margin = initial_margin
        self.short_ratio = short_ratio
        self.trade_ratio = trade_ratio
        self.verbosity = verbosity

        self.metadata_wrapper = MarketEnvMetadataWrapper
        self.integer_sizing = IntegerAssetQuantityActionWrapper
        self.min_trade = MinTradeSizeActionWrapper
        self.position_sizing = NetWorthRelativeUniformPositionSizing
        self.margin_sizing = FixedMarginActionWrapper
        self.short_sizing = NetWorthRelativeMaximumShortSizing
        self.clip = ActionClipperWrapper
        self.render = ConsoleTearsheetRenderWrapper
        self.flatten = FlattenDictToNumpyObservationWrapper

        return None

    def pipe(self, env):
        
        env = self.metadata_wrapper(env)
        env = self.min_trade(env, min_action=self.min_action)
        env = self.integer_sizing(env, self.integer)
        env = self.margin_sizing(env, initial_margin = self.initial_margin)
        env = self.short_sizing(env, short_ratio = self.short_ratio)
        env = self.position_sizing(env, trade_ratio=self.trade_ratio)
        env = self.clip(env)
        env = self.flatten(env)
        env = self.render(env, verbosity = self.verbosity)

        return env
