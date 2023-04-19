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

    """
    Abstract class for environment pipes, which add functionality to an existing
    environment by applying wrappers successively.
    """


    @abstractmethod
    def pipe(self, env):
        """
        Abstract method for piping an environment. Wrappers
        are added successively akin to layers in PyTorch.
        """

        raise NotImplementedError


    def warmup(
            self,
            env: AbstractMarketEnv,
            n_episodes: Optional[int] = None
            ) -> None:
        """
        Runs environment with random actions. Useful for tuning
        parameters of normalizer wrappers that depend on having seen observations.
        
        Args:
        - env: AbstractMarketEnv object to warm up.
        - n_episodes: Number of episodes to run. If None, runs until env is done.

        Returns:
        None
        """

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


    """
    A pipe is a stack of market wrappers applied in a non-conflicting way. Use
    wrappers to customize the base market env, manipulate actions and
    observations, impose trading logic, etc. according to your specific needs.
    """

    def __init__(
            self,
            min_action: float = 1,
            integer: bool = False,
            initial_margin: float = 1,
            short_ratio: float = 0,
            trade_ratio: float = 0.02,
            verbosity: int = 20,
            ) -> None:
        
        """
        Args:
        - min_action (float): minimum trade size.
        - integer (bool): whether the asset quantities must be integers.
        - initial_margin (float): the initial leverage allowed by the trader.
        - short_ratio (float): the percentage of the net worth that can be
        shorted.
        - trade_ratio (float): the percentage of the net worth that is traded
        at each trade.
        - verbosity (int): the level of detail of the output of the market env.
        """

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

        """
        Applies a stack of market wrappers successively to an environment.
        Wrappers are addedd successively akin to layers in PyTorch.

        Args:
        - env (AbstractMarketEnv): the environment to be wrapped.

        Returns:
        - env (AbstractMarketEnv): the wrapped environment.
        """

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
