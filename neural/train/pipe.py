from abc import abstractmethod, ABC
from typing import Optional

from neural.train.wrapper import (
    MinTradeSizeActionWrapper,
    FixedMarginActionWrapper, 
    NetWorthRelativeMaximumShortSizing,
    NetWorthRelativeUniformPositionSizing, 
    ConsoleTearsheetRenderWrapper,
    MarketEnvMetadataWrapper,
    IntegerAssetQuantityActionWrapper,
    ActionClipperWrapper,
    FlattenToNUmpyObservationWrapper,
    ObservationStackerWrapper, 
    ObservationBufferWrapper)

from neural.train.env.base import AbstractMarketEnv



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

        for episode in n_episodes:
            observation = piped_env.reset()

            while True:
                action = piped_env.action_space.sample()
                observation, reward, done, info = piped_env.step(action)

                if done:
                    break

        return None


class NetWorthRelativeShortMarginPipe(AbstractPipe):

    """
    A pipe is a sequence of market wrappers applied in a non-conflicting way. Use
    wrappers to customize the base market env, manipulate actions and
    observations, impose trading logic, etc. according to your specific needs.
    """

    def __init__(
            self,
            trade_ratio: float = 0.02,
            short_ratio: float = 0,
            initial_margin: float = 1,
            min_action: float = 1,
            integer: bool = False,
            buffer_size: int = 10,
            stack_size: int = None,
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

        self.trade_ratio = trade_ratio
        self.short_ratio = short_ratio
        self.initial_margin = initial_margin
        self.min_action = min_action
        self.integer = integer
        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.verbosity = verbosity

        self.metadata_wrapper = MarketEnvMetadataWrapper
        self.render = ConsoleTearsheetRenderWrapper

        self.integer_sizing = IntegerAssetQuantityActionWrapper
        self.min_trade = MinTradeSizeActionWrapper
        self.margin_sizing = FixedMarginActionWrapper
        self.short_sizing = NetWorthRelativeMaximumShortSizing
        self.position_sizing = NetWorthRelativeUniformPositionSizing
        self.clip = ActionClipperWrapper
        
        self.flatten = FlattenToNUmpyObservationWrapper
        self.buffer = ObservationBufferWrapper
        self.stacker = ObservationStackerWrapper

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

        # helper wrappers
        env = self.metadata_wrapper(env)
        env = self.render(env, verbosity=self.verbosity)

        # observation wrappers
        env = self.flatten(env)
        env = self.buffer(env, buffer_size= self.buffer_size)
        env = self.stacker(env, stack_size = self.stack_size)

        # action wrappers
        env = self.min_trade(env, min_action=self.min_action)
        env = self.integer_sizing(env, self.integer)
        env = self.margin_sizing(env, initial_margin=self.initial_margin)
        env = self.short_sizing(env, short_ratio=self.short_ratio)
        env = self.position_sizing(env, trade_ratio=self.trade_ratio)
        
        return env
