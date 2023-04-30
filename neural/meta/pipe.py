from abc import abstractmethod, ABC
from typing import Optional

from neural.wrapper.base import (
    MarketEnvMetadataWrapper, 
    ConsoleTearsheetRenderWrapper)

from neural.wrapper.action import (
    MinTradeSizeActionWrapper,
    FixedMarginActionWrapper, 
    NetWorthRelativeMaximumShortSizing,
    NetWorthRelativeUniformPositionSizing, 
    ConsoleTearsheetRenderWrapper,
    MarketEnvMetadataWrapper,
    IntegerAssetQuantityActionWrapper,
    ActionClipperWrapper)

from neural.wrapper.observation import (
    ObservationStackerWrapper, 
    ObservationBufferWrapper, 
    FlattenToNUmpyObservationWrapper,
    NormalizeObservationWrapper)

from neural.wrapper.reward import NormalizeRewardWrapper

from neural.utils.base import RunningStatistics

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




class NetWorthRelativeShortMarginPipe(AbstractPipe):

    """
    A pipe is a stack of wrappers applied in a non-conflicting way. Use
    wrappers to customize the base market env, manipulate actions and
    observations, impose trading logic, etc. according to your specific needs.
    Wrappers are intantiated every time the pipe method is called. If you need 
    to restore state of some wrappers, you can make that state a constructor
    argument of both wrapper class and and the pipe and return the state in wrapper constructor
    so that pipe can track this state and pass it to the wrapper constructor when loading the pipe.
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
        observation_statistics: Optional[RunningStatistics] = None,
        reward_statistics: Optional[RunningStatistics] = None,
        ) -> None:
    

        self.trade_ratio = trade_ratio
        self.short_ratio = short_ratio
        self.initial_margin = initial_margin
        self.min_action = min_action
        self.integer = integer
        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.verbosity = verbosity
        self.observation_statistics = observation_statistics
        self.reward_statistics = reward_statistics

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
        self.normalize_observation = NormalizeObservationWrapper

        self.normalize_reward = NormalizeRewardWrapper

        return None


    def pipe(self, env):

        """
        Applies a stack of market wrappers successively to an environment.
        Wrappers are addedd successively akin to layers in PyTorch. state of
        wrappers are seved to an attribute of the Pipe class so that they can
        be restored later.

        Args:
        - env (AbstractMarketEnv): the environment to be wrapped.

        Returns:
        - env (gym.Env): the wrapped environment.
        """

        # helper wrappers
        env = self.metadata_wrapper(env)
        env = self.render(env, verbosity=self.verbosity)

        # observation wrappers
        env = self.flatten(env)
        env = self.buffer(env, buffer_size= self.buffer_size)
        env = self.stacker(env, stack_size = self.stack_size)

        env, self.observation_statistics = self.normalize_observation(
            env, observation_statistics=self.observation_statistics)

        # action wrappers
        env = self.min_trade(env, min_action=self.min_action)
        env = self.integer_sizing(env, self.integer)
        env = self.margin_sizing(env, initial_margin=self.initial_margin)
        env = self.short_sizing(env, short_ratio=self.short_ratio)
        env = self.position_sizing(env, trade_ratio=self.trade_ratio)

        env, self.reward_statistics = self.normalize_reward(env, self.reward_statistics)

        
        return env
