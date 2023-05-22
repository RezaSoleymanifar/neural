from abc import abstractmethod, ABC
from typing import Optional

from neural.wrapper.base import (
    MarginAccountMetaDataWrapper,
    ConsoleTearsheetRenderWrapper)

from neural.wrapper.action import (
    MinTradeSizeActionWrapper,
    IntegerAssetQuantityActionWrapper,
    PositionCloseActionWrapper,
    InitialMarginActionWrapper,
    ExcessMarginActionWrapper,
    ShortingActionWrapper,
    EquityBasedUniformActionInterpreter,
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
    Abstract class for environment pipes, which add extended
    functionality to an existing environment by applying wrappers
    successively. A pipe is a stack of wrappers applied in a
    non-conflicting way. Use wrappers to customize the base market env,
    manipulate actions and observations, impose trading logic, etc.
    according to your specific needs. Wrappers are intantiated every
    time the pipe method is called. If you need to restore state of some
    wrappers, you can make that state a constructor argument of both
    wrapper class and and the pipe and set the argument passed to
    wrapper equal to state of wrapper. If both satate are immutable, the
    values will be synchronized.
    """


    @abstractmethod
    def pipe(self, env):
        """
        Abstract method for piping an environment. Wrappers
        are added successively akin to layers in PyTorch.
        """

        raise NotImplementedError


class ObservationPipe(AbstractPipe)

    def __init__(
        self,
        trade_equity_ratio: float = 0.02,
        min_trade: float = 1,
        integer: bool = False,
        buffer_size: int = 10,
        stack_size: int = None,
        observation_statistics: Optional[RunningStatistics] = None,
        track_statistics: bool = True
        ) -> None:

class MarginAccountPipe(AbstractPipe):

    """
    A pipe for margin account environments. The pipe adds the trading
    logics of a margin account to the base market environment.
    """

    def __init__(
        self,
        trade_equity_ratio: float = 0.02,
        min_trade: float = 1,
        integer: bool = False,
        buffer_size: int = 10,
        stack_size: int = None,
        verbosity: int = 20,
        observation_statistics: Optional[RunningStatistics] = None,
        reward_statistics: Optional[RunningStatistics] = None,
        track_statistics: bool = True
        ) -> None:
    
        self.trade_ratio = trade_equity_ratio
        self.short_ratio = short_ratio
        self.initial_margin = initial_margin
        self.min_trade = min_trade
        self.integer = integer
        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.verbosity = verbosity
        self.observation_statistics = observation_statistics
        self.reward_statistics = reward_statistics
        self.track_statistics = track_statistics

        self.metadata_wrapper = MarginAccountMetaDataWrapper
        self.render = ConsoleTearsheetRenderWrapper

        
        self.min_trade = MinTradeSizeActionWrapper
        self.integer_sizing = IntegerAssetQuantityActionWrapper
        self.position_close = PositionCloseActionWrapper
        self.initial_margin = InitialMarginActionWrapper
        self.excess_margin = ExcessMarginActionWrapper
        self.shorting = ShortingActionWrapper
        self.action_interpreter = EquityBasedUniformActionInterpreter
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

        env = self.normalize_observation(
            env, observation_statistics=self.observation_statistics,
            track_statistics=self.track_statistics)

        # action wrappers
        env = self.min_trade(env, min_action=self.min_trade)
        env = self.integer_sizing(env, self.integer)
        env = self.margin_sizing(env, initial_margin=self.initial_margin)
        env = self.short_sizing(env, short_ratio=self.short_ratio)
        env = self.action_interpreter(env, trade_ratio=self.trade_ratio)

        env = self.normalize_reward(
            env, self.reward_statistics, track_statistics=self.track_statistics)
                
        return env
