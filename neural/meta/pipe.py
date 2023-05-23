from abc import abstractmethod, ABC
from typing import Optional

from neural.wrapper.base import (MarginAccountMetaDataWrapper,
                                 ConsoleTearsheetRenderWrapper)

from neural.wrapper.action import (
    MinTradeSizeActionWrapper, IntegerAssetQuantityActionWrapper,
    PositionCloseActionWrapper, InitialMarginActionWrapper,
    ExcessMarginActionWrapper, ShortingActionWrapper,
    EquityBasedUniformActionInterpreter, ActionClipperWrapper)

from neural.wrapper.observation import (ObservationStackerWrapper,
                                        ObservationBufferWrapper,
                                        FlattenToNUmpyObservationWrapper,
                                        ObservationNormalizerWrapper)

from neural.wrapper.reward import (RewardNormalizerWrapper,
                                   RewardGeneratorWrapper,
                                   LiabilityInterstRewardWrapper)
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
    values will be synchronized pointing at same memory space. This way
    When saving the pipe, the state of the wrappers will be saved as
    well. The pipe class is an abstract class and must be subclassed.

    Notes:
    -----
        Pipes can be combined to create more complex pipes. For example,
        you can save and reuse a predefined set of wrappers as a pipe
        for convenience.
    """

    @abstractmethod
    def pipe(self, env):
        """
        Abstract method for piping an environment. Wrappers
        are added successively akin to layers in PyTorch.
        By calling pipe, the environment is wrapped in a stack
        of wrappers.
        """

        raise NotImplementedError


class RewardPipe(AbstractPipe):
    """
    This pipe adds reward generation, interest on debt, and
    normalization to the base market environment. The pipe adds the
    following functionality to the base environment:
        - Reward generation
        - Interest on debt (cash/asset liability)
        - Reward normalization
    
    Attributes:
    -----------
    reward_statistics (RunningStatistics): 
        statistics of the reward distribution. Useful for restoring 
        state of the pipe after loading from a file. State of the pipe 
        is saved as attributes of the pipe object.
    track_statistics (bool):
        whether to track and update the reward statistics during
        training. If False, the statistics will be tracked and updated
        during training.



    """
    def __init__(self,
                 reward_statistics: Optional[RunningStatistics] = None,
                 track_statistics=True) -> None:

        self.reward_statistics = reward_statistics
        self.track_statistics = track_statistics

        self.reward_generator = RewardGeneratorWrapper
        self.interest = LiabilityInterstRewardWrapper
        self.reward_normalizer = RewardNormalizerWrapper

    def pipe(self, env):
        env = self.reward_generator(env)
        env = self.interest(env)
        env = self.reward_normalizer(env,
                                     reward_statistics=self.reward_statistics,
                                     track_statistics=self.track_statistics)


class ObservationPipe(AbstractPipe):
    """
    Observation pipe for market environments. The pipe adds the
    following functionality to the base environment:
        - Observation flattening (dict/numpy array)
        - Observation buffering (dict/numpy array)
        - Observation stacking (dict/numpy array)
        - Observation normalization (dict/numpy array)
    

    """
    def __init__(self,
                 buffer_size: int = 10,
                 stack_size: int = None,
                 observation_statistics: Optional[RunningStatistics] = None,
                 track_statistics: bool = True) -> None:

        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.observation_statistics = observation_statistics
        self.track_statistics = track_statistics

        self.flatten = FlattenToNUmpyObservationWrapper
        self.buffer = ObservationBufferWrapper
        self.stacker = ObservationStackerWrapper
        self.normalize_observation = ObservationNormalizerWrapper

    def pipe(self, env):
        env = self.flatten(env)
        env = self.buffer(env, buffer_size=self.buffer_size)
        env = self.stacker(env, stack_size=self.stack_size)

        env = self.normalize_observation(
            env,
            observation_statistics=self.observation_statistics,
            track_statistics=self.track_statistics)

        return env


class ActionPipe(AbstractPipe):

    def __init__(self, min_trade: float = 1, integer: bool = False) -> None:

        self.min_trade = min_trade
        self.integer = integer

        self.min_trade = MinTradeSizeActionWrapper
        self.integer_quantity = IntegerAssetQuantityActionWrapper
        self.position_close = PositionCloseActionWrapper
        self.shorting = ShortingActionWrapper

    def pipe(self, env):
        env = self.min_trade(env, min_trade=self.min_trade)
        env = self.integer_quantity(env, integer=self.integer)
        env = self.position_close(env)
        env = self.shorting(env)


class MarginAccountPipe(AbstractPipe):
    """
    A pipe for margin account environments. The pipe adds the trading
    logics of a margin account to the base market environment.
    """

    def __init__(self,
                 trade_equity_ratio: float = 0.02,
                 excess_margin_ratio_threshold: float = 0.1,
                 min_trade: float = 1,
                 integer: bool = False,
                 buffer_size: int = 10,
                 stack_size: int = None,
                 observation_statistics: Optional[RunningStatistics] = None,
                 reward_statistics: Optional[RunningStatistics] = None,
                 track_statistics: bool = True,
                 verbosity: int = 20) -> None:

        self.trade_equity_ratio = trade_equity_ratio
        self.excess_margin_ratio_threshold = excess_margin_ratio_threshold

        self.min_trade = min_trade
        self.integer = integer

        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.observation_statistics = observation_statistics

        self.reward_statistics = reward_statistics
        self.track_statistics = track_statistics

        self.verbosity = verbosity

        self.margin_account_metadata = MarginAccountMetaDataWrapper
        self.render = ConsoleTearsheetRenderWrapper

        self.initial_margin = InitialMarginActionWrapper
        self.excess_margin = ExcessMarginActionWrapper

        self.action_interpreter = EquityBasedUniformActionInterpreter
        self.clip = ActionClipperWrapper

        self.observation_pipe = ObservationPipe
        self.action_pipe = ActionPipe
        self.reward_pipe = RewardPipe

        return None

    def pipe(self, env):
        """
        Applies a stack of market wrappers successively to an environment.
        Wrappers are addedd successively akin to layers in PyTorch. state of
        wrappers are saved to an attribute of the Pipe class so that they can
        be restored later.

        Args:
        ------
        env (AbstractMarketEnv): the environment to be wrapped.

        Returns:
        ---------
        env (gym.Env): the wrapped environment.
        """

        env = self.margin_account_metadata(env)
        env = self.render(env, verbosity=self.verbosity)

        env = self.initial_margin(env)
        env = self.excess_margin(
            env,
            excess_margin_ratio_threshold=self.excess_margin_ratio_threshold)

        env = self.observation_pipe(
            buffer_size=self.buffer_size,
            stack_size=self.stack_size,
            observation_statistics=self.observation_statistics,
            track_statistics=self.track_statistics).pipe(env)
        env = self.action_pipe(min_trade=self.min_trade,
                               integer=self.integer).pipe(env)
        env = self.reward_pipe(reward_statistics=self.reward_statistics,
                               track_statistics=self.track_statistics).pipe(env)

        env = self.action_interpreter(env, trade_ratio=self.trade_equity_ratio)
        env = self.clip(env)

        return env
