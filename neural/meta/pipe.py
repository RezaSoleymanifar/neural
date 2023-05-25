"""
pipes.py

This module defines pipes for market environments.
"""
from abc import abstractmethod, ABC
from typing import Callable, Optional

from gym import Env
from neural.wrapper.base import (MarginAccountMetaDataWrapper,
                                 ConsoleTearsheetRenderWrapper)

from neural.wrapper.action import (
    MinTradeSizeActionWrapper, IntegerAssetQuantityActionWrapper,
    PositionCloseActionWrapper, InitialMarginActionWrapper,
    ExcessMarginActionWrapper, ShortingActionWrapper,
    EquityBasedFixedUniformActionParser, ActionClipperWrapper)

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

    Methods:
    --------
        pipe(env):
            Applies a stack of market wrappers successively to an
            environment.
    Notes:
    -----
        Pipes can be combined to create more complex pipes. For example,
        you can save and reuse a predefined set of wrappers as a pipe
        for convenience.
    """

    @abstractmethod
    def pipe(self, env: Env) -> Env:
        """
        Abstract method for piping an environment. Wrappers
        are added successively akin to layers in PyTorch.
        By applying pipe, the environment is wrapped in a stack
        of wrappers.
        """

        raise NotImplementedError


class RewardPipe(AbstractPipe):
    """
    This pipe adds reward generation, interest on debt, and
    normalization to the base market environment. The pipe adds the
    following functionality to the base environment:
        - Reward generation:
            This will be the change in equity of the account.
        - Interest on debt
            Reduces reward by amount of interest on debt. Computed at 
            end of day.
        - Reward normalization:
            Ensures that the reward distribution has zero mean and unit.
    
    Attributes:
    -----------
        interest_rate (float):
            interest rate on debt. Defaults to 0.08.
        epsilon (float):
            small number to avoid division by zero. Defaults to 1e-8.
        clip_threshold (float):
            threshold for clipping rewards. Defaults to 10.
        reward_generator (RewardGeneratorWrapper):
            reward generator wrapper. Set to RewardGeneratorWrapper at
            construction.
        interest (LiabilityInterstRewardWrapper):
            interest on debt wrapper. Set to
            LiabilityInterstRewardWrapper at construction.
        reward_normalizer (RewardNormalizerWrapper):
            reward normalizer wrapper. Set to RewardNormalizerWrapper at
            construction.

    Methods:
    --------
        pipe(env):
            Applies a stack of market wrappers successively to an
            environment.
    """

    def __init__(
        self,
        interest_rate: float = 0.08,
    ) -> None:
        """
        Initializes the reward pipe.

        Args:
        -----
            interest_rate (float):
                interest rate on debt. Defaults to 0.08.
            epsilon (float):
                small number to avoid division by zero. Defaults to 1e-8.
            clip_threshold (float):
                threshold for clipping rewards. Defaults to 10.
        """

        self.interest_rate = interest_rate

        self.reward_generator = RewardGeneratorWrapper
        self.interest = LiabilityInterstRewardWrapper
        self.reward_normalizer = RewardNormalizerWrapper

        return None

    def pipe(self, env: Env) -> Env:
        """
        A method for piping an environment. Applies a stack of market wrappers
        successively to an environment:
            1. Reward generation:
                This will be the change in equity of the account.
            2. Interest on debt:
                Reduces reward by amount of interest on debt (asset/cash).
                Computed at end of day.
            3. Reward normalization:
                Ensures that the reward distribution has zero mean and unit
                variance.
            
        Args:
        -----
            env (Env):
                environment to be wrapped.
            
        Returns:
        --------
            env (Env):
                wrapped environment.
        """
        env = self.reward_generator(env)
        env = self.interest(env, interest_rate=self.interest_rate)
        env = self.reward_normalizer(env)

        return env


class ObservationPipe(AbstractPipe):
    """
    Observation pipe for market environments. Provides basic functionality for
    manipulating observations. The pipe adds the following functionality to the
    base environment:
        - Observation flattening:
            If numpy array, flattens the observation to a 1D array. If dict,
            flattens the observation to a 1D array for each key. then joins the
            arrays into a single 1D array.
        - Observation buffering:
            Buffers the last n observations. If numpy array, buffers the last n
            observations in a deque. If dict, buffers the last n observations
            in a deque for each key.
        - Observation stacking:
            Stacks the last n observations. If numpy array, stacks the last n
            observations along axis = 0. If dict, stacks the last n
            observations along axis = 0 for each key.
        - Observation normalization:
            Ensures that the observation distribution has zero mean and unit
            variance. If numpy array, normalizes the observation using a
            running mean and standard deviation. If dict, normalizes the
            observation using a running mean and standard deviation for each
            key.
    
    Attributes:
    -----------
        buffer_size (int):
            size of the buffer for buffering observations. Set to 10 at
            construction.
        stack_size (int):
            size of the stack for stacking observations. Set to None at
            construction. If None, the stack size will be set to the buffer
            size.
        observation_statistics (RunningStatistics):
            statistics of the observation distribution. Set to None at
            construction. If track_statistics is True, the statistics will be
            synchronized with the statistics of the observation normalizer
            wrapper. This will be reused with the wrapper when the pipe is
            saved and loaded.
        track_statistics (bool):
            whether to track and update the observation statistics during
            training. If False, the statistics will not be tracked and updated
            during training. If yes statistics are tracked as pipe attribute
            and can be saved and loaded with the pipe object.
        flatten (FlattenObservationWrapper):
            observation flattening wrapper. Set to FlattenObservationWrapper at
            construction.
        buffer (BufferObservationWrapper):
            observation buffering wrapper. Set to BufferObservationWrapper at
            construction.
        stack (StackObservationWrapper):
            observation stacking wrapper. Set to StackObservationWrapper at
            construction.
        normalizer (NormalizeObservationWrapper):
            observation normalizer wrapper. Set to NormalizeObservationWrapper
            at construction.
    
    Methods:
    --------
        pipe(env):
            Applies a stack of market wrappers successively to an environment.
    """

    def __init__(
        self,
        buffer_size: int = 10,
        stack_size: int = None,
    ) -> None:
        """
        Initializes the observation pipe.

        Args:
        ------
        buffer_size (int):
            size of the buffer for buffering observations. Set to 10 at
            construction.
        stack_size (int):
            size of the stack for stacking observations. Set to None at 
            construction. If None, the stack size will be set to the
            buffer size.
        observation_statistics (RunningStatistics):
            statistics of the observation distribution. Set to None at
            construction.
        """
        self.buffer_size = buffer_size
        self.stack_size = stack_size
        self.observation_statistics = None

        self.flatten = FlattenToNUmpyObservationWrapper
        self.buffer = ObservationBufferWrapper
        self.stacker = ObservationStackerWrapper
        self.normalizer = ObservationNormalizerWrapper

        return None

    def pipe(self, env: Env) -> Env:
        """
        Applies the following functionalities to the base environment:

            1. Observation flattening:
                If numpy array, flattens the observation to a 1D array. If
                dict, flattens the observation to a 1D array for each key.
                then joins the arrays into a single 1D array.

            2. Observation buffering:
                Buffers the last n observations. If numpy array, buffers the
                last n observations in a deque. If dict, buffers the last n
                observations in a deque for each key.

            3. Observation stacking:
                Stacks the last n observations. If numpy array, stacks the last
                n observations along axis = 0. If dict, stacks the last n
                observations along axis = 0 for each key.

            4. Observation normalization:
                Ensures that the observation distribution has zero mean and
                unit variance. If numpy array, normalizes the observation
                using a running mean and standard deviation. If dict,
                normalizes the observation using a running mean and standard
                deviation for each key.

        Args:
        -----
            env (gym.Env):
                environment to be wrapped
        
        Returns:
        --------
            env (gym.Env):
                wrapped environment
        """
        env = self.flatten(env)
        env = self.buffer(env, buffer_size=self.buffer_size)
        env = self.stacker(env, stack_size=self.stack_size)

        env = self.normalizer(
            env, observation_statistics=self.observation_statistics)

        return env


class ActionPipe(AbstractPipe):
    """
    Action pipe for market environments. The pipe adds the following
    functionality to the base environment:
        - Minimum trade size:
            Sets actions bellow a minimum trade size to zero.
        - Integer asset quantity:
            Modifies actions so that they map to integer asset quantities.
        - Position close:
            Flipping sides long/short happens with closing the position first.
        - Shorting:
            Shorting is only possible with integer asset quantities.

    Minimum trade size ensures that the notional value of a trade is greater
    than a minimum threshold; if asset is held. Integer asset quantity ensures
    that the number of assets traded is an integer. Position close ensures that
    the agent can close positions, before flipping the sign of the quantity.
    Shorting ensures that the agent short actions map to integer asset
    quantities.

    Attributes:
    -----------
        min_trade (float):
            minimum trade size in terms of notional value of base currency. Set
            to 1 at construction.
        integer (bool):
            whether to modify notional value of trades to match integer number
            of assets. Set to False at construction.
        min_trade (MinTradeSizeActionWrapper):
            minimum trade size wrapper. Set to MinTradeSizeActionWrapper at
            construction.
        integer_quantity (IntegerAssetQuantityActionWrapper):   
            integer asset quantity wrapper. Set to
            IntegerAssetQuantityActionWrapper at construction.
        position_close (PositionCloseActionWrapper):
            position close wrapper. Set to PositionCloseActionWrapper at
            construction.
        shorting (ShortingActionWrapper):   
            Allows shorting wrapper. Set to ShortingActionWrapper at
            construction.
    
    Methods:
    --------
        pipe(env):
            Applies a stack of market wrappers successively to an environment.
    """

    def __init__(self, min_trade: float = 1, integer: bool = False) -> None:
        """
        Initializes the action pipe.

        Args:
        ------
        min_trade (float):
            minimum trade size in terms of notional value of base
            currency. Set to 1 at construction.
        integer (bool):
            whether to modify notional value of trades to match integer
            number of assets. Set to False at construction.
        """
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

        return env


class HeadActionPipe(AbstractPipe):
    """
    This pipe is responsible for parsing the immediate actions of the model,
    hence the name head. It is the last pipe applied in the action pipe stack
    (first pipe to receive actions). The pipe adds the following functionality
    to the base environment:
        - Action parsing:
            maps actions from neural network output to notional value of trade
            in base currency.
        - Action mapping:
            maps action of discrete values neural network output to expected
            range of continuous actions.
        - Action clipping:
            clips actions to expected range of continuous actions. Do not use
            with discrete actions.
    
    After parsing the actions should correspond the notional value of trade in
    base currency (e.g. 100$ for USDT-BTC pair). In general it is assumed that
    a fixed percentage of the equity (trade equity ratio) is traded at each
    interval. The percentage can be fixed at construction. This trading budget
    so to speak can be distributed uniformly, or non-uniformly across the
    assets. Models that produce discrete actions are only compatible with
    uniform distribution of the trading budget. The trading equity ratio can
    also be determined by the model, in this case the trading budget is
    determined by the model and the trade equity ratio at construction is
    ignored. Types of actions expected from the model are:
        - Uniform fixed ratio: 
            one neuron for each asset each with value (-1, 1). This will be
            used to infer both side and value of trade. Interpretting this
            signal is left to an action parser. This can be achieved by
            applying tanh() to the output of the model, or
            simply just clipping the actions of the model to this range. For
            discrete models, the output of the model is mapped to (-1, 1) using
            an action mapper.
        - Uniform variable ratio:
            one neuron for each asset each with value (-1, 1). One neuron with
            value in (0,1) indicating the trade equity ratio. This can be
            achieved by applying sigmoid to a corresponding neuron. For
            discrete models, the output of the model for each asset is mapped
            to (-1, 1) using an action mapper. Similarly output of the model
            for the trade equity ratio is mapped to (0, 1) using the same
            action mapper.
        - Non-uniform fixed ratio:
            This is only viable for continuous models. The model should output
            one neuron for each asset each with value (-1, 1), showing trade
            side (buy/sell). Apply tanh() to the relevant output of the model.
            Also it has one neuron for each asset each with value in (0, 1)
            summing to 1, showing the distribution of budget across assets.
            This can be achieved by applying softmax to the corresponding
            outputs of the model. An action interpreter then uses these 2n
            neurons to infer the notional value of trade for each asset. The
            trade equity ratio is fixed at construction.
        - Non-uniform variable ratio:
            This is only viable for continuous models. The model is identical
            to the non-uniform fixed ratio model, except that the trade equity
            ratio is determined by the model. A neuron with value in (0,1) is
            responsible for determining the trade equity ratio. This can be
            achieved by applying sigmoid to a corresponding neuron.
        
    Training the non-uniform/variable ratio models is more difficult than the
    uniform/fixed ratio models. This is because the model is dealing with a
    larger action space primarily. Moreover model can produce actions that lead
    to more frequent trading anomalies. It is recommended to use a tiered
    training approach with restrictions on the degrees of freedom of the model
    and gradually removing them at each tier.

    Attributes:
    -----------
        uniform (bool):
            whether to use uniform distribution of trading budget. Set to True
            at construction.
        fixed (bool):
            whether to use fixed trading equity ratio. Set to True at
            construction.
        discrete (bool):
            whether to use discrete actions. Set to True at construction.
        trade_equity_ratio (float):
            fixed trading equity ratio. Set to 0.1 at construction.
        hold_threshold (float):
            threshold for holding an asset. Set to 0.15 at construction.
        clip (bool):
            whether to clip actions to (low, high). Set to False at
            construction.
        low (float):
            lower bound for clipping actions. Set to -1 at construction.
        high (float):
            upper bound for clipping actions. Set to 1 at construction.
        fixed_uniform (EquityBasedFixedUniformActionParser):
            action parser for fixed uniform ratio models.
        variable_uniform ($$$):
            action parser for variable uniform ratio models.
        fixed_nonuniform ($$$):
            action parser for fixed non-uniform ratio models.
        variable_nonuniform ($$$):
            action parser for variable non-uniform ratio models.
            
    Methods:
    --------
        pipe(env):
            Applies a stack of market wrappers successively to an environment.
    
    Raises:
    -------
        ValueError:
            if discrete and non-uniform distribution of trading budget is
            requested.
    """

    def __init__(self,
                 uniform: bool = True,
                 fixed: bool = True,
                 discrete: bool = False,
                 trade_equity_ratio: float = 0.05,
                 hold_threshold: float = 0.15,
                 clip: bool = False,
                 low: float = -1,
                 high: float = 1) -> None:
        """
        Initializes the head action pipe.

        Arguments:
        ----------
            uniform (bool):
                whether to use uniform distribution of trading budget. Set to
                True at construction.
            fixed (bool):
                whether to use fixed trading equity ratio. Set to True at
                construction.
            discrete (bool):
                whether to use discrete actions. Set to True at construction.
            trade_equity_ratio (float):
                fixed trading equity ratio. Set to 0.05 at construction.
            hold_threshold (float):
                threshold for holding an asset. Set to 0.15 at construction.
            clip (bool):
                whether to clip actions to (low, high). Set to False at
                construction.
            low (float):
                lower bound for clipping actions. Set to -1 at construction.
            high (float):
                upper bound for clipping actions. Set to 1 at construction.
        """

        if uniform and discrete:
            raise ValueError(
                'Discrete models can only be used with uniform trading '
                'budget distribution.')

        self.uniform = uniform
        self.fixed = fixed
        self.discrete = discrete

        self.trade_equity_ratio = trade_equity_ratio
        self.hold_threshold = hold_threshold

        self.clip = clip
        self.low = low
        self.high = high

        self.fixed_uniform_parser = EquityBasedFixedUniformActionParser
        self.variable_uniform_parser = None
        self.fixed_nonuniform_parser = None
        self.variable_nonuniform_parser = None

        self.fixed_uniform_mapper = None
        self.fixed_nonuniform_mapper = None

        self.action_clipper = ActionClipperWrapper

    @property
    def parser(self) -> Callable:
        """
        Parser is the first pipe to receive actions. It is responsible for
        parsing the actions of the model. If model is discrete, it may be 
        preceded by an action mapper wrapper to map discrete actions to
        the expected action space.
        """
        if self.fixed:
            if self.uniform:
                parser = lambda env: self.fixed_uniform_parser(
                    self.trade_equity_ratio, self.hold_threshold, env)
            else:
                parser = self.fixed_nonuniform_parser
        else:
            if self.uniform:
                parser = self.variable_uniform_parser
            else:
                parser = self.variable_nonuniform_parser

        return parser

    @property
    def mapper(self):
        """
        Used with discrete models to map the actions to the expected action
        space.

        Examples:
        ---------
        Model produces k actions for buy orders, k actions for sell orders
        and 1 action for hold. The mapper will map the k buy actions to
        the range (hold_threshold, 1) and k sell actions to the range
        (-1, -hold_threshold). The hold action will be mapped to 0. This 
        is the expected input of the uniform action parser.
        """
        if self.uniform:
            mapper = lambda env: self.fixed_uniform_mapper(env)
        else:
            mapper = lambda env: self.fixed_nonuniform_mapper(env)

        return mapper

    def pipe(self, env):
        """
        Applies the head pipe to the environment. Infers action parser from

        """
        env = self.parser(env)
        if self.discrete:
            env = self.mapper(env)
        if self.clip:
            env = self.action_clipper(self.low, self.high, env)

        return env


class RenderPipe(AbstractPipe):
    """
    A pipe to render the environment. This usually is the last pipe in the
    stack of pipes. It is responsible for rendering the environment to the
    console, GUI or a file.
    """

    def __init__(self, verbosity=10) -> None:
        self.verbosity = verbosity
        self.render = ConsoleTearsheetRenderWrapper

    def pipe(self, env):
        """ 
        Adds the following functionality to the environment:
        - prints the current state of the environment to the console
        includes:
            - progress
            - return
            - sharpe ratio
            - profit 
            - equity
            - cash
            - portfolio value
            - longs
            - shorts
        """
        env = self.render(env)
        return env


class BasePipe(RewardPipe, ObservationPipe, ActionPipe, HeadActionPipe,
               RenderPipe):
    """
    A basic pipe to provide fundamental trading and training functionalities to
    the environment.
    """

    def __init__(self,
                 trade_equity_ratio: float = 0.05,
                 verbosity: int = 0,
                 interest_rate: float = 0.08,
                 buffer_size: int = 1,
                 stack_size: int = 1,
                 min_trade: float = 0.01,
                 integer: bool = False,
                 uniform: bool = True,
                 fixed: bool = True,
                 discrete: bool = False,
                 hold_threshold: float = 0.15,
                 clip: bool = False,
                 low: float = -1,
                 high: float = 1) -> None:

        RewardPipe.__init__(self, interest_rate=interest_rate)
        ObservationPipe.__init__(self,
                                 buffer_size=buffer_size,
                                 stack_size=stack_size)
        ActionPipe.__init__(self, min_trade=min_trade, integer=integer)
        HeadActionPipe.__init__(self,
                                uniform=uniform,
                                fixed=fixed,
                                discrete=discrete,
                                trade_equity_ratio=trade_equity_ratio,
                                hold_threshold=hold_threshold,
                                clip=clip,
                                low=low,
                                high=high)
        RenderPipe.__init__(self, verbosity=verbosity)

        return None

    def pipe(self, env):
        """
        Adds the functionality of the following pipes to the environment:
        - RewardPipe
        - ObservationPipe
        - ActionPipe
        - HeadActionPipe
        - RenderPipe
        """
        env = RewardPipe.pipe(self, env)
        env = ObservationPipe.pipe(self, env)
        env = ActionPipe.pipe(self, env)
        env = HeadActionPipe.pipe(self, env)
        env = RenderPipe.pipe(self, env)

        return env


class MarginAccountPipe(BasePipe):
    """
    A pipe to simulate a margin account environment. The pipe adds the
    trading logics of a margin account to the base market environment.

    It offers following functionalities:
        - Margin account metadata
        - Console tearsheet render
        - Initial margin
        - Excess margin

    Also adds the base pipe functionalities:
        - RewardPipe
        - ObservationPipe
        - ActionPipe
        - HeadActionPipe
        - RenderPipe

    and following action interpreter:
        - Equity based uniform action interpreter

    Attributes:
    -----------
        excess_margin_ratio_threshold (float):
            threshold for excess margin ratio. Set to 0.1 at
            construction.
        trade_equity_ratio (float):
            ratio of equity to be traded. Set to 0.02 at construction.
        min_trade (float):
            minimum trade size in terms of notional value of base
            currency. Set to 1 at construction.
        integer (bool):
            whether to modify notional value of trades to match integer
            number of assets. Set to False at construction.
        buffer_size (int):
            size of the buffer for buffering observations. Set to 10 at
            construction.
        stack_size (int):
            size of the stack for stacking observations. Set to None at
            construction. If None, the stack size will be set to the    
            buffer size.
        observation_statistics (RunningStatistics):
            statistics of the observation distribution. Set to None at
            construction. If track_statistics is True, the statistics
            will be synchronized with the statistics of the observation
            normalizer wrapper. This will be reused with the wrapper
            when the pipe is saved and loaded.
        track_statistics (bool):
            whether to track and update the observation statistics
            during training. If False, the statistics will not be   
            tracked and updated during training. If yes statistics are
            tracked as pipe attribute and can be saved and loaded with
            the pipe object.
        verbosity (int):
            verbosity level of the console tearsheet render. Set to 20
            at construction.

        margin_account_metadata (MarginAccountMetaDataWrapper):
            margin account metadata wrapper. Set to
            MarginAccountMetaDataWrapper at construction.   
        render (ConsoleTearsheetRenderWrapper):
            console tearsheet render wrapper. Set to
            ConsoleTearsheetRenderWrapper at construction.
        initial_margin (InitialMarginActionWrapper):
            initial margin wrapper. Set to InitialMarginActionWrapper
            at construction.
        excess_margin (ExcessMarginActionWrapper):
            excess margin wrapper. Set to ExcessMarginActionWrapper at
            construction.
        observation_pipe (ObservationPipe):
            observation pipe. Set to ObservationPipe at construction.
        action_pipe (ActionPipe):
            action pipe. Set to ActionPipe at construction.
        reward_pipe (RewardPipe):
            reward pipe. Set to RewardPipe at construction.
        action_interpreter (EquityBasedUniformActionInterpreter):
            equity based uniform action interpreter wrapper. Set to
            EquityBasedUniformActionInterpreter at construction.
        
    Methods:
    --------
        pipe(env):
            Applies a stack of market wrappers successively to an
            environment.
    """

    def __init__(self,
                 excess_margin_ratio_threshold: float = 0.1,
                 trade_equity_ratio: float = 0.02,
                 verbosity: int = 0,
                 interest_rate: float = 0.08,
                 buffer_size: int = 1,
                 stack_size: int = 1,
                 min_trade: float = 0.01,
                 integer: bool = False,
                 uniform: bool = True,
                 fixed: bool = True,
                 discrete: bool = False,
                 hold_threshold: float = 0.15,
                 clip: bool = False,
                 low: float = -1,
                 high: float = 1) -> None:

        self.excess_margin_ratio_threshold = excess_margin_ratio_threshold
        BasePipe.__init__(self,
                          verbosity=verbosity,
                          interest_rate=interest_rate,
                          buffer_size=buffer_size,
                          stack_size=stack_size,
                          min_trade=min_trade,
                          integer=integer,
                          uniform=uniform,
                          fixed=fixed,
                          discrete=discrete,
                          hold_threshold=hold_threshold,
                          clip=clip,
                          low=low,
                          high=high)

        self.margin_account_metadata = MarginAccountMetaDataWrapper
        self.initial_margin = InitialMarginActionWrapper
        self.excess_margin = ExcessMarginActionWrapper

        self.base_pipe = BasePipe

        return None

    def pipe(self, env):
        """
        Applies a stack of market wrappers successively to an
        environment.

        Args:
        ------
        env (AbstractMarketEnv): 
            the environment to be wrapped.

        Returns:
        ---------
        env (gym.Env): 
            the wrapped environment.
        """

        env = self.margin_account_metadata(env)
        env = self.initial_margin(env)
        env = self.excess_margin(
            env,
            excess_margin_ratio_threshold=self.excess_margin_ratio_threshold)
        env = self.base_pipe(verbosity=self.verbosity,
                             interest_rate=self.interest_rate,
                             buffer_size=self.buffer_size,
                             stack_size=self.stack_size,
                             min_trade=self.min_trade,
                             integer=self.integer,
                             uniform=self.uniform,
                             fixed=self.fixed,
                             discrete=self.discrete,
                             trade_equity_ratio=self.trade_equity_ratio,
                             hold_threshold=self.hold_threshold,
                             clip=self.clip,
                             low=self.low,
                             high=self.high).pipe(env)

        return env
