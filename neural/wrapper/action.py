from typing import Type, Dict
from gym.core import Env

import numpy as np
from gym import (ActionWrapper, Env, Wrapper, spaces, Space)

from neural.common.constants import ACCEPTED_ACTION_TYPES, GLOBAL_DATA_TYPE
from neural.common.exceptions import IncompatibleWrapperError
from neural.wrapper.base import metadata



def validate_actions(wrapper: Wrapper, actions: np.ndarray[float] | Dict[str, np.ndarray[float]]) -> None:
    
    """
    Validates the type of the given action against a list of accepted types.

    Args:
        wrapper (Wrapper): The wrapper object calling the validation function.
        actions (numpy.ndarray[float] or dict[str, numpy.ndarray[float]]): The action to validate.

    Returns:
        None

    Raises:
        IncompatibleWrapperError: If the action type is not in the accepted action types.
    """

    valid = False

    if isinstance(actions, dict):
        if all(isinstance(actions[key], np.ndarray) for key in actions):
            valid = True

    elif isinstance(actions, np.ndarray):
        valid = True

    if not valid:
        raise IncompatibleWrapperError(
            f'Wrapper {type(wrapper).__name__} received an action of type {type(actions)}, '
            F'which is not in the accepted action types {ACCEPTED_ACTION_TYPES}.')

    return False



def action(wrapper_class: Type[ActionWrapper]) -> Type[ActionWrapper]:
 
    """
    A decorator that augments an existing Gym wrapper to sanity check the
    actions being passed to it.

    Args
    ----------
    wrapper_class : type[gym.Wrapper]
        The base Gym wrapper class to be augmented. This should be a subclass
        of `gym.Wrapper`.

    Raises
    ------
    TypeError
        If the `wrapper_class` argument is not a subclass of `gym.Wrapper`.

    Returns
    -------
    type[gym.Wrapper]
        A new wrapper class that checks if an action is in the action space
        before calling the step function of the base class.

    Notes
    -----
    The `ActionSpaceCheckerWrapper` class wraps the given `wrapper_class`
    and overrides its `step` method to perform action space validation
    before calling the base class's `step` method.

    Examples
    --------
        >>> from gym import ActionWrapper
        >>> from neural.meta.env.wrapper.action import action
        >>> @action
        ... class CustomActionWrapper(ActionWrapper):
        ...    pass
    """

    if not issubclass(wrapper_class, Wrapper):
        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class ActionSpaceCheckerWrapper(wrapper_class):

        """
        A wrapper that checks if an action is in the action space before calling the step function
        of the base class.

        Args
        ----------
        env : gym.Env
            The environment being wrapped.

        Raises
        ------
        IncompatibleWrapperError
            If the action is not in the action space.

        Attributes
        ----------
        action_space : gym.spaces.Space
            The action space of the environment.

        Methods
        -------
        __init__(self, env: Env, *args, **kwargs) -> None:
            Initializes the `ActionSpaceCheckerWrapper` instance.

        _validate_actions(self, actions: np.ndarray[float] | Dict[str, np.ndarray[float]]) -> None:
            Validates the type of the given actions and checks if they are in the action space of the environment.

        step(self, actions: np.ndarray[float]):
            Checks if the given actions are in the action space of the environment and calls
            the `step` function of the base class.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:

            """
            Initializes the `ActionSpaceCheckerWrapper` instance.

            Args
            ----------
            env : gym.Env
                The environment being wrapped.
            *args : tuple
                Optional arguments to pass to the wrapper.
            **kwargs : dict
                Optional keyword arguments to pass to the wrapper.
            """

            super().__init__(env, *args, **kwargs)

            if not hasattr(self, 'action_space') or not isinstance(self.action_space, Space):

                raise IncompatibleWrapperError(
                    f"Applying {action} decorator to{wrapper_class.__name__} "
                    "requires a non None action space of type {Space} to be defined first.")

            self._validate_actions(self.action_space.sample())

            return None

        def _validate_actions(
            self,
            actions: np.ndarray[float] | Dict[str, np.ndarray[float]]
            ) -> None:

            """
            Validates the type of the given actions and checks if they are in
            the action space of the environment.

            Args
            ----------
            actions : numpy.ndarray[float] or dict[str, numpy.ndarray[float]]
                The actions to validate.

            Raises
            ------
            IncompatibleWrapperError
                If the action is not in the action space of the environment.
            """

            validate_actions(self, actions)
            if not self.action_space.contains(actions):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} received an action of type {type(actions)}, '
                    'which is not in the expected action space {self.action_space}.')

            return None

        def step(self, actions: np.ndarray[float]):

            """
            Checks if the given actions are in the action space of the
            environment and calls the `step` function of the base class.

            Args
            ----------
            actions : numpy.ndarray[float]
                The actions to take.

            Returns
            -------
            Tuple
                The result of calling the `step` function of the base class.
            """

            self._validate_actions(actions)
            return super().step(actions)

    return ActionSpaceCheckerWrapper



@metadata
@action
class MinTradeSizeActionWrapper(ActionWrapper):
    
    """
    A wrapper that limits the minimum trade size for all actions in the environment. 
    If the absolute value of any action is below min_trade, it will be replaced with 0.
    Actions received are notional (USD) asset values.

    Args:
        env: The environment to wrap.
        min_trade: The minimum trade size allowed in the environment. Default is 1.

    Attributes:
        min_trade (float): The minimum trade size allowed in the environment.
        n_symbols (int): The number of symbols in the environment.
        action_space (gym.spaces.Box): The action space of the environment.

    Methods:
        __init__(self, env: Env, min_trade: float = 1) -> None:
            Initializes the MinTradeSizeActionWrapper.

        action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
            Modify the actions to enforce the minimum trade size.
                 
    Example:
        >>> from neural.meta.env.base import TrainMarketEnv
        >>> env = TrainMarketEnv(...)
        >>> wrapped_env = MinTradeSizeActionWrapper(env, min_trade=2)
    """

    def __init__(self, env: Env, min_trade = 1) -> None:

        """
        Initialize the MinTradeSizeActionWrapper.

        Args:
            env: The environment to wrap.
            min_trade: The minimum trade size allowed in the environment. Default is 1.
        """

        super().__init__(env)

        assert min_trade >= 0, 'min_trade must be greater than or equal to 0'

        self.min_trade = min_trade
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = (
            spaces.Box(-np.inf, np.inf, shape=(self.n_symbols,), dtype= GLOBAL_DATA_TYPE))

        return None


    def action(
        self, 
        actions: np.ndarray[float]) -> np.ndarray[float]:
        
        """
        Modify the actions to enforce the minimum trade size.

        Args:
            actions: The original actions.

        Returns:
            The modified actions.
        """

        for asset, action in actions:
            if abs(action) < self.min_trade:
                actions[asset] = 0
    
        return actions.astype(GLOBAL_DATA_TYPE)



@action
@metadata
class IntegerAssetQuantityActionWrapper(ActionWrapper):

    """
    A wrapper for OpenAI Gym trading environments that modifies the 
    agent's actions to ensure they correspond to an integer
    number of shares for each asset.

    This class should be used with caution, as the modification of the 
    agent's actions to enforce integer quantities may not
    be valid in some trading environments due to price slippage. Ensure other 
    action wrappers applied before this would not modify
    the actions in a way that asset quantities are not integer anymore.

    Args:
    -----------
    env : gym.Env
        The trading environment to be wrapped.
    integer : bool, optional
        A flag that indicates whether to enforce integer asset quantities or not. 
        Defaults to True.

    Attributes:
    -----------
    env : gym.Env
        The trading environment to be wrapped.
    integer : bool
        A flag that indicates whether to enforce integer asset quantities or not.
    asset_prices : ndarray or None
        An array containing the current prices of each asset in the environment, 
        or None if the prices have not been set yet.

    Example:
    --------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> env = TrainMarketEnv(...)
    >>> wrapped_env = IntegerAssetQuantityActionWrapper(env, integer=True)
    """

    def __init__(self, env: Env, integer: bool=True) -> None:
        
        """
        Initializes a new instance of the IntegerAssetQuantityActionWrapper class.

        Args:
        -----------
        env : gym.Env
            The trading environment to be wrapped.
        integer : bool, optional
            A flag that indicates whether to enforce integer asset quantities or not. 
            Defaults to True.
        """

        super().__init__(env)
        self.integer = integer
        self.asset_prices = None
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = (
            spaces.Box(-np.inf, np.inf, shape=(self.n_symbols,), dtype=GLOBAL_DATA_TYPE))
        return None


    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        """
        Modifies the agent's actions to ensure they correspond to an integer number of 
        shares for each asset.

        Args:
        -----------
        actions : ndarray
            An array containing the agent's original actions.

        Returns:
        --------
        ndarray
            An array containing the modified actions, where each asset quantity 
            is an integer multiple of its price.
        """

        if self.integer:

            asset_prices = self.market_metadata_wrapper.asset_prices

            for asset, action in enumerate(actions):

                action = (
                    action // asset_prices[asset]) * asset_prices[asset]
                actions[asset] = action

        return actions.astype(GLOBAL_DATA_TYPE)
    


class LiabilitySizingActionWrapper(ActionWrapper):
    """
    This action wrapper curbs the liabilities taken when it reaches a certain threshold. This threshold
    can have a cushion around the maintenance margin to avoid a margin call. It automatically ignores
    actions that lead to taking more liabilities when the threshold is reached. This included borrowing
    more cash or shorting more assets. It is recommended to apply this wrapper after the position sizing
    wrappers to mechanically limit the liabilities taken by the agent. It is still possible that maintenance 
    margin can be violated due price change. In such scenarios it is beneficial to penalize the agent
    for violating the set threshold.
    """
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape= (self.n_symbols,), dtype= GLOBAL_DATA_TYPE)

@action
@metadata
class NetWorthRelativeMaximumShortSizing(ActionWrapper):

    """
    A wrapper for OpenAI Gym trading environments that modifies the agent's actions
    to ensure that the maximum short amount is sized relative to net worth, ensuring
    that the short to net worth ratio is not violated. Takes notional asset value
    to buy/sell as actions. It must be applied before a position sizing wrapper and
    should not be combined with wrappers that increase the sell amount.

    Attributes:
    -----------
    env : gym.Env
        The trading environment to be wrapped.
    short_ratio : float
        The maximum allowed short position as a percentage of the net worth. A value
        of 0 means no shorting, while a value of 0.2 means the maximum short position
        can be 20% of the net worth.

    Methods:
    --------
    __init__(self, env: Env, short_ratio: float = 0.2) -> None:
        Initializes the NetWorthRelativeMaximumShortSizing class with the given environment and short_ratio.
        
    _set_short_budget(self) -> None:
        Sets the short budget based on the net worth and short_ratio.
    
    action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        Processes the given actions without applying the effects, and modifies actions 
        that would lead to short sizing limit violations.

    Example:
    --------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> env = TrainMarketEnv()
    >>> wrapped_env = NetWorthRelativeMaximumShortSizing(env, short_ratio=0.2)
    """

    def __init__(self, env: Env, short_ratio: float = 0.2) -> None:

        """
        Initializes a new instance of the NetWorthRelativeMaximumShortSizing class.

        Args:
        -----------
        env : gym.Env
            The trading environment to be wrapped.
        short_ratio : float, optional
            The maximum allowed short position as a percentage of the net worth. A value
            of 0 means no shorting, while a value of 0.2 means the maximum short position
            can be 20% of the net worth. Default is 0.2.
        """

        super().__init__(env)
        
        assert short_ratio >= 0, "short_ratio must be non-negative"

        self.short_ratio = short_ratio
        self.short_budget = None
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape= (self.n_symbols,), dtype= GLOBAL_DATA_TYPE)

        return None
    

    def _set_short_budget(self) -> None:
        """
        Sets the short budget based on the net worth and short_ratio.
        """

        max_short_size = self.short_ratio * max(self.market_metadata_wrapper.net_worth, 0)
        self.short_budget = max(max_short_size - abs(self.market_metadata_wrapper.shorts), 0)

        return None
    

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        """
        Processes the given actions without applying the effects, and modifies actions 
        that would lead to short sizing limit violations.

        Args:
        -----------
        actions : np.ndarray
            The actions to process.

        Returns:
        --------
        np.ndarray
            The modified actions respecting the short sizing limits.
        """

        positions = self.market_metadata_wrapper.positions

        self._set_short_budget()

        for asset, action in enumerate(actions):
            
            if action > 0:
                continue

            # if short budget is zero and there is no owned assets ignore action
            if self.short_budget == 0 and positions[asset] <= 0:
                actions[asset] = 0
                continue
            
            sell = abs(action)

            # if sell amount exceeds current asset portfolio value shorting occurs
            new_short = abs(min(positions[asset] - sell, 0)) if positions[asset] >=0 else sell

            # modify sell amount if short excess over budget exists.          
            excess = max(new_short - self.short_budget, 0)
            sell -= excess
            new_short -= excess
            
            self.short_budget -= new_short
            actions[asset] = -sell
           
        return actions.astype(GLOBAL_DATA_TYPE)



@action
@metadata
class FixedMarginActionWrapper(ActionWrapper):

    """

    A wrapper for OpenAI Gym trading environments that limits the maximum margin amount
    relative to net worth. Margin trading allows buying more than available cash using leverage.
    The initial_margin parameter specifies the fraction of the trade value that must be
    covered by available cash (i.e., not on margin). For example, an initial margin of 0.1 means
    that only 10% of the trade value needs to be present in the account as cash, thus allowing
    up to 10 times the cash value in assets to be purchased on margin.

    Args:
    -----------
    initial_margin (float): The initial margin required for each trade. Must be a float in (0, 1].

    Methods:
    --------
    __init__(self, env: Env, initial_margin: float = 1) -> None:
        Constructor for the FixedMarginActionWrapper class.
    action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        Processes and modifies the given actions to respect the margin requirements.

    Example:
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> env = TrainMarketEnv()
    >>> wrapped_env = FixedMarginActionWrapper(env, initial_margin=0.2)
    """


    def __init__(
        self, 
        env: Env, 
        initial_margin: float = 1, 
        leverage = 1,
        maintenance_margin: float = 0.25
        ) -> None:
        """
        Initializes a new instance of the FixedMarginActionWrapper class.

        Args:
        -----------
        env : Env
            The trading environment to be wrapped.
        initial_margin : float, optional
            The initial margin required for each trade. Must be a float in (0, 1]. Default is 1.

        Raises:
        -------
        AssertionError:
            If initial_margin is not a float in (0, 1].

        """

        super().__init__(env)

        if not 0 < initial_margin <= 1:
            raise AssertionError("Initial margin must be a float in (0, 1].")   
        
        if not 1 <= leverage:
            raise AssertionError("Leverage must be a float greater than 1.")

        self.initial_margin = initial_margin
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape= (self.n_symbols,), dtype= GLOBAL_DATA_TYPE)

        return None


    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        """
        Processes the given actions without applying the effects, and modifies actions
        that would lead to margin limit violations.

        Args:
        -----------
        actions : ndarray
            An array containing the agent's original actions.

        Returns:
        --------
        ndarray
            An array containing the modified actions, where each buy action is limited
            by the available cash and the initial margin.

        Comments:
        ---------
        - Margin requires available cash. No cash means no margin.
        - Sell actions are ignored due to having no effect on margin.

        """



        cash = self.market_metadata_wrapper.cash
        net_worth = self.market_metadata_wrapper.net_worth

        for asset, action in enumerate(actions):
            
            if action <= 0:
                continue
            
            leverage = 1/self.initial_margin
            buy = min(action, leverage * cash)
            cash -= buy
            actions[asset] = buy

        return actions



@action
@metadata
class NetWorthRelativeUniformPositionSizing(ActionWrapper):


    """An action wrapper that ensures positions taken at each step is a maximum fixed percentage
    of net worth. It maps actions in the range (-1, 1) to buy/sell/hold using fixed zones for each
    action type. The trade_ratio parameter controls the maximum percentage of net worth that 
    can be traded at each step. The hold_threshold parameter defines the threshold
    for holding the current positions. The action in the range (-threshold, threshold)
    is parsed as hold. The action outside this range is linearly projected to (0, max_trade).
    """

    def __init__(self, env: Env, trade_ratio = 0.02, hold_threshold = 0.15):

        """
        Initializes a new instance of the action wrapper with the given environment, 
        trade ratio, and hold threshold.

        Args:
            env (Env): The environment to wrap.
            trade_ratio (float, optional): The maximum percentage of net worth that can be 
            traded at each step. Defaults to 0.02.
            hold_threshold (float, optional): The threshold for holding the current positions. 
            Defaults to 0.15.

        Attributes:
            trade_ratio (float): The maximum percentage of net worth that can be traded at each step.
            hold_threshold (float): The threshold for holding the current positions.
            _max_trade_per_asset (float): The maximum trade that can be made for each asset. 
            Initialized to None.
            action_space (Box): The action space of the wrapped environment.

        Returns:
            None.
        """

        super().__init__(env)

        assert 0 <= trade_ratio <= 1, "Trade ratio must be a float in [0, 1]."
        assert 0 < hold_threshold < 1, "Hold threshold must be a float in (0, 1)."

        self.trade_ratio = trade_ratio
        self.hold_threshold = hold_threshold
        self._max_trade_per_asset = None
        self.n_symbols = self.market_metadata_wrapper.n_symbols

        self.action_space = spaces.Box(
            low = -1, high = 1, shape = (self.n_symbols,), dtype= GLOBAL_DATA_TYPE)
        
        return None


    def _set_max_trade_per_asset(self, trade_ratio: float) -> None:

        """
        Sets the value for the maximum trade that can be made for each asset based on the 
        given trade ratio.

        Args:
            trade_ratio(float): The maximum percentage of net worth that can be traded at each step.

        Returns:
            float: The maximum trade that can be made for each asset.

        Notes:
            The recommended value for initial_cash is >= n_symbols/trade_ratio. 
            In Alpaca API trades with notional value below $1 is not allowed thus if above
            guideline is not followed it is possible that (0, 1) overlaps (0, max_trade)
            to an extent that most, and sometimes all actions end up being no executed.
        """

        self._max_trade_per_asset = (
            trade_ratio * self.market_metadata_wrapper.net_worth)/self.n_symbols

        return None


    def parse_action(self, action: float) -> float:

        """
        Parses the given action value as buy, sell, or hold based on the hold threshold 
        and maximum trade per asset.

        Args:
            action (float): The action value to parse.

        Returns:
            float: The parsed action value.

        Notes:
            Actions within the range (-hold_threshold, hold_threshold) are parsed as hold. 
            Actions outside this range are linearly projected to the range (0, max_trade_per_asset)
        """

        fraction = (
            abs(action) - self.hold_threshold)/(1- self.hold_threshold)

        parsed_action =  (
            fraction * self._max_trade_per_asset * np.sign(action) 
            if fraction > 0 else 0)
        
        return parsed_action

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        """
        Limits the maximum percentage of net worth that can be traded at each step and
        maps actions to buy, sell, or hold positions.

        Args:
            actions (np.ndarray[float]): The actions to perform.

        Returns:
            np.ndarray[float]: The parsed actions.

        Notes:
            This method sets the maximum trade per asset based on the trade ratio and
            parses each action value as buy, sell, or hold using the parse_action method.
        """

        self._set_max_trade_per_asset(self.trade_ratio)

        for asset, action in actions:
            actions[asset] = self.parse_action(action)
        
        return actions


@metadata
class DirectionalTradeActionWrapper(ActionWrapper):

    """
    A wrapper that enforces directional trading by zeroing either positive 
    action values (no long) or negative values (no short). Serves as an upstream action wrapper that
    modifies the actions for downstream trade sizing
    wrappers.

    Args
    ----------
    env : gym.Env
        The environment to wrap.
    long : bool, optional
        If True, only long trades are allowed. If False, only short trades are allowed. 
        Default is True.

    Attributes
    ----------
    long : bool
        If True, only long trades are allowed. If False, only short trades are allowed.
    n_symbols : int
        The number of symbols in the environment.
    action_space : gym.spaces.Box
        The action space of the environment.

    Methods
    -------
    __init__(self, env: Env, long: bool = True) -> None
        Initializes the DirectionalTradeActionWrapper.
    action(self, actions: np.ndarray[float]) -> np.ndarray[float]
        Modifies the actions to enforce directional trading.

    """

    def __init__(self, env: Env, long: bool = True) -> None:

        """
        Initializes the `DirectionalTradeActionWrapper` instance.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        long : bool, optional
            If True, only long trades are allowed. If False, only short 
            trades are allowed. Default is True.
        """

        super().__init__(env)

        self.long = long
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = spaces.Box(-np.inf,
                                       np.inf, shape=(self.n_symbols,))

        return None

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        """
        Modifies the actions to enforce directional trading.

        Parameters
        ----------
        actions : np.ndarray[float]
            The array of actions to enforce directional trading.

        Returns
        -------
        np.ndarray[float]
            The modified array of actions.
        """
        
        if self.long:
            actions[actions < 0] = 0
        else:
            actions[actions > 0] = 0

        return actions
    

@action
@metadata
class ActionClipperWrapper(ActionWrapper):


    """
    A wrapper that clips actions to an expected range for downstream position sizing wrappers.
    This wrapper is applied upstream and enforces that the action values are within a certain range
    before being passed to the downstream wrapper. The action values received are usually the
    immediate output of the model.

    Args:
        env (gym.Env): The environment to wrap.
        low (float, optional): The minimum value for the actions. Defaults to -1.
        high (float, optional): The maximum value for the actions. Defaults to 1.

    Attributes:
        low (float): The minimum value for the actions.
        high (float): The maximum value for the actions.
        n_symbols (int): The number of symbols in the environment.
        action_space (gym.spaces.Box): The action space of the environment.

    Methods:
        __init__(self, env: Env, low: float=-1, high: float=1) -> None:
            Initializes the ActionClipperWrapper instance.
        action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
            Clips the actions to be within the given low and high values.

    Example:
        >>> from neural.meta.env.base import TrainMarketEnv
        >>> env = TrainMarketEnv(...)
        >>> wrapped_env = ActionClipperWrapper(env, low=-0.5, high=0.5)
    """

    def __init__(self, env: Env, low: float=-1, high: float = 1) -> None:

        super().__init__(env)

        self.low = low
        self.high = high
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = (
            spaces.Box(self.low, self.high, shape= (self.n_symbols,), dtype= GLOBAL_DATA_TYPE))

        return None


    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        """
        Clips the actions to be within the given low and high values.

        Args:
            actions (np.ndarray): The array of actions to clip.

        Returns:
            np.ndarray: The clipped array of actions.
        """

        actions = np.clip(actions, self.low, self.high)

        return actions.astype(GLOBAL_DATA_TYPE)
