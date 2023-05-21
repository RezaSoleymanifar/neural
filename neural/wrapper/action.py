from typing import Type, Dict

import numpy as np
from gym import ActionWrapper, Env, Wrapper, spaces, Space

from neural.common.constants import ACCEPTED_ACTION_TYPES, GLOBAL_DATA_TYPE
from neural.common.exceptions import IncompatibleWrapperError
from neural.wrapper.base import metadata


def validate_actions(
        wrapper: Wrapper,
        actions: np.ndarray[float] | Dict[str, np.ndarray[float]]) -> None:
    """
    Validates the type of the given action against a list of accepted
    types.

    Args:
    ----------
        wrapper (Wrapper): The wrapper object calling the validation
        function. actions (numpy.ndarray[float] or dict[str,
        numpy.ndarray[float]]): The action to validate.

    Returns:
    ----------
        None

    Raises:
    ----------
        IncompatibleWrapperError: If the action type is not in the
        accepted action types.
    
    Notes:
    ----------
        This function is called by the `action` decorator to validate
        the type of the action before calling the `step` function of the
        base class.
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
            F'which is not in the accepted action types {ACCEPTED_ACTION_TYPES}.'
        )
    return False


def action(wrapper_class: Type[ActionWrapper]) -> Type[ActionWrapper]:
    """
    A decorator that augments an existing Gym wrapper to sanity check
    the actions being passed to it.

    Args
    ----------
    wrapper_class : type[gym.Wrapper]
        The base Gym wrapper class to be augmented. This should be a
        subclass of `gym.Wrapper`.

    Raises
    ------
    TypeError
        If the `wrapper_class` argument is not a subclass of
        `gym.Wrapper`.

    Returns
    -------
    type[gym.Wrapper]
        A new wrapper class that checks if an action is in the action
        space before calling the step function of the base class.

    Notes
    -----
    The `ActionSpaceCheckerWrapper` class wraps the given
    `wrapper_class` and overrides its `step` method to perform action
    validation before calling the base class's `step` method.

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
        A wrapper that checks if an action is in the action space before
        calling the step function of the base class.

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

        _validate_actions(self, actions: np.ndarray[float] | Dict[str,
        np.ndarray[float]]) -> None:
            Validates the type of the given actions and checks if they
            are in the action space of the environment.

        step(self, actions: np.ndarray[float]):
            Checks if the given actions are in the action space of the
            environment and calls the `step` function of the base class.
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
            
            Raises
            ------
            IncompatibleWrapperError
                If non None action space of type `gym.spaces.Space` is
                not defined first.
            """

            super().__init__(env, *args, **kwargs)

            if not hasattr(self, 'action_space') or not isinstance(
                    self.action_space, Space):

                raise IncompatibleWrapperError(
                    f'Applying {action} decorator to{wrapper_class.__name__} '
                    f'requires a non None action space of type {Space} '
                    ' to be defined first.')

            self._validate_actions(self.action_space.sample())

            return None

        def _validate_actions(
                self, actions: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> None:
            """
            Validates the type of the given actions and checks if they
            are in the action space of the environment.

            Args
            ----------
            actions : numpy.ndarray[float] or dict[str,
            numpy.ndarray[float]]
                The actions to validate.

            Raises
            ------
            IncompatibleWrapperError
                If the action is not in the action space of the
                environment.
            """

            validate_actions(self, actions)
            if not self.action_space.contains(actions):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} received an action of '
                    f'type {type(actions)}, which is not in the expected '
                    'action space {self.action_space}.')

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
                The result of calling the `step` function of the base
                class.
            """

            self._validate_actions(actions)
            return super().step(actions)

    return ActionSpaceCheckerWrapper


@metadata
@action
class MinTradeSizeActionWrapper(ActionWrapper):
    """
    A wrapper that limits the minimum trade size for all actions in the
    environment. If the absolute value of any action is below min_trade,
    it will be replaced with 0. Actions received are notional base
    currency asset values.

    Args:
    ----------
        env: gym.Env 
            The environment to wrap. 
        min_trade:  float, optional
            The minimum trade size allowed in the environment. Default
            is 1.

    Attributes:
    ----------
        min_trade (float): 
            The minimum trade size allowed in the environment.
        n_symbols (int):
            The number of symbols in the environment.
        action_space (gym.spaces.Box):
            The action space of the environment.

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

    def __init__(self, env: Env, min_trade=1) -> None:
        """
        Initialize the MinTradeSizeActionWrapper.

        Args:
            env: The environment to wrap. min_trade: The minimum trade
            size allowed in the environment. Default is 1.
        """

        super().__init__(env)

        assert min_trade >= 0, 'min_trade must be greater than or equal to 0'

        self.min_trade = min_trade
        self.n_assets = self.market_metadata_wrapper.n_assets
        self.action_space = (spaces.Box(-np.inf,
                                        np.inf,
                                        shape=(self.n_assets, ),
                                        dtype=GLOBAL_DATA_TYPE))

        return None

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        """
        Modify the actions to enforce the minimum trade size.

        Args:
            actions: The original actions.

        Returns:
            The modified actions.
        """

        for asset, action in enumerate(actions):
            if abs(action) < self.min_trade:
                actions[asset] = 0

        return actions.astype(GLOBAL_DATA_TYPE)


@action
@metadata
class IntegerAssetQuantityActionWrapper(ActionWrapper):
    """
    Fractional quantity of shares is natively allowed by the base market
    environment. This wrapper modifies the agent's actions to ensure
    they correspond to an integer quantity of assets, i.e. no fractional
    quantities are allowed. This is useful for trading environments that
    do not allow fractional quantities of assets or modifying actions
    for assets that are inherently non-fractionable even on platforms
    that do allow fractional trading.

    The modification of the agent's actions to enforce integer
    quantities may not be valid in live trading environments due to
    price slippage. Trader object is responsible for handling this
    type of anomalies.

    Args:
    -----------
    env : gym.Env
        The trading environment to be wrapped.
    integer : bool, optional
        A flag that indicates whether to enforce integer asset
        quantities or not. Defaults to True.  

    Attributes:
    -----------
    env : gym.Env
        The trading environment to be wrapped.
    integer : bool
        A flag that indicates whether to enforce integer asset
        quantities or not.
    asset_prices : ndarray or None
        An array containing the current prices of each asset in the
        environment, or None if the prices have not been set yet.
    n_assets : int
        The number of assets in the environment.
    action_space : gym.spaces.Box
        The action space of the environment.

    Example:
    --------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> env = TrainMarketEnv(...)
    >>> wrapped_env = IntegerAssetQuantityActionWrapper(env, integer=True)
    """

    def __init__(self, env: Env, integer: bool = True) -> None:
        """
        Initializes a new instance of the
        IntegerAssetQuantityActionWrapper class.

        Args:
        -----------
        env : gym.Env
            The trading environment to be wrapped.
        integer : bool, optional
            A flag that indicates whether to enforce integer asset
            quantities or not. Defaults to True.
        """

        super().__init__(env)
        self.integer = integer
        self.asset_prices = None
        self.n_assets = self.market_metadata_wrapper.n_assets
        self.action_space = (spaces.Box(-np.inf,
                                        np.inf,
                                        shape=(self.n_assets, ),
                                        dtype=GLOBAL_DATA_TYPE))
        return None

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        """
        Modifies the agent's actions to ensure they correspond to an
        integer number of shares for each asset.

        Args:
        -----------
        actions : ndarray
            An array containing the agent's original actions.

        Returns:
        --------
        ndarray
            An array containing the modified actions, where each asset
            quantity is an integer multiple of its price.
        """

        if self.integer:
            asset_prices = self.market_metadata_wrapper.asset_prices
            for asset, action, price in enumerate(actions, asset_prices):
                action = (action // price) * price
                actions[asset] = action

        return actions.astype(GLOBAL_DATA_TYPE)


@action
class PositionCloseActionWrapper:
    """
    Transition from long to short position and vice versa is a two step
    process. Close the current long/short position, and then open a new
    short/long position. This wrapper modifies the agent's actions to
    ensure that if action exceeds the current position, then the excess
    is ignored and only the current position is closed. For example, in
    order to open short positions, the current open long position must
    be closed first in a separate order, and then a short position can
    be opened.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf,
                                       np.inf,
                                       shape=(self.n_assets, ),
                                       dtype=GLOBAL_DATA_TYPE)

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        """
        Actions are modified to ensure that if action exceeds the
        current position, then the excess is ignored and only the
        current position is closed.

        Args:
        --------
            actions (np.ndarray[float]): The actions to perform.

        Returns:
        --------
            np.ndarray[float]: The modified actions.
        """
        positions = self.market_metadata_wrapper.positions
        for asset, action, position in enumerate(zip(actions, positions)):
            if (position > 0 and action + position < 0
                    or position < 0 and action + position > 0):
                action = -position
            actions[asset] = action
        return actions.astype(GLOBAL_DATA_TYPE)


@action
@metadata
class InitialMarginActionWrapper(ActionWrapper):
    """
    Implements the position opening logic of the margin account. This
    wrapper assumes exclusive all marginable or all nonmarginable
    assets, but not both. This wrapper modifies the agent's actions to
    ensure that if new positions are opened it 1) satisfies intial
    margin constraint, and 2) satisfies maintenance margin constraint
    after opening the new position. In the context of nonmarginable
    assets this wrapper ensures that there is enough cash to open new
    positions. If amount of free liquidity (marginable equity or cash)
    is not enough to open new positions then the portfolio increasing
    actions are set to zero.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.asset_quantities = self.market_metadata_wrapper.quantities
        self.asset_prices = self.market_metadata_wrapper.asset_prices
        self.portfolio_value = self.market_metadata_wrapper.portfolio_value
        self.positions = self.market_metadata_wrapper.positions
        self.assets = self.market_metadata_wrapper.assets
        self.excess_margin = self.market_metadata_wrapper.excess_margin

        self.action_space = spaces.Box(-np.inf,
                                       np.inf,
                                       shape=(self.n_assets, ),
                                       dtype=GLOBAL_DATA_TYPE)

    @property
    def portfolio_increase_action_indices(self, actions):
        """
        Initial margin is checked for all assets that are being
        increased in position value. This property returns the indices
        of all assets that are being increased in position value.
        """
        indices = [
            index for index, quantity, action in enumerate(
                zip(self.quantities, actions))
            if (action < 0 and quantity <= 0 or action > 0 and quantity >= 0)
        ]
        return indices

    def initial_margin_required(self):
        """
        Initial margin is a percentage of notional value of trade that
        needs to be available in form of marginable equity. In the
        context of nonmarginable assets marginable equity is equivalent
        to cash. Aggregates the intial margin required for all assets
        that are being increased in position value. Note that notion of
        initial margin is only applicable to marginable assets. However
        with abuse of terminaology we use the same term for
        nonmarginable assets as well to indicate the amount of cash
        required to increase the position value of nonmarginable assets.
        """
        margin_required = sum(
            self.assets[index].initial_margin(
                short=self.asset_quantities[index] <= 0) * self.positions[index]
            for index in self.portfolio_increase_action_indices)
        return margin_required

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        """
        Checks if initial margin requirement is met for all assets. If
        not then sets all porfolio increasing actions to zero. In the
        context of marginable assets this is a conservative approach to
        ensure that initial margin requirement is firstly satisfied,
        because if excess margin is greater than initial margin
        requirement then by definition marginable equity is greater
        than initial margin requirement. Secondly post transaction
        excess margin is greater than zero, which ensures that no margin
        call is received.

        In the context of nonmarginable assets this rule ensures that
        there is enough cash to increase the position value of
        nonmarginable assets.

        Args:
        ----------
            actions (np.ndarray[float]): The actions to perform.
        
        Returns:
        ----------
            np.ndarray[float]: The modified actions.

        Notes:
        -------
            ExcessMarginActionWrapper must be applied upstream to ensure
            a certain elvel of excess margin is maintained at all times.
            In conjunction with this wrapper it ensures excess margin is
            always large enough that initial margin requirement, and 
            maintenance margin requirement is met at all times. However
            this wrapper operates as a last line of defense to ensure
            that no margin call is received or equivalently cash is 
            not negative at any point in time, for nonmarginable assets.
        """
        excess_margin = self.market_metadata_wrapper.excess_margin
        if excess_margin < self.initial_margin_required:
            actions[self.portfolio_increase_action_indices] = 0
        return actions


@action
class ExcessMarginActionWrapper(ActionWrapper):
    """
    Excess margin caputres the concept of free liquidity that needs to
    be mainted to facilitate increasing positions and maintaining
    existing positions. This wrapper in conjunction with appropriate
    trade ratio ensures that this free liquidity is alwasy large enough
    so that requirements for opening and maintaining positions are met
    at all times. 
    
    In the context of marginable assets this excess margin translates to
    the amount of marginable equity after subtracting maintenance margin
    requirement. In the context of nonmarginable assets this is
    equivalent to the amount of cash that is available at all times for
    trading as a percentage of portfolio value. 
    
    In the context of marginable assets it can be shown that if excess
    margin ratio (ratio of ecexss margin to portfolio value) is greater
    than delta > 0 then maintenance margin requirement and initial
    margin requirements are met at all times, if ratio of total value of
    trades with respect to equity is less than 1/(gross_initial_margin).
    Similarly for nonmarginable assets if excess margin ratio is greater
    than delta > 0 then it can be shown that there is always enough cash
    to buy more nonmarginable assets, given that trade ratio is less
    than delta/(1+ delta) < 1. 
    
    Respecting the excess margin ratio constraint also ensure that no
    margin call is received, since it by definition satisfies the
    maintenance margin requirement. If excess margin ratio is violated
    then actions that lead to increasing portfolio value are ignored
    until the ratio is restored to be greater than delta.

    Use this wrapper to:
        1. Provide some form of cushion around margin call/negative cash
           thresholds for safety, and more liquidity for trading.
        2. Proactively avoid triggering margin call avoidance mechanism
           in PositionOpenActionWrappper.
        3. Proactively avoid triggering negative cash avoidance in
           PositionOpenActionWrapper as long as trade ratio < delta/(1+
           delta) < 1.
        
    Args:
    ----------
    env: gym.Env
        The environment to wrap.
    delta: float
        The minimum excess margin ratio that needs to be maintained at
        all times.
    
    Attributes:
    ----------
    delta: float
        The minimum excess margin ratio that needs to be maintained at
        all times.
    excess_margin_ratio: float
        The ratio of excess margin to portfolio value.
    action_space: gym.spaces.Box    
        The action space of the environment.
    
    Examples
    --------
    marginable assets:
        if gross maintenance margin = 0.25 and delta = 0.05 then wrapper
        tries tries to maintain equity >= (0.25 + 0.05) *
        portfolio_value at all times. Note that delta >= 0 can be any
        non negative value. If delta is set to 0 then wrapper modifies
        agent's actions after margin call happens. Thus to avoid margin
        call it is recommended to set delta to a value high enough to
        avoid the inevitable effect of slippage.
    nonmarginable assets:
        if delta = 0.15 then wrapper tries to maintain cash >= 0.15 *
        portfolio_value at all times. Avaiability of cash for increasing
        portfolio value is ensured if trade ratio is less than 0.15 /

    Notes:
    ----------
    In order for initial margin requirement and cash availability is met
    at all times, trade_ratio in position sizing wrapper must be less
    than delta/(1+ delta) < 1 for nonmarginable assets, and less than
    1/(gross_initial_margin) for marginable assets. This ensures that
    size of trade is small enough and amount of liquidity is large
    enough that initial margin requirement and cash availability
    requirement is met at all times. If ratio if violated then actions
    leading to increasing portfolio value are ignored until the ratio is
    restored to be greater than delta. gross_initial_margin is defined
    as the sum of initial margin requirements for all assets divided by
    the sum of total value of all assets or portfolio value.
    """

    def __init__(self, env: Env, delta: float) -> None:
        """
        Initializes a new instance of the ExcessMarginActionWrapper
        class with the given environment and delta.

        Args:
        ----------
        env (Env):
            The environment to wrap.
        delta (float):
            The minimum excess margin ratio that needs to be maintained
            at all times.
        """
        super().__init__(env)
        if not delta > 0:
            raise ValueError('delta must be greater than 0.')

        self.delta = delta
        self.action_space = spaces.Box(-np.inf,
                                       np.inf,
                                       shape=(self.n_assets, ),
                                       dtype=GLOBAL_DATA_TYPE)

    @property
    def excess_margin_ratio(self):
        """
        Excess margin is defined as marginable equity minus maintenance
        marging requirement. This is the amount of equity that is
        available in form of cash or marginable securities that is free
        to be used for trading marginable securities. As long as excess
        margin is greater than zero a margin call will not be received.
        Note that if asset is non marginable then trade can only happe
        with cash. So excess margin can only be used to buy other
        marginable assets.
        """
        excess_margin = self.market_metadata_wrapper.excess_margin
        portfolio_value = self.market_metadata_wrapper.portfolio_value
        excess_margin_ratio = excess_margin / portfolio_value
        return excess_margin_ratio

    def action(self, actions: np.ndarray) -> np.ndarray:
        """
        If excess_margin_ratio is violated the corrective action is
        taken to ensure the portfolio value does not further increase.
        As long as the ratio stays bellow delta this corrective measure
        is taken until the ratio is restored to be greater than delta.
        This mechanism works by ignoreing any action that would lead to
        increasing portfolio value. This does not gurarantee that the
        ratio will be restored however it does gurarantee agents actions
        will not increase portfolio_value. This can be paired with
        a reward wrapper to penalize the agent for triggering the
        threshold, or receiving a margin call.
        """
        asset_quantities = self.market_metadata_wrapper.asset_quantities
        for action, quantity in zip(actions, asset_quantities):
            if self.excess_margin_ratio < self.delta:
                if (action < 0 and quantity <= 0
                        or action > 0 and quantity >= 0):
                    actions[action] = 0


@action
@metadata
class ShortingActionWrapper(ActionWrapper):
    """
    A wrapper that implements the shorting logic of the margin account.
    This wrapper modifies the agent's actions to ensure that if new
    short positions are opened then they are for shortable assets only.
    If shorting is not allowed for an asset then the action is modified
    to be zero. By default integer quantities are enforced for short
    actions. In other words fractional shorting is not allowed. This is
    Alpaca API default behavior.

    Args:
    ----------
    env: gym.Env
        The environment to wrap.

    Attributes:
    ----------
    n_assets (int):
        The number of assets in the environment.
    action_space (gym.spaces.Box):
        The action space of the environment.
    
    Example:
    --------
    >>> from neural.meta.env.base import TrainMarketEnv 
    >>> env = TrainMarketEnv(...)
    >>> wrapped_env = AlpacaShortActionWrapper(env)
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(-np.inf,
                                       np.inf,
                                       shape=(self.n_assets, ),
                                       dtype=GLOBAL_DATA_TYPE)

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        quantities = self.market_metadata_wrapper.asset_quantities
        asset_prices = self.market_metadata_wrapper.asset_prices
        assets = self.market_metadata_wrapper.assets

        for asset, quantity, action, price, asset_ in enumerate(
                zip(quantities, actions, asset_prices, assets)):
            if quantity <= 0 and action < 0:
                if not asset_.shortable:
                    action = 0
                elif asset.shortable:
                    actions[asset] = (action // price) * price

        return actions.astype(GLOBAL_DATA_TYPE)


@action
@metadata
class EquityBasedUniformActionInterpreter(ActionWrapper):
    """
    Transforms the actions produced by the agent that is in (-1, 1)
    range to be in (-max_trade, max_trade) range corresponding to
    notional value of buy/sell orders for each asset. if 0 then asset is
    held. The max_trade is calculated as a percentage of equity. It maps
    actions in the range (-1, 1) to buy/sell/hold using fixed zones for
    each action type. (-1, -hold_threshold) is mapped to sell,
    (-hold_threshold, hold_threshold) is mapped to hold, and
    (hold_threshold, 1) is mapped to buy. The hold_threshold is
    specified by user. The trade_ratio parameter controls the maximum
    percentage of equity that can be traded at each step. The action in
    the range (-threshold, threshold) is parsed as hold. The action
    outside this range is linearly projected to (0, max_trade). All
    assets have the same `budet` to trade, hence the name uniform. The
    budget is max_trade.
    """

    def __init__(self, env: Env, trade_equity_ratio=0.02, hold_threshold=0.15):
        """
        Initializes a new instance of the action wrapper with the given
        environment, trade ratio, and hold threshold.

        Args:
            env (Env): 
                The environment to wrap. 
            trade_equity_ratio (float, optional): 
                The maximum percentage of equity that can be traded
                at each step. Defaults to 0.02. 
            hold_threshold (float, optional): 
                The threshold for holding the current positions.
                Defaults to 0.15. This reserves the region (-0.15, 0.15)
                in (-1, 1) for hold actions.

        Attributes:
            trade_ratio (float): 
                The maximum percentage of equity that can be traded
                at each step. 
            hold_threshold (float): The threshold for
                holding the current positions.
            _max_trade_per_asset (float): The maximum trade that can be
                made for each asset. Initialized to None. 
            action_space (Box): 
                The action space of the wrapped environment.

        Returns:
            None.
        """

        super().__init__(env)

        assert 0 < trade_equity_ratio <= 1, "Trade ratio must be a float in (0, 1]."
        assert 0 < hold_threshold < 1, "Hold threshold must be a float in (0, 1)."

        self.trade_ratio = trade_equity_ratio
        self.hold_threshold = hold_threshold
        self._max_trade_per_asset = None
        self.n_assets = self.market_metadata_wrapper.n_assets

        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(self.n_assets, ),
                                       dtype=GLOBAL_DATA_TYPE)

        return None

    def _set_max_trade_per_asset(self, trade_ratio: float) -> None:
        """
        Sets the value for the maximum trade that can be made for each
        asset based on the given trade ratio.

        Args:
        --------
            trade_ratio(float): 
                The maximum percentage of net worth that can be traded
                at each step.

        Returns:
        --------
            float: The maximum trade that can be made for each asset.

        Notes:
        --------
            The recommended value for initial_cash is >=
            n_assets/trade_ratio.
        """

        self._max_trade_per_asset = (
            trade_ratio * self.market_metadata_wrapper.equity) / self.n_assets

        return None

    def parse_action(self, action: float) -> float:
        """
        Parses the given action value as buy, sell, or hold based on the
        hold threshold and maximum trade per asset.

        Args:

            action (float): 
                The action value to parse.

        Returns:
        --------
            float: 
                The parsed action value.

        Notes:
            Actions within the range (-hold_threshold, hold_threshold)
            are parsed as hold. Actions outside this range are linearly
            projected to the range (0, max_trade_per_asset)
        """

        fraction = (abs(action) - self.hold_threshold) / (1 -
                                                          self.hold_threshold)

        parsed_action = (fraction * self._max_trade_per_asset *
                         np.sign(action) if fraction > 0 else 0)

        return parsed_action

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        """
        Limits maximum value of trades to a fixed percentage of equity. 
        This ensures that total value of trades does not exceed this 
        fixed percentage of equity. It also parses the actions to be
        buy, sell, or hold.

        Args:
            actions (np.ndarray[float]): The actions to perform.

        Returns:
            np.ndarray[float]: The parsed actions.

        Notes:
            This method sets the maximum trade per asset based on the
            trade ratio and parses each action value as buy, sell, or
            hold using the parse_action method.
        """

        self._set_max_trade_per_asset(self.trade_ratio)
        assets = self.market_metadata_wrapper.assets
        asset_prices = self.market_metadata_wrapper.asset_prices
        for asset, action, asset_, price in enumerate(
                zip(actions, assets, asset_prices)):
            action = self.parse_action(action)
            actions[asset] = action if asset_.fractionable else (action //
                                                                 price) * price
        return actions


@action
@metadata
class ActionClipperWrapper(ActionWrapper):
    """
    A wrapper that clips actions to an expected range for downstream
    position sizing wrappers. This wrapper is applied upstream and
    enforces that the action values are within a certain range before
    being passed to the downstream wrappers. The action values received
    are usually the immediate output of the model.

    Args:
    ----------
        env (gym.Env): 
            The environment to wrap. 
        low (float, optional): 
            The minimum value for the actions. Defaults to -1. high 
        (float, optional): 
            The maximum value for the actions. Defaults to 1.

    Attributes:
    ----------
        low (float): 
            The minimum value for the actions. 
        high (float):
            The maximum value for the actions. 
        n_assets (int): 
            The number of symbols in the environment. 
        action_space (gym.spaces.Box):
            The action space of the environment.

    Methods:
    ----------
        __init__(self, env: Env, low: float=-1, high: float=1) -> None:
            Initializes the ActionClipperWrapper instance.
        action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
            Clips the actions to be within the given low and high
            values.

    Example:
        >>> from neural.meta.env.base import TrainMarketEnv
        >>> env = TrainMarketEnv(...)
        >>> wrapped_env = ActionClipperWrapper(env, low=-0.5, high=0.5)
    """

    def __init__(self, env: Env, low: float = -1, high: float = 1) -> None:

        super().__init__(env)

        self.low = low
        self.high = high
        self.n_assets = self.market_metadata_wrapper.n_assets
        self.action_space = (spaces.Box(self.low,
                                        self.high,
                                        shape=(self.n_assets, ),
                                        dtype=GLOBAL_DATA_TYPE))

        return None

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:
        """
        Clips the actions to be within the given low and high values.

        Args:
        ----------
            actions (np.ndarray): The array of actions to clip.

        Returns:
        ----------
            np.ndarray: The clipped array of actions.
        """

        actions = np.clip(actions, self.low, self.high)

        return actions.astype(GLOBAL_DATA_TYPE)


class DirectionalTradeActionWrapper:
    """
    Checks if trend up or trend down condition is met then modifies 
    agents actions to open buy only or sell only positions respectively.
    Useful if injecting the notion of trend following into the agent can
    lead to potential performance improvements. This happens by
    rejecting actions of agent that are in the opposite direction of
    current trend.
    """
