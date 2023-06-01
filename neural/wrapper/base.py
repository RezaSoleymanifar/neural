from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Type, Dict, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from gym import Env, Wrapper

from neural.common.log import logger
from neural.common.exceptions import IncompatibleWrapperError
from neural.env.base import AbstractMarketEnv, TrainMarketEnv, TradeMarketEnv

from neural.utils.base import sharpe_ratio, tabular_print


def market(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:
    """
    A decorator that augments an existing Gym wrapper class by adding a
    pointer to the underlying market environment, if the base
    environment is a market environment. The pointer is accessible via
    the `market_env` attribute of the augmented wrapper class.

    Args: 
    ----------
    wrapper_class (Type[Wrapper]): 
        The base Gym wrapper class to be augmented. This should be a
        subclass of `gym.Wrapper`.

    Returns:
    ----------
        Type[Wrapper]: 
            A new wrapper class that augments the input wrapper class
            with a pointer to the underlying market environment.

    Raises:
    -------
        TypeError: 
            If the `wrapper_class` is not a subclass of `Wrapper`. 
        IncompatibleWrapperError: 
            If the unwrapped base environment is not of type
            AbstractMarketEnv.

    Examples:
    ---------
        >>> from neural.wrapper.base import market

        >>> @market
        ... class MyWrapper:
        ...    ...
    """

    if not issubclass(wrapper_class, Wrapper):

        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class MarketEnvDependentWrapper(wrapper_class):
        """
        A wrapper that creates a pointer to the base market environment
        and checks if the unwrapped base env is a market env. If the
        check fails, an error is raised.

        Parameters:
        -----------
            env (gym.Env): The environment being wrapped.

        Raises:
        -------
            IncompatibleWrapperError: 
                If the unwrapped base environment is not of type
                AbstractMarketEnv.
        
        Attributes:
        ----------
            market_env (AbstractMarketEnv): A pointer to the underlying
            market environment.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:
            """
            Initializes the MarketEnvDependentWrapper instance. It first
            checks if the unwrapped base env is a market env and creates
            a pointer to it. If the check fails, an error is raised.
            After this process the constructor of the base wrapper class
            is called. Thus variables in constructor of the base wrapper
            have access to market env pointer, prior to initialization.

            Parameters:
            ----------
                env: gym.Env 
                    (gym.Env): The environment to be wrapped.
                *args: Variable length argument list to be passed to
                    the wrapper.
                **kwargs: Arbitrary keyword arguments to be passed to
                    the wrapper.
            """

            self.market_env = self.check_unwrapped(env)
            super().__init__(env, *args, **kwargs)

        def check_unwrapped(self, env: Env) -> AbstractMarketEnv:
            """
            Checks if the unwrapped base env is a market env.

            Parameters:
            -----------
                env (gym.Env): The environment being wrapped.

            Raises:
            -------
                IncompatibleWrapperError: 
                    If the unwrapped base environment is not of type
                    AbstractMarketEnv.

            Returns:
            -
                AbstractMarketEnv: The unwrapped market env.
            """

            if not isinstance(env.unwrapped, AbstractMarketEnv):

                raise IncompatibleWrapperError(
                    f'{wrapper_class.__name__} requires its unwrapped '
                    'env to be of type {AbstractMarketEnv}.')

            return env.unwrapped

    return MarketEnvDependentWrapper


def metadata(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:
    """
    A decorator that augments an existing Gym wrapper class by adding a
    pointer to the underlying market metadata wrapper. A market metadata
    wrapper provides any form of information beyond asset quantities and
    cash that is already provided by the base market environment. This
    includes information on equity, shorts, longs, portfolio value,
    maintenance margin, and more. The pointer is accessible via the
    `market_metadata_wrapper` attribute of the augmented wrapper class.
    The unwrapped environment is accessible through
    self.market_metadata_wrapper.market_env 

    Args:
    ----------
        wrapper_class (Type[Wrapper]):
            The base Gym wrapper class to be augmented. This should be a
            subclass of `gym.Wrapper`.
        
    Returns:
    ----------
        Type[Wrapper]:
            A new wrapper class that augments the input wrapper class
            with a pointer to the underlying market metadata wrapper.
    
    Raises:
    -------
        TypeError:
            If the `wrapper_class` is not a subclass of `Wrapper`.
        IncompatibleWrapperError:
            If the unwrapped base environment is not of type
            AbstractMarketEnvMetadataWrapper.

    Examples:
    ---------
        >>> from neural.wrapper.base import metadata

        >>> @metadata
        ... class MyWrapper:
        ...    ...
    """
    if not issubclass(wrapper_class, Wrapper):

        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class MarketMetadataDependentWrapper(wrapper_class):
        """
        A wrapper that creates a pointer to the underlying market
        metadata wrapper. If the check fails, an error is raised.

        Args:
        ----------
            env (gym.Env): The environment being wrapped.

        Raises:
        ----------
            IncompatibleWrapperError: 
                If the market metadata wrapper is not found in the
                wrapper stack.
        
        Attributes:
        ----------
            market_metadata_wrapper (AbstractMarketEnvMetadataWrapper):
                A pointer to the underlying market metadata wrapper.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:
            """
            Initializes the MarketEnvMetadataWrapper dependent wrapper
            class. pointer to metadata wrapper is created prior to
            initialization of base wrapper class. Thus variables in
            constructor of the base wrapper have access to metadata
            wrapper pointer prior to initialization.

            Args:
                env (gym.Env): 
                    The environment to be wrapped. 
                *args: 
                    Variable length argument list to be passed to the
                    wrapper. 
                **kwargs: 
                    Arbitrary keyword arguments to be passed to the
                    wrapper.
            """

            self.market_metadata_wrapper = self.find_metadata_wrapper(env)
            super().__init__(env, *args, **kwargs)

            return None

        def find_metadata_wrapper(self,
                                  env: Env) -> AbstractMarketEnvMetadataWrapper:
            """
            Recursively searches through wrapped environment to find the
            first instance of an AbstractMarketEnvMetadataWrapper in the
            wrapper stack. Note that by inheriting from gym.Wrapper,
            then self.env is the next wrapper in the stack. also if not
            overridden, self.step() calls self.env.step() and so on.
            Same for reset(). Wrappers are stacked though passing the
            previous wrapper as an argument to the constructor of the
            next wrapper. Thus, if the wrapper stack is
            MarketEnvMetadataWrapper -> MarketEnvWrapper -> MarketEnv
            then self.env is MarketEnvWrapper.

            Args
            ----------
            env : gym.Env
                The environment to search through.

            Returns
            -------
            AbstractMarketEnvMetadataWrapper
                The first instance of AbstractMarketEnvMetadataWrapper
                found in the wrapper stack.
            
            Raises
            ------
            IncompatibleWrapperError
                If an AbstractMarketEnvMetadataWrapper is not found in
                the wrapper stack.
            """

            if isinstance(env, AbstractMarketEnvMetadataWrapper):
                return env

            elif hasattr(env, 'env'):
                return self.find_metadata_wrapper(env.env)

            else:
                raise IncompatibleWrapperError(
                    f'{wrapper_class.__name__} requires a wrapper of type '
                    f'{AbstractMarketEnvMetadataWrapper} in enclosed wrappers.')

    return MarketMetadataDependentWrapper


class AbstractMarketEnvMetadataWrapper(Wrapper, ABC):
    """
    A blueprint class for market metadata wrappers. A metadata wrapper
    acts as a hub for storing metadata about the environment during an
    episode. Other wrappers that need access to this metadata can use
    the decorator `@metadata` to create a pointer to the metadata
    wrapper. This pointer is accessible via the
    `market_metadata_wrapper` attribute of the augmented wrapper class.
    The unwrapped environment is accessible through
    self.market_metadata_wrapper.market_env attribute.
    This this wrapper acts as a data hub for other wrappers, it
    duplicates some of the attributes of the underlying market env. This
    is done to avoid the need for wrappers to have access to the
    underlying market env in addition to the metadata wrapper.

    This class is designed to be subclassed for creating custom metadata
    wrappers for market environments. Metadata wrappers are used to keep
    track of additional information about the environment during an
    episode, such as equity, shorts, longs, portfolio value, maintenance
    margin, and more.

    Attributes:
    ----------
        initial_cash (float):
            The initial amount of cash available in the environment.
        initial_asset_quantities (np.ndarray):
            The initial quantities of assets available in the
            environment.
        data_metadata (DataMetadata):
            The metadata of the data used to create the environment.
        feature_schema (FeatureSchema):
            The schema of the features used to create the environment.
        assets (List[str]):
            The assets used to create the environment.
        n_steps (int):  
            The number of steps in the environment.
        history (defaultdict): 
            A defaultdict object for storing metadata during an episode.

    Properties:
    ----------
        schedule (pd.DataFrame):
            The schedule of the market environment.
        index (int):
            The current index of the episode.
        day (int):
            The current day of the episode.
        date (pd.Timestamp):
            The current date of the episode.
        cash (float):
            The current amount of cash available in the environment.
        asset_quantities (np.ndarray):
            The current quantities of assets available in the
            environment.
        asset_prices (np.ndarray):
            The current prices of assets available in the environment.
        progress (float):
            The current progress of the episode.
        profit (float):
            The current profit of the episode.
        return_ (float):
            The current return of the episode.
        sharpe (float):
            The current sharpe ratio of the episode.
    
    Methods:
    ----------
        _cache_metadata():
            Caches the metadata of the environment.
        reset():
            Resets the environment to the initial state.
        step(action):
            Steps the environment forward by one step.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.initial_cash = self.market_env.initial_cash
        self.initial_asset_quantities =\
              self.market_env.initial_asset_quantities

        self.data_metadata = self.market_env.data_metadata
        self.feature_schema = self.market_env.feature_schema
        self.assets = self.market_env.assets

        self.n_steps = self.market_env.n_steps
        self.n_assets = self.market_env.n_assets

        self.history = defaultdict(list)
    
    @property
    def index(self) -> int:
        """
        The current index of the episode.

        Returns:
        ----------
            index (int):
                The current index of the episode.
        """
        return self.market_env.index

    @property
    def date(self) -> pd.Timestamp:
        """
        Returns the current date of the episode.
        """
        date = self.market_env.data_feed.get_date(self.index)
        return date

    @property
    def cash(self) -> float:
        """
        The current amount of cash available in the environment.
        Can be positive or negative.

        Returns:
        ----------
            cash (float):
                The current amount of cash available in the environment.
        """
        return self.market_env.cash

    @property
    def asset_quantities(self) -> np.ndarray[float]:
        """
        The current quantity of each asset held.

        Returns:
        ----------
            asset_quantities (np.ndarray[float]):
                The current quantity of each asset held.
        """
        return self.market_env.asset_quantities

    @property
    def asset_prices(self) -> np.ndarray[float]:
        """
        The current price of each asset held.

        Returns:    
        ----------
            asset_prices (np.ndarray[float]):
                The current price of each asset held by the trader.
        """
        return self.market_env.asset_prices

    @property
    def progress(self) -> str:
        """
        If the underlying market env is a `TrainMarketEnv`, this method returns
        the progress of the episode as a percentage string in ['0%', '100%'].
        If the underlying market env is a `TradeMarketEnv`, this method returns
        current date and time.

        Returns:
        ----------
            progress (str): 
                The progress of the episode as a percentage string or the
                current date and time.
        """
        if isinstance(self.market_env, TradeMarketEnv):
            progress = datetime.now().strftime("%m/%d/%Y, %H:%M")
        else:
            progress_percentage = self.market_env.index / self.market_env.n_steps
            progress = f'{progress_percentage:.0%}'

        return progress

    @property
    def profit(self):
        """
        The current profit made in the environment. This is the
        difference between the current equity and the initial equity.

        Returns:
        ----------
            profit (float):
                The current profit of the trader.
        """
        profit = self.equity - self.initial_equity
        return profit

    @property
    def return_(self):
        """
        The current return on initial equity. This is the ratio of the
        current profit to the initial equity.

        Returns:
        ----------
            return_ (float):
                The current return of the trader.
        """
        return_ = self.profit / self.initial_equity
        return return_

    @property
    def sharpe(self):
        """
        The current sharpe ratio of the trader. This is the ratio of the
        current return to the current volatility of the equity.

        Returns:
        ----------
            sharpe (float):
                The current sharpe ratio of the trader.
        """
        sharpe = sharpe_ratio(self.history['equity'])
        return sharpe
    
    @abstractmethod
    def _cache_metadata(self):
        """
        An abstract method for caching metadata.

        This method should be implemented by subclasses to cache
        metadata as necessary during an episode. The method takes an
        arbitrary number of arguments and keyword arguments, depending
        on the specific metadata that needs to be cached.
        """

        raise NotImplementedError

    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Resets the environment and updates metadata.

        This method resets the environment and updates the metadata
        stored in the wrapper. It returns the initial observation from
        the environment.

        Returns:
        ----------
            observation (Dict[str, np.ndarray[float]]): 
                The initial observation from the environment.
        """

        observation = self.env.reset()
        self._cache_metadata()

        return observation

    def step(
        self, action: np.ndarray[float] | Dict[Any, np.ndarray[float]]
    ) -> Tuple[np.ndarray[float] | Dict[str, np.ndarray[float]], float, bool,
               Dict]:
        """
        Performs a step in the environment. Actions taken can be either
        numpy arrays or dictionaries of numpy arrays.

        Args:
        ----------
            action (np.ndarray[float] | Dict[str, np.ndarray[float]]): 
                The action to be taken by the agent. Action can be
                either a numpy array or a dictionary of numpy arrays.

        Returns:
        ----------
            Tuple[np.ndarray[float] | Dict[str, np.ndarray[float]],
            float, bool, Dict]: 
                A tuple containing the new observation, the reward
                obtained, a boolean indicating whether the episode has
                ended, and a dictionary containing additional
                information. Observation can be either a numpy array or
                a dictionary of numpy arrays.
        """

        observation, reward, done, info = self.env.step(action)
        self._cache_metadata()

        return observation, reward, done, info


@market
class MarginAccountMetaDataWrapper(AbstractMarketEnvMetadataWrapper):
    """
    A metadata wrapper acts as a hub for storing metadata about the
    environment during an episode. This wrapper assumes that the info
    collected from the base env is for a margin account. Collects
    metadata of a margin account including equity, shorts, longs,
    portfolio value, maintenance margin, and more. Other wrappers that
    need access to this metadata can use the decorator `@metadata` to
    create a pointer to the metadata wrapper. This pointer is accessible
    via the `market_metadata_wrapper` attribute of the augmented wrapper
    class. This this wrapper acts as a data hub for other wrappers, it
    duplicates some of the attributes of the underlying market env. This
    is done to avoid the need for wrappers to have access to the
    underlying market env in addition to the metadata wrapper. This
    wrapper is designed to be used with `AbstractMarketEnv`
    environments, to simulate a margin account.

    Attributes:
    ----------
        initial_cash (float):
            The initial amount of cash available in the environment.
        initial_asset_quantities (np.ndarray[float]):
            The initial quantity of each asset held by the trader.
        feature_schema (Dict[str, str]):
            A dictionary mapping feature names to their data types.
        assets (List[Asset]):
            A list of assets in the environment.
        n_steps (int):
            The number of steps in the environment.
        n_assets (int): 
            The number of assets in the environment.
        history (defaultdict):
            A defaultdict object for storing metadata during an episode.

    Properties:
    ----------
        schedule (pd.DataFrame):
            The schedule of the market environment.
        index (int):
            The current index of the episode.
        day (int):
            The current day of the episode.
        date (pd.Timestamp):
            The current date of the episode.
        cash (float):
            The current amount of cash available in the environment.
        asset_quantities (np.ndarray):
            The current quantities of assets available in the
            environment.
        asset_prices (np.ndarray):
            The current prices of assets available in the environment.
        progress (float):
            The current progress of the episode.
        profit (float):
            The current profit of the episode.
        return_ (float):
            The current return of the episode.
        sharpe (float):
            The current sharpe ratio of the episode.
        longs (float):
            The current notional value of long positions held in the
            market environment.
        shorts (float):
            The current notional value of short positions held in the
            market environment. 
        positions (np.ndarray[float]):
            The current positions (notional base currency value) of each
            asset held. The position of each asset is the quantity of
            each asset held times its price. Position of an asset is
            always positive.
        portfolio_value (float):
            The current portfolio value of the trader. this includes sum
            of all positions, both long and short.
        equity (float):
            The current equity of the trader. Equity is the sum of all
            long positions and cash(+/-), minus short positions. This
            caputures the concept of owened liquidity that can be used
            to maintain existing assets in the context of marginable
            assets. In the context of nonmarginable assets, since
            shorting is not allowed, this is the same as the cash plus
            the value of long positions. There is no concept of
            maintaining a position in nonmarginable assets, since they
            are purchased with cash and cannot be used as collateral.
        marginable_equity (float):
            The current marginable equity of the trader. This is the
            equity that can be used to open new positions and acts
            similar to available cash, due to marginability of the
            underlying assets. When trading assets the gross intial
            margin of the assets should not exceed the marginable
            equity. In the context of non-margin trading, this is the
            same as the cash. In the context of margin trading, this is
            the same as equity. In short, equity = marginable_equity +
            non_marginable_longs. Since non-marginable_longs cannot be
            used to open new positions (cannot be used as collateral),
            this value is not included in the marginable equity.
        maintenance_margin_requirement (float):
            The current maintenance margin requirement of the trader.
            This is the minimum amount of equity that must be maintained
            in the account to avoid a margin call. Maintenance margin
            requirement of nonmarginable assets is always zero. This is
            an abuse of terminology, since nonmarginable assets do not
            have a concept of margin.
        excess_margin (float):
            Excess margin is the amount of marginable equity above the
            maintenance margin requirement. In the context of
            nonmarginable assets, this is the same as the cash. Excess
            margin is set to maintain a ceratain ratio with respect to
            porfolio value. This ensures that:
                1) maintenance margin requirement is always met (by
                    definition)
                2) Given small enough trade to equity ratio, the trader
                    has enough marginable equity to open new positions
                    (automatically satisfying initial margin
                    reuirements at all trades).
                3) If initial margin of purchased assets do not exceed
                    the excess margin, it also guarantees that post
                    transaction the maintenance margin requirement is
                    also met.

    Methods:
    ----------
        _cache_metadata:
            An abstract method for caching metadata.
        reset:
            Resets the environment and updates metadata.
        step:
            Performs a step in the environment. Actions taken can be
            either numpy arrays or dictionaries of numpy arrays.
        render:
            Prints the trading metadata tear sheet to console.

    Examples:
    ---------
        >>> from neural.wrapper.base import MarketEnvMetadataWrapper
        >>> from neural.env.base import TrainMarketEnv
        >>> env = TrainMarketEnv(...)
        >>> env = MarketEnvMetadataWrapper(env)
    """

    def __init__(self, env: Env) -> None:

        super().__init__(env)

        return None

    @property
    def longs(self) -> float:
        """
        The current notional value of long positions held in the market
        environment.

        Returns:
        ----------
            longs (float):
                The current notional value of long positions held in the
                market environment.
        """
        long_mask = self.asset_quantities > 0

        longs = self.asset_quantities[long_mask] @ self.asset_prices[long_mask]

        return longs

    @property
    def shorts(self) -> float:
        """
        The current notional value of short positions held in the 
        market environment.

        Returns:
        ----------
            shorts (float):
                The current notional value of short positions held in
                the market environment.
        """
        short_mask = self.asset_quantities < 0
        shorts = np.abs(
            self.asset_quantities[short_mask]) @ self.asset_prices[short_mask]

        return shorts

    @property
    def positions(self) -> np.ndarray[float]:
        """
        The current positions (notional base currency value) of each
        asset. The position of each asset is the quantity of each
        asset times its price. Position of an asset is always
        positive.

        Returns:
        ----------
            positions 
                (np.ndarray[float]): The current positions of each asset
                in the environment.
        """
        positions = np.abs(self.asset_quantities * self.asset_prices)
        return positions

    @property
    def portfolio_value(self) -> float:
        """
        The current portfolio value of the trader. this includes sum of
        all positions, both long and short.

        Returns:
        ----------
            portfolio_value (float):
                The current portfolio value of the trader.
        """
        portfolio_value = sum(self.positions)
        return portfolio_value

    @property
    def equity(self) -> float:
        """
        The current equity of the trader. Equity is the sum of all long
        positions and cash(+/-), minus short positions. This caputures
        the concept of total ownership or debt substracted value of
        assets/cash that can be used to maintain existing assets in the
        context of marginable assets. In the context of nonmarginable
        assets, since shorting is not allowed, this is the same as the
        cash plus the value of long positions. There is no concept of
        maintaining a position in nonmarginable assets, since they are
        purchased with cash and cannot be used as collateral.

        Returns:
        ----------
            equity (float):
                The current equity of the trader.
        """
        equity = self.longs + self.cash - self.shorts
        return equity

    @property
    def marginable_equity(self) -> float:
        """
        The current marginable equity of the trader. This is the equity
        that can be used to open new positions and acts similar to
        available cash, due to marginability of the underlying assets.
        When trading assets the gross intial margin of the assets should
        not exceed the marginable equity. In the context of non-margin
        trading, this is the same as the cash. In the context of margin
        trading, this is the same as equity minus non_marginable_longs.
        Namely value of non marginable assets held in the portfolio do
        not contribute to marginable equity. In short, equity =
        marginable_equity + non_marginable_longs. Since
        non_marginable_longs cannot be used to open new positions
        (cannot be used as collateral), this value is not included in
        the marginable equity.

        Returns:
        ----------
            marginable_equity (float):
                The current marginable equity of the trader.
        """
        non_marginable_longs = 0
        for asset, quantity, position in zip(self.assets, self.asset_quantities,
                                             self.positions):
            if not asset.marginable:
                non_marginable_longs += (position if quantity > 0 else 0)

        marginable_equity = self.equity - non_marginable_longs
        return marginable_equity

    @property
    def maintenance_margin_requirement(self) -> float:
        """
        The current maintenance margin requirement of the trader. This
        is the minimum amount of equity that must be maintained in the
        account to avoid a margin call. Maintenance margin requirement
        of nonmarginable assets is always  set to zero. This is an abuse
        of terminology, since nonmarginable assets do not have a concept
        of margin.

        Returns:
        ----------
            maintenance_margin_requirement (float):
                The current maintenance margin requirement of the
                trader.
        """
        margin_required = sum(
            asset.get_maintenance_margin(short=quantity < 0, price=price) *
            position for asset, position, quantity, price in zip(
                self.assets, self.positions, self.asset_quantities,
                self.asset_prices))
        return margin_required

    @property
    def excess_margin(self) -> float:
        """
        Excess margin is the amount of marginable equity above the
        maintenance margin requirement. In the context of nonmarginable
        assets, this is the same as the positive cash. Excess margin is
        set to maintain a certain ratio with respect to porfolio value.
        This ensures that: 
            1) maintenance margin requirement is always met (by
                definition) 
            2) Given small enough trade to equity ratio, the trader
                has enough marginable equity to open new positions
                (implicitly satisfying initial margin reuirements). 
            3) If initial margin of purchased assets do not exceed the
               excess margin, it also gurarantees that post transaction
               the maintenance margin requirement is also met.
        
        Returns:
        ----------
            excess_margin (float):
                Excess margin is the amount of marginable equity above
                the maintenance margin requirement. In the context of
                nonmarginable assets, this is the same as the cash.
        """
        excess_margin = (
            self.marginable_equity - self.maintenance_margin_requirement)
        return excess_margin

    @property
    def excess_margin_ratio(self):
        """
        Result of dividing excess margin by portfolio value. This metric
        can be used to determine the amount of cushion around the margin
        call threshold. Can be used to provide both protection and
        guaranteed liquidity.

        Returns:
        ----------
            excess_margin_ratio (float):
                Excess margin ratio is the ratio of excess margin to
                portfolio value.
        """
        excess_margin = self.market_metadata_wrapper.excess_margin
        portfolio_value = self.market_metadata_wrapper.portfolio_value
        excess_margin_ratio = excess_margin / portfolio_value
        return excess_margin_ratio
    
    def _cache_metadata(self) -> None:
        """
        Caches metadata during an episode. This method is called by the
        `reset` and `step` methods of the wrapper.
        """
        self.history['cash'].append(self.cash)
        self.history['longs'].append(self.longs)
        self.history['shorts'].append(self.shorts)
        self.history['portfolio_value'].append(self.portfolio_value)
        self.history['equity'].append(self.equity)
        self.history['profit'].append(self.profit)
        self.history['return'].append(self.return_)
        self.history['sharpe'].append(self.sharpe)

        return None


@market
@metadata
class ConsoleTearsheetRenderWrapper(Wrapper):
    """
    A wrapper that prints a tear sheet to console showing market env
    metadata. The tear sheet is printed every 'verbosity' times. If the
    underlying market env is a `TrainMarketEnv`, the tear sheet is
    printed every 'n_steps / verbosity' steps. If the underlying market
    env is a `TradeMarketEnv`, the tear sheet is printed every step of 
    trading. The tear sheet shows the following information:
        - Progress: The progress of the episode as a percentage.
        - Return: The current return of the trader. 
        - Sharpe ratio: The current sharpe ratio of the trader. 
        - Profit: The current profit of the trader. 
        - Equity: The current equity of the trader. 
        - Cash: The current amount of cash available to the trader.
        - Portfolio value: The current portfolio value of the trader.
        - Longs: The current notional value of long positions held in
            the market environment.
        - Shorts: The current notional value of short positions held in
            the market environment.
    
    Attributes:
    ----------
        env (gym.Env):
            The environment being wrapped.
        market_env (AbstractMarketEnv):
            A pointer to the underlying market environment.
        market_metadata_wrapper (AbstractMarketEnvMetadataWrapper):
            A pointer to the underlying market metadata wrapper.
        verbosity (int):
            The frequency of printing the tear sheet. If the underlying
            market env is a `TrainMarketEnv`, the tear sheet is printed
            every 'n_steps / verbosity' steps. If the underlying market
            env is a `TradeMarketEnv`, the tear sheet is printed every
            step of trading.
        render_every (int):
            The frequency of printing the tear sheet. If the underlying
            market env is a `TrainMarketEnv`, the tear sheet is printed 
            every 'n_steps / verbosity' steps. If the underlying market 
            env is a `TradeMarketEnv`, the tear sheet is printed every
            step of trading.
        
    Methods:
    ----------
        reset:
            Resets the environment and updates metadata.
        step:
            Performs a step in the environment. Actions taken can be
            either numpy arrays or dictionaries of numpy arrays.
        render:
            Prints the trading metadata tear sheet to console.
        
    Raises:
    -------
        ValueError:
            If the verbosity is not an integer in [1, n_steps].
    
    Examples:
    ---------
        >>> from neural.wrapper.base import ConsoleTearsheetRenderWrapper
        >>> from neural.env.base import TrainMarketEnv  
        >>> env = TrainMarketEnv(...)
        >>> env = ConsoleTearsheetRenderWrapper(env)
    """

    def __init__(self, env: Env, verbosity: int = 20) -> None:
        """
        Initializes the ConsoleTearsheetRenderWrapper instance.

        Args:
        ----------
            env (gym.Env):
                The environment being wrapped.
            verbosity (int):
                The frequency of printing the tear sheet. If the
                underlying market env is a `TrainMarketEnv`, the tear
                sheet is printed every 'n_steps / verbosity' steps. If
                the underlying market env is a `TradeMarketEnv`, the
                tear sheet is printed every step of trading.
        
        Raises:
        -------
            ValueError:
                If the verbosity is not an integer in [1, n_steps].
        """
        super().__init__(env)
        self.verbosity = verbosity

        if not isinstance(self.verbosity, int) or not (1 <= self.verbosity <=
                                                       self.market_env.n_steps):
            raise ValueError(
                f'Verbosity must be integer in [1, {self.market_env.n_steps}]'
                ', but got {verbosity}')
        self._set_render_frequency()
        return None

    def _set_render_frequency(self) -> None:
        """
        Sets the frequency of printing the tear sheet. If the underlying
        market env is a `TrainMarketEnv`, the tear sheet is printed
        every 'n_steps / verbosity' steps. If the underlying market env
        is a `TradeMarketEnv`, the tear sheet is printed every step of
        trading.
        """
        if isinstance(self.market_env, TrainMarketEnv):
            self.render_every = self.market_env.n_steps // self.verbosity
        elif isinstance(self.market_env, TradeMarketEnv):
            self.render_every = 1
        return None

    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Resets the environment and updates metadata. Prints the trading
        metadata tear sheet to console.

        Returns:
        ----------
            observation (Dict[str, np.ndarray[float]]):
                The initial observation from the environment.
        """
        observation = self.env.reset()
        resolution = self.market_env.data_metadata.resolution
        start_date = self.market_env.data_feeder.start_date
        end_date = self.market_env.data_feeder.end_date
        days = self.market_env.data_feeder.days

        logger.info('Episode:'
                    f'\n\t start = {start_date}'
                    f'\n\t end = {end_date}'
                    f'\n\t days = {days}'
                    f'\n\t resolution = {resolution}'
                    f'\n\t n_assets = {self.market_env.n_assets}'
                    f'\n\t n_steps = {self.market_env.n_steps:,}'
                    f'\n\t n_features = {self.market_env.n_features:,}')

        self.render()
        return observation

    def step(
        self, actions: np.ndarray[float] | Dict[Any, np.ndarray[float]]
    ) -> Tuple[np.ndarray[float] | Dict[str, np.ndarray[float]], float, bool,
               dict()]:
        """
        Takes a step in the environment and updates the tear sheet if
        necessary.

        Args:
        ----------
            actions (np.ndarray | Dict[str, np.ndarray]): The actions to
            be taken by the agent.

        Returns:
        ----------
            Tuple: A tuple containing the new observation, the reward
            obtained, a boolean indicating whether the episode has
            ended, and a dictionary containing additional information.
        """

        observation, reward, done, info = self.env.step(actions)

        if self.market_env.index % self.render_every == 0 or done:
            self.render()

        return observation, reward, done, info

    def render(self, mode='human') -> None:
        """
        Prints the trading metadata tear sheet to console.
        """
        cash = self.market_metadata_wrapper.cash
        portfolio_value = sum(self.market_metadata_wrapper.positions)
        longs = self.market_metadata_wrapper.longs
        shorts = self.market_metadata_wrapper.shorts
        equity = self.market_metadata_wrapper.equity

        progress = self.market_metadata_wrapper.progress
        profit = self.market_metadata_wrapper.profit
        return_ = self.market_metadata_wrapper.return_
        sharpe = self.market_metadata_wrapper.sharpe

        metrics = [
            progress, f'{return_:.2%}', f'{sharpe:.4f}',
            f'${profit:,.0f}', f'${equity:,.0f}', f'${cash:,.0f}',
            f'${portfolio_value:,.0f}', f'${longs:,.0f}', f'${shorts:,.0f}'
        ]

        index = self.market_env.index
        if index == 0:

            title = [
                'Progress', 'Return', 'Sharpe ratio', 'Profit', 'Equity',
                'Cash', 'Portfolio value', 'Longs', 'Shorts'
            ]

            print(tabular_print(title, header=True))

        print(tabular_print(metrics))

        return None
