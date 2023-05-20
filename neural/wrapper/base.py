from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Type, Dict, Tuple, Any
from datetime import datetime

import numpy as np
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
        history (defaultdict): 
            A defaultdict object for storing metadata during an episode.
        initial_cash (float):
            The initial amount of cash available in the environment.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.initial_cash = self.market_env.initial_cash
        self.initial_asset_quantities =\
              self.market_env.initial_asset_quantities

        self.feature_schema = self.market_env.feature_schema
        self.assets = self.market_env.assets

        self.n_steps = self.market_env.n_steps
        self.n_assets = self.market_env.n_assets

        self.history = defaultdict(list)

    @property
    def index(self) -> int:
        """
        The current index of the episode.
        """
        return self.market_env.index

    @property
    def cash(self) -> float:
        """
        The current amount of cash available to the trader.
        """
        return self.market_env.cash

    @property
    def asset_quantities(self) -> np.ndarray[float]:
        """
        The current quantity of each asset held by the trader.
        """
        return self.market_env.asset_quantities

    @property
    def asset_prices(self) -> np.ndarray[float]:
        """
        The current price of each asset held by the trader.
        """
        return self.market_env.asset_prices

    @abstractmethod
    def _cache_metadata(self, *args, **kwargs):
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
class MarketEnvMetadataWrapper(AbstractMarketEnvMetadataWrapper):
    """
    A metadata wrapper acts as a hub for storing metadata about the
    environment during an episode. Other wrappers that need access to
    this metadata can use the decorator `@metadata` to create a pointer
    to the metadata wrapper. This pointer is accessible via the
    `market_metadata_wrapper` attribute of the augmented wrapper class.
    This this wrapper acts as a data hub for other wrappers, it
    duplicates some of the attributes of the underlying market env. This
    is done to avoid the need for wrappers to have access to the
    underlying market env in addition to the metadata wrapper.
    """

    def __init__(self, env: Env) -> None:

        super().__init__(env)

        self.cash = None
        self.asset_quantities = None

        self.longs = None
        self.shorts = None
        self.positions = None

        self.equity = None
        self.portfolio_value = None

        self.profit = None
        self.return_ = None
        self._progress = None

        return None

    @property
    def longs(self) -> float:
        """
        The current notional value of long positions held in the market
        environment.
        """
        long_mask = self.asset_quantities > 0

        self.longs = self.asset_quantities[long_mask] @ self.asset_prices[
            long_mask]

        return self._longs

    @property
    def shorts(self) -> float:
        """
        The current notional value of short positions held in the 
        market environment.
        """
        short_mask = self.asset_quantities < 0
        self.shorts = abs(
            self.asset_quantities[short_mask] @ self.asset_prices[short_mask])

        return self._shorts

    @property
    def positions(self) -> np.ndarray[float]:
        """
        The current positions (notional base currency value) of each
        asset held by the trader. The position of each asset is the
        quantity of each asset held times its price. Position of an
        asset is always positive.

        Returns:
        ----------
            positions 
                (np.ndarray[float]): The current positions of each asset
                held by the trader.
        """
        self._positions = np.abs(self.asset_quantities * self.asset_prices)
        return self._positions
    
    @property
    def portfolio_value(self) -> float:
        """
        The current portfolio value of the trader. this includes ab
        value of all positions, both long and short.
        """
        self._portfolio_value = sum(self.positions)
        return self._portfolio_value
    
    @property
    def equity(self) -> float:
        """
        The current equity of the trader. Note cash can be negative if
        the trader is in debt.
        """
        self._equity = self.longs + self.cash - self.shorts
        return self._equity

    @property
    def marginable_equity(self) -> float:
        """
        The current marginable equity of the trader. This is the equity
        that can be used to open new positions and acts similar to 
        available cash, due to marginability of the underlying assets.
        When trading assets the gross intial margin of the assets should
        not exceed the marginable equity.
        """
        non_marginable_longs = 0
        for asset, quantity, position in zip(
            self.assets, self.asset_quantities, self.positions):
            if not asset.marginable:
                non_marginable_longs += (
                    position if quantity > 0 else 0)

        marginable_equity = self.equity - non_marginable_longs
        return marginable_equity

    @property
    def maintenance_margin_requirement(self) -> float:
        """
        The current maintenance margin requirement of the trader. This
        is the minimum amount of equity that must be maintained in the
        account to avoid a margin call.
        """
        margin_required = 0
        for asset, position in zip(self.assets, self.positions):
            margin_required += asset.maintenance_margin * position
        return margin_required
    
    @property
    def initial_margin_requirement(self) -> float:
        """
        The current initial margin requirement of the trader. This is
        the minimum amount of equity that must be maintained in the
        account to avoid a margin call.
        """
        margin_required = 0
        for asset, position in zip(self.assets, self.positions):
            margin_required += asset.initial_margin * position
        return margin_required

    @property
    def excess_margin(self) -> float:
        """
        The current maintenance margin of the trader. Excess margin is
        the amount of marginable equity above the maintenance margin
        requirement. This amount in margin account behaves similar to
        available cash, due to marginability of the underlying assets.
        Usually initial margin is checked before opening a position and
        never checked again, but we ensure intial margin is not
        accommulated for in a convservative manner.
        """
        excess_margin = self.equity - max(
            self.maintenance_margin_requirement, self.initial_asset_quantities)
        return excess_margin
    
    @property
    def progress(self) -> float | str:
        """
        If the underlying market env is a `TrainMarketEnv`, this method returns
        the progress of the episode as a float in [0, 1]. If the underlying
        market env is a `TradeMarketEnv`, this method returns current date and
        time.

        Returns:
        ----------
            progress (float | str): 
                The progress of the episode as a percentage or the current date
                and time.
        """
        if isinstance(self.market_env, TradeMarketEnv):
            self._progress = datetime.now().strftime("%m/%d/%Y, %H:%M")
        else:
            self._progress = self.market_env.index / self.market_env.n_steps

        return self._progress

    def _cache_metadata(self) -> None:
        """
        Caches metadata during an episode. This method is called by the
        `reset` and `step` methods of the wrapper.
        """
        self.history['cash'].append(self.cash)
        self.history['equity'].append(self.equity)
        self.history['longs'].append(self.longs)
        self.history['shorts'].append(self.shorts)
        self.history['portfolio_value'].append(self.portfolio_value)

        return None


@market
@metadata
class ConsoleTearsheetRenderWrapper(Wrapper):
    """
    A wrapper that prints a tear sheet to console showing market env
    metadata.

    """

    def __init__(self, env: Env, verbosity: int = 20) -> None:
        """

        """

        super().__init__(env)
        self.verbosity = verbosity

        self._set_render_frequency()

        return None

    def _set_render_frequency(self, verbosity: int) -> None:

        if isinstance(self.market_env, TrainMarketEnv):
            self.render_every = (self.market_env.n_steps //
                                 self.verbosity if self.verbosity > 0 else
                                 self.market_env.n_steps)
        elif isinstance(self.market_env, TradeMarketEnv):
            self.render_every = 1

    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Reset the environment and the tear sheet index.
        """

        observation = self.env.reset()

        resolution = self.market_env.data_metadata.resolution
        start_date = self.market_env.data_metadata.start.date()
        end_date = self.market_env.data_metadata.end.date()
        days = (end_date - start_date).days

        logger.info('Episode: '
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
            actions (np.ndarray | Dict[str, np.ndarray]): The actions to
            be taken by the agent.

        Returns:
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
        initial_cash = self.market_metadata_wrapper.initial_cash

        equity_history = self.market_metadata_wrapper.history['equity']
        initial_equity = equity_history[0]
        equity = self.market_metadata_wrapper.equity

        progress = self.market_metadata_wrapper.progress

        portfolio_value = sum(self.market_metadata_wrapper.positions)
        cash = self.market_metadata_wrapper.cash

        longs = self.market_metadata_wrapper.longs
        shorts = self.market_metadata_wrapper.shorts

        # financial metrics
        profit = equity - initial_equity
        return_ = (equity - initial_equity) / initial_equity
        sharpe = sharpe_ratio(self.market_metadata_wrapper.history['equity'])

        metrics = [
            f'{progress:.0%}', f'{return_:.2%}', f'{sharpe:.4f}',
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