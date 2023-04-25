from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Type, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
from gym import (ActionWrapper, Env, Wrapper, 
    ObservationWrapper, RewardWrapper, spaces, Space)

from neural.common.log import logger
from neural.common.constants import ACCEPTED_OBSERVATION_TYPES, ACCEPTED_ACTION_TYPES, GLOBAL_DATA_TYPE
from neural.tools.misc import FillDeque, RunningMeanStandardDeviation
from neural.common.exceptions import IncompatibleWrapperError
from neural.meta.env.base import AbstractMarketEnv, TrainMarketEnv, TradeMarketEnv

from neural.tools.ops import get_sharpe_ratio, tabular_print



def validate_observation(wrapper: Wrapper, observation: np.ndarray | Dict[str, np.ndarray]) -> None:

    """
    Validate the observation type against a list of accepted types.
    
    Parameters:
    -----------
    observation : np.ndarray[float] | Dict[str, np.ndarray[float]]
        The observation to validate.
    accepted_types : List[type]
        A list of accepted types.
        
    Returns:
    --------
    bool
        True if the observation type is accepted, False otherwise.
    """


    if isinstance(observation, dict):
        if all(isinstance(observation[key], np.ndarray) for key in observation):
            valid = True

    elif isinstance(observation, np.ndarray):
        valid = True

    if not valid:
        raise IncompatibleWrapperError(
            f'Wrapper {type(wrapper).__name__} received an observation of type {type(observation)}, '
            F'which is not in the accepted observation types {ACCEPTED_OBSERVATION_TYPES}.')
    
    return False


def validate_actions(wrapper: Wrapper, actions: np.ndarray | Dict[str, np.ndarray]) -> None:
    """
    Validate the observation type against a list of accepted types.
    
    Parameters:
    -----------
    observation : np.ndarray[float] | Dict[str, np.ndarray[float]]
        The observation to validate.
    accepted_types : List[type]
        A list of accepted types.
        
    Returns:
    --------
    bool
        True if the observation type is accepted, False otherwise.
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


def market(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:

    """
    A decorator that augments an existing Gym wrapper class by adding a
    pointer to the underlying market environment, if the base environment
    is a market environment.

    :param wrapper_class: The base Gym wrapper class to be augmented.
                        This should be a subclass of `gym.Wrapper`.
    :type wrapper_class: type[gym.Wrapper]

    :return: A new wrapper class that augments the input wrapper class with
            a pointer to the underlying market environment, if applicable.
    :rtype: type[gym.Wrapper]
    """

    if not issubclass(wrapper_class, Wrapper):

        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class MarketEnvDependentWrapper(wrapper_class):

        """
        A wrapper that creates a pointer to the base market environment and checks
        if the unwrapped base env is a market env. If the check fails, an error is raised.

        Args:
            env (gym.Env): The environment being wrapped.

        Raises:
            IncompatibleWrapperError: If the unwrapped base environment is not of 
            type AbstractMarketEnv.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:

            """
            Initializes the MarketEnvDependentWrapper instance.

            Args:
                env (gym.Env): The environment being wrapped.
                *args: Optional arguments to pass to the wrapper.
                **kwargs: Optional keyword arguments to pass to the wrapper.
            """

            self.market_env = self.check_unwrapped(env)
            super().__init__(env, *args, **kwargs)

        def check_unwrapped(self, env: Env) -> AbstractMarketEnv:

            """
            Checks if the unwrapped base env is a market env.

            Args:
                env (gym.Env): The environment being wrapped.

            Raises:
                IncompatibleWrapperError: If the unwrapped base environment is 
                not of type AbstractMarketEnv.

            Returns:
                AbstractMarketEnv: The unwrapped market env.
            """

            if not isinstance(env.unwrapped, AbstractMarketEnv):

                raise IncompatibleWrapperError(
                    f'{wrapper_class.__name__} requires unwrapped env '
                    'to be of type {AbstractMarketEnv}.')
            
            return env.unwrapped
        
    return MarketEnvDependentWrapper



def metadata(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:

    """
    a wrapper decorator that augments a wrapper class so that it checks if the unwrapped base env
    is a market env and creates a pointer to it. If the search fails, an error is raised.

    Args:
        wrapper_class (Type[Wrapper]): The wrapper class to be augmented.

    Returns:
        MarketEnvDependentWrapper: The augmented wrapper class.
    """

    if not issubclass(wrapper_class, Wrapper):

        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class MarketMetadataWrapperDependentWrapper(wrapper_class):

        """
        A class for checking if an unwrapped environment is a market environment 
        and creating a pointer to it.

        Args:
            env (Env): The environment to wrap.

        Raises:
            IncompatibleWrapperError: If the unwrapped environment is not a market environment.

        Attributes:
            market_env (AbstractMarketEnv): A pointer to the underlying market environment.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:

            """
            Initializes the MarketEnvMetadataWrapper dependent wrapper class.

            Args:
                env (gym.Env): The environment to be wrapped.
                *args: Variable length argument list to be passed to the wrapper.
                **kwargs: Arbitrary keyword arguments to be passed to the wrapper.
            """

            self.market_metadata_wrapper = self.find_metadata_wrapper(env)
            super().__init__(env, *args, **kwargs)

            return None


        def find_metadata_wrapper(self, env: Env) -> AbstractMarketEnvMetadataWrapper:

            """
            Recursively searches through wrapped environment to find the first instance
            of an AbstractMarketEnvMetadataWrapper in the wrapper stack.

            Parameters
            ----------
            env : gym.Env
                The environment to search through.

            Returns
            -------
            AbstractMarketEnvMetadataWrapper
                The first instance of AbstractMarketEnvMetadataWrapper found in the
                wrapper stack.
            
            Raises
            ------
            IncompatibleWrapperError
                If an AbstractMarketEnvMetadataWrapper is not found in the wrapper stack.
            """

            if isinstance(env, AbstractMarketEnvMetadataWrapper):
                return env

            elif hasattr(env, 'env'):
                return self.find_metadata_wrapper(env.env)
            
            else:
                raise IncompatibleWrapperError(
                f'{wrapper_class.__name__} requires a wrapper of type '
                f'{AbstractMarketEnvMetadataWrapper} in enclosed wrappers.')


    return MarketMetadataWrapperDependentWrapper



def buffer(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:

    """
    A decorator that augments an existing Gym wrapper class by recursively
    searching through enclosed wrappers for an observation buffer and creating
    a pointer to it, if available.

    :param wrapper_class: The base Gym wrapper class to be augmented. This
                          should be a subclass of `gym.Wrapper`.
    :type wrapper_class: type[gym.Wrapper]

    :return: A new wrapper class that augments the input wrapper class by
             creating a pointer to any observation buffer found in the enclosed
             wrappers, if applicable.
    :rtype: type[gym.Wrapper]
    """

    if not issubclass(wrapper_class, Wrapper):
        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class ObservationBufferDependentWrapper(wrapper_class):

        """
        A wrapper that searches recursively through enclosed wrappers for an
        observation buffer and creates a pointer to it. If search fails, an error
        is raised.

        Args:
            env (gym.Env): The environment being wrapped.

        Raises:
            IncompatibleWrapperError: If no observation buffer is found in
            any of the enclosed wrappers.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:

            """
            Initializes the ObservationBufferDependentWrapper instance.

            Args:
                env (gym.Env): The environment being wrapped.
                *args: Optional arguments to pass to the wrapper.
                **kwargs: Optional keyword arguments to pass to the wrapper.
            """

            self.observation_buffer_wrapper = self.find_observation_buffer_wrapper(env)
            super().__init__(env, *args, **kwargs)


        def find_observation_buffer_wrapper(self, env: Env) -> ObservationBufferWrapper:
            """
            Searches recursively for an observation buffer wrapper in enclosed wrappers.

            Args:
                env (gym.Env): The environment being wrapped.

            Raises:
                IncompatibleWrapperError: If no observation buffer is found in
                any of the enclosed wrappers.

            Returns:
                ObservationBuffer: The first observation buffer found.
            """

            if isinstance(env, ObservationBufferWrapper):
                return env

            if hasattr(env, 'env'):
                return self.find_observation_buffer_wrapper(env.env)

            else:
                raise IncompatibleWrapperError(
                    f'{wrapper_class.__name__} requires an observation buffer in one of '
                    'the enclosed wrappers.')

    return ObservationBufferDependentWrapper


def action(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:

    """
    A decorator that augments an existing Gym wrapper to 
    sanity check the actions being passed to it.

    :param wrapper_class: The base Gym wrapper class to be augmented. This
                          should be a subclass of `gym.Wrapper`.
    :type wrapper_class: type[gym.Wrapper]

    :raises TypeError: If the `wrapper_class` argument is not a subclass of
                       `gym.Wrapper`.

    :return: A new wrapper class that checks if an action is in the action space
             before calling the step function of the base class.
    :rtype: type[gym.Wrapper]
    """

    if not issubclass(wrapper_class, Wrapper):
        raise TypeError(f"{wrapper_class.__name__} must be a subclass of {Wrapper}")


    class ActionSpaceCheckerWrapper(wrapper_class):

        """
        A wrapper that checks if an action is in the action space before calling
        the step function of the base class.

        Args:
            env (gym.Env): The environment being wrapped.

        Raises:
            IncompatibleWrapperError: If the action is not in the action space.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:
            """
            Initializes the ActionSpaceCheckerWrapper instance.

            Args:
                env (gym.Env): The environment being wrapped.
                *args: Optional arguments to pass to the wrapper.
                **kwargs: Optional keyword arguments to pass to the wrapper.
            """

            super().__init__(env, *args, **kwargs)

            if not hasattr(self, 'action_space') or not isinstance(self.action_space, Space):
                
                raise IncompatibleWrapperError(
                    f"Applying {observation} decorator to{wrapper_class.__name__} "
                    "requires a non None action space of type {Space} to be defined first.")

            self._validate_actions(self.action_space.sample())

            return None


        def _validate_actions(
            self, 
            actions: np.ndarray[float] | Dict[str, np.ndarray[float]]
            ) -> None:

            validate_actions(self, actions)
            if not self.action_space.contains(actions):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} received an action of type {type(actions)}, '
                    'which is not in the expected action space {self.action_space}.')

            return None
        

        def step(self, actions: np.ndarray[float]):

            """
            Checks if the action is in the action space before calling the step function
            of the base class.

            Args:
                action: The action to take.

            Raises:
                IncompatibleWrapperError: If the action is not in the action space.

            Returns:
                The result of calling the step function of the base class.
            """

            self._validate_actions(actions)
            return super().step(actions)

    return ActionSpaceCheckerWrapper



def observation(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:

    """
    A decorator that augments an existing Gym wrapper class by checking if an
    observation is in it's expected observation space before returning it from the reset
    and step functions.

    :param wrapper_class: The base Gym wrapper class to be augmented. This should
                          be a subclass of `gym.Wrapper`.
    :type wrapper_class: type[gym.Wrapper]

    :raises TypeError: If the `wrapper_class` argument is not a subclass of
                       `gym.Wrapper`.
    :raises IncompatibleWrapperError: If the observation space is not defined in
                                      the enclosed environment, or if the expected
                                      observation type is not valid.

    :return: A new wrapper class that checks if an observation is in the observation
             space before returning it from the reset and step functions.
    :rtype: type[gym.Wrapper]
    """

    if not issubclass(wrapper_class, Wrapper):
        raise TypeError(f"{wrapper_class.__name__} must be a subclass of {Wrapper}")


    class ObservationSpaceCheckerWrapper(wrapper_class):

        """
        A wrapper that checks if an observation is in the observation space before returning it
        from the reset and step functions.

        Args:
            env (gym.Env): The environment being wrapped.

        Raises:
            IncompatibleWrapperError: If the observation is not in the observation space.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:

            """
            Initializes the ObservationSpaceCheckerWrapper instance.

            Args:
                env (gym.Env): The environment being wrapped.
                *args: Optional arguments to pass to the wrapper.
                **kwargs: Optional keyword arguments to pass to the wrapper.
            """

            # Bellow also automatically guarantees that valid observation space is defined
            # due to inheritance from gym.Wrapper
            if not hasattr(
                env, 'observation_space') or not isinstance(env.observation_space, Space):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} requires a non None observation '
                    f'space of type {Space} to be defined in the enclosed environment '
                    f'{type(env).__name__}.')
            
            super().__init__(env, *args, **kwargs)

            self._validate_expected_observation_type()
            self._validate_observation_in_expected_observation_type(
                env.observation_space.sample())

            return None
        

        def _validate_expected_observation_type(self):

            if (not hasattr(self, 'expected_observation_type') or
                    self.expected_observation_type is None):

                raise IncompatibleWrapperError(
                    f'Wrapper {type(self).__name__} needs to have a '
                    f'non None expected observation type '
                    f'defined before applying {observation} decorator.')
            
            valid = False
            # expected observation type is Space or subset list of [np.ndarray, dict]
            if isinstance(self.expected_observation_type, Space):
                valid = True

            elif isinstance(self.expected_observation_type, list
                ) and set(self.expected_observation_type).issubset(ACCEPTED_OBSERVATION_TYPES):
                valid = True
            
            if not valid:

                raise IncompatibleWrapperError(
                f'Wrapper {type(self).__name__} is defining an exptected observation type '
                f'of type {type(self.expected_observation_type)}, which is not in the accepted '
                'observation types {ACCEPTED_OBSERVATION_TYPES}.')


        def _validate_observation_in_expected_observation_type(
            self, 
            observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
            ) -> None:
            
            validate_observation(self, observation)
            # when expected observation type is Space
            if isinstance(self.expected_observation_type, Space
                ) and not self.expected_observation_type.contains(observation):

                raise IncompatibleWrapperError(
                f'Wrapper {type(self).__name__} received an observation of type {type(observation)}, '
                'which is not in the expected observation space {self.exptected_observation_space}.')
            
            return None


        def _validate_observation_in_observation_space(
            self,
            observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
            ) -> None:

            """
            Checks if the observation is in the observation space.

            Args:
                observation: The observation to check.

            Returns:
                True if the observation is in the observation space, False otherwise.
            """

            validate_observation(self, observation)
            if not self.self.observation_space.contains(observation):

                raise IncompatibleWrapperError(f'Wrapper {type(self).__name__} outputs an observation '
                    f'that is not in its defined observation space {self.exptected_observation_space}.')
            
            return None


        def observation(
            self, 
            observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
            ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:

            """
            Checks if the observation is in the observation space before returning it
            from the observation method of the base class.

            Raises:
                IncompatibleWrapperError: If the observation is not in the observation space.

            Returns:
                The result of calling the observation method of the base class.
            """

            self._validate_observation_in_expected_observation_type(observation)
            observation = super().observation(observation)
            self._validate_observation_in_observation_space(observation)

            return 


    return ObservationSpaceCheckerWrapper



class AbstractMarketEnvMetadataWrapper(Wrapper, ABC):

    """
    A blueprint class for market metadata wrappers.

    This class is designed to be subclassed for creating custom metadata wrappers
    for market environments. Metadata wrappers are used to keep track of additional
    information about the environment during an episode, such as cash balance, asset
    quantities, net worth, and more.

    Attributes:
        history (defaultdict): A defaultdict object for storing metadata during an episode.
        initial_cash (float): The initial amount of cash available in the environment.
        n_symbols (int): The number of symbols available for trading in the environment.
        cash (float): The current amount of cash available in the environment.
        asset_quantities (np.ndarray): An array of asset quantities held in the environment.
        positions (np.ndarray): An array of total asset positions in the environment.
        net_wroth (float): The current net worth of the portfolio in the environment.
        portfolio_value (float): The total value of the portfolio in the environment.
        longs (float): The total value of long positions in the portfolio.
        shorts (float): The total value of short positions in the portfolio.
        profit (float): The profit or loss made in the current episode.
        return_ (float): The rate of return made in the current episode.
        progress (int): A counter for tracking the number of episodes completed.

    Methods:
        update_metadata: An abstract method for updating metadata.
        _cache_metadata_history: An abstract method for caching metadata.
        reset: Resets the environment and updates metadata.
        step: Advances the environment by one step and updates metadata.

    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.history = defaultdict(list)
        self.initial_cash = self.market_env.initial_cash
        self.n_symbols = self.market_env.n_symbols
        self.cash = None
        self.asset_quantities = None
        self.positions = None
        self.net_wroth = None
        self.portfolio_value = None
        self.longs = None
        self.shorts = None
        self.profit = None
        self.return_ = None
        self.progress = None


    @abstractmethod
    def update_metadata(self, *args, **kwargs):


        """
        An abstract method for updating metadata.

        This method should be implemented by subclasses to update metadata as necessary
        during an episode. The method takes an arbitrary number of arguments and keyword
        arguments, depending on the specific metadata that needs to be updated.
        """

        raise NotImplementedError
        

    @abstractmethod
    def _cache_metadata_history(self, *args, **kwargs):

        """
        An abstract method for caching metadata.

        This method should be implemented by subclasses to cache metadata as necessary
        during an episode. The method takes an arbitrary number of arguments and keyword
        arguments, depending on the specific metadata that needs to be cached.
        """

        raise NotImplementedError


    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:

        """
        Resets the environment and updates metadata.

        This method resets the environment and updates the metadata stored in the
        wrapper. It returns the initial observation from the environment.

        Returns:
            observation (object): The initial observation from the environment.
        """

        observation = self.env.reset()
        self.update_metadata()

        return observation


    def step(self, action: np.ndarray[float] | Dict[str, np.ndarray[float]]):

        """
        Performs a step in the environment.

        Args:
            action (any): The action to be taken by the agent.

        Returns:
            Tuple: A tuple containing the new observation, the reward obtained,
            a boolean indicating whether the episode has ended, and a dictionary
            containing additional information.
        """

        observation, reward, done, info = self.env.step(action)
        self.update_metadata()

        return observation, reward, done, info



@market
class MarketEnvMetadataWrapper(AbstractMarketEnvMetadataWrapper):

    """
    A wrapper for market environments that tracks and stores environment metadata.
    Enclosing wrappers can utilize this metadata for various purposes, such as
    position sizing, risk management, and performance analysis.

    The wrapper supports both TrainMarketEnv and TradeMarketEnv instances and
    updates metadata accordingly.

    Attributes:
    asset_quantities (np.ndarray): The quantities of assets held by the trader.
    asset_prices (np.ndarray): The current asset prices in the market.
    net_worth (float): The current net worth of the trader.
    initial_cash (float): The initial cash amount when the environment started.
    cash (float): The current cash amount held by the trader.
    positions (np.ndarray): The current market value of all asset positions.
    longs (np.ndarray): The current market value of long asset positions.
    shorts (np.ndarray): The current market value of short asset positions.
    progress (float or str): The progress of the environment, either as a fraction
                              of steps for TrainMarketEnv or as a timestamp for
                              TradeMarketEnv.
    history (defaultdict): A dictionary containing the historical values of the
                           metadata attributes.

    """

    def __init__(self, env: Env) -> None:

        """
        Initializes the MarketEnvMetadataWrapper instance.

        Args:
        env (Env): The environment to wrap.
        """

        super().__init__(env)

        if isinstance(self.market_env, TradeMarketEnv):
            self.trader = self.market_env.trader
        
        return None


    def _upadate_train_env_metadata(self) -> None:

        """
        Updates the metadata attributes for TrainMarketEnv instances.
        """

        self.asset_quantities = self.market_env.asset_quantities
        self.asset_prices = self.market_env.asset_prices
        self.net_worth = self.market_env.net_worth
        self.initial_cash = self.market_env.initial_cash
        self.cash = self.market_env.cash
        self.positions = self.asset_quantities * self.asset_prices

        short_mask = self.asset_quantities < 0
        long_mask = self.asset_quantities > 0

        self.longs = self.asset_quantities[
            long_mask] @ self.asset_prices[long_mask]
        self.shorts = self.asset_quantities[
            short_mask] @ self.asset_prices[short_mask]
        
        self.progress = self.market_env.index/self.market_env.n_steps

        return None


    def _update_trade_env_metadata(self) -> None:

        """
        Updates the metadata attributes for TradeMarketEnv instances.
        """

        self.history = defaultdict(list)
        self.asset_quantities = self.trader.asset_quantities
        self.net_worth = self.trader.net_worth
        self.initial_cash = self.market_env.initial_cash
        self.cash = self.trader.cash

        self.positions = self.trader.positions
        self.longs = self.trader.longs
        self.shorts = self.trader.shorts

        now = datetime.now()
        self.progress = now.strftime("%Y-%m-%d %H:%M")

        return None


    def update_metadata(self) -> None:

        """
        Updates the metadata attributes based on the type of market environment
        (TrainMarketEnv or TradeMarketEnv), and caches the history.
        """

        if isinstance(self.market_env, TrainMarketEnv):
            self._upadate_train_env_metadata()

        elif isinstance(self.market_env, TradeMarketEnv):
            self._update_trade_env_metadata()

        self._cache_metadata_history()

        return None


    def _cache_metadata_history(self) -> None:

        """
        Caches the historical values of the metadata attributes.
        """

        self.history['net_worth'].append(self.net_worth)

        return None
    


@market
@metadata
class ConsoleTearsheetRenderWrapper(Wrapper):


    """A wrapper that prints a tear sheet to console showing trading metadata.

    Args:
    env (gym.Env): The environment to wrap.
    verbosity (int, optional): Controls the frequency at which the tear sheet is printed. Defaults to 20.
    Set to 1 if you want to only show the final performance in episode.
    """

    def __init__(
        self, env: Env,
        verbosity: int = 20
        ) -> None:

        super().__init__(env)
        self.verbosity = verbosity
        self.index = None

        if isinstance(self.market_env, TrainMarketEnv):
            self.render_every = self.market_env.n_steps//self.verbosity

        elif isinstance(self.market_env, TradeMarketEnv):
            self.render_every = 1 # every step
        
        return None


    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:    

        """
        Reset the environment and the tear sheet index.
        """

        observation = self.env.reset()
        self.index = 0

        resolution = self.market_env.dataset_metadata.resolution

        logger.info(
            'Episode details: '
            f'symbols = {self.market_env.n_symbols}, '
            f'resolution = {resolution}, '
            f'steps = {self.market_env.n_steps:,}, '
            f'features = {self.market_env.n_features:,}'
            )
        
        self.render()

        return observation
    

    def step(
        self, 
        actions: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> Tuple[np.ndarray[float] | Dict[str, np.ndarray[float]],
            float, bool, dict()]:

        """Take a step in the environment and update the tear sheet if necessary."""

        observation, reward, done, info = self.env.step(actions)
        self.index += 1

        if self.index % self.render_every == 0 or done:
            self.render()

        return observation, reward, done, info


    def render(self, mode='human') -> None:

        initial_cash = self.market_metadata_wrapper.initial_cash
        progress = self.market_metadata_wrapper.progress
        net_worth = self.market_metadata_wrapper.net_worth
        portfolio_value = sum(self.market_metadata_wrapper.positions)
        cash = self.market_metadata_wrapper.cash
        profit = self.market_metadata_wrapper.profit
        longs = self.market_metadata_wrapper.longs
        shorts = self.market_metadata_wrapper.shorts


        # financial metrics
        profit = net_worth - initial_cash
        return_ = (net_worth - initial_cash)/initial_cash
        sharpe = get_sharpe_ratio(
            self.market_metadata_wrapper.history['net_worth'])

        metrics = [
            f'{progress:.0%}',
            f'{return_:.2%}',
            f'{sharpe:.4f}',
            f'${profit:,.0f}',
            f'${net_worth:,.0f}',
            f'${cash:,.0f}',
            f'${portfolio_value:,.0f}',
            f'${longs:,.0f}',
            f'${shorts:,.0f} ' 
            f'{abs(shorts/net_worth):.0%}']

        if self.index == 0:

            title = [
                'Progress',
                'Return',
                'Sharpe ratio',
                'Profit',
                'Net worth',
                'Cash',
                'Portfolio value',
                'Longs',
                'Shorts']
            
            print(tabular_print(title, header=True))

        print(tabular_print(metrics))

        return None



@metadata
@observation
class MinTradeSizeActionWrapper(ActionWrapper):
    

    """
    A wrapper that limits the minimum trade size for all actions in the environment. 
    If the absolute value of any action is below min_action, it will be replaced with 0.
    Actions received are notional (USD) asset values.

    Args:
    env (gym.Env): The environment to wrap.
    min_action (float): The minimum trade size allowed in the environment. Default is 1.

    """

    def __init__(self, env: Env, min_trade = 1) -> None:

        super().__init__(env)

        assert min_trade >= 0, 'min_trade must be greater than or equal to 0'

        self.min_trade = min_trade
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = (
            spaces.Box(-np.inf, np.inf, shape=(self.n_symbols,), dtype= GLOBAL_DATA_TYPE))

        return None


    def action(
        self, 
        actions: np.ndarray[float]) -> np.ndarray[np.float]:

        for asset, action in actions:
            if abs(action) < self.min_trade:
                actions[asset] = 0
    
        return actions



@observation
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

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    integer : bool
        A flag that indicates whether to enforce integer asset quantities or not.
    asset_prices : ndarray or None
        An array containing the current prices of each asset in the environment, 
        or None if the prices have not been set yet.
    """

    # modifies actions to amount to integer number of shares
    # would not be valid in trade market env due to price
    # slippage.

    def __init__(self, env: Env, integer: bool=True) -> None:
        """
        Initializes a new instance of the IntegerAssetQuantityActionWrapper class.

        Parameters:
        -----------
        env : Env
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

        Parameters:
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

        return actions
    


@observation
@metadata
class NetWorthRelativeMaximumShortSizing(ActionWrapper):

    """
    This class is designed to size the maximum short amount relative to net worth, ensuring
    that the short to net worth ratio is not violated. Takes notional asset value to buy/sell
    as actions. It must be applied before a position sizing wrapper and should not be combined
    with wrappers that increase the sell amount.

    The short_ratio parameter determines the maximum allowed short position as a percentage
    of the net worth. For example, a short_ratio of 0.2 means the maximum short position can
    be 20% of the net worth. Actions are partially fulfilled to the extent that the short_ratio
    is not violated. Adherence to this ratio at the time orders are placed is guaranteed. 
    Note that since metadata wrapper updates after prices change, short ratio
    observed in render wrappers may occasionally seem to slightly deviate from the short_raio specified.
    This is due to the fact that metadata observed is not calculated at the previous price point.

    Attributes:
        short_ratio (float): Represents the maximum short ratio allowed. A value of 0 means no shorting,
                            while a value of 0.2 means the maximum short position can be 20% of net worth.
                            
    Methods:
        __init__(self, env: Env, short_ratio: float = 0.2) -> None: Constructor for the class.
        _set_short_budget(self) -> None: Sets the short budget based on net worth and short_ratio.
        action(self, actions) -> np.array: Processes and modifies actions to respect short sizing limits.
    """

    def __init__(self, env: Env, short_ratio: float = 0.2) -> None:

        """
        Initializes the NetWorthRelativeMaximumShortSizing class with the given environment and short_ratio.
        
        Args:
            env (Env): The trading environment.
            short_ratio (float): The maximum short ratio allowed. Default is 0.2 (20% of net worth).
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
            actions (np.ndarray): The actions to process.
        
        Returns:
            np.ndarray: The modified actions respecting the short sizing limits.
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
           
        return actions



@observation
@metadata
class FixedMarginActionWrapper(ActionWrapper):

    """
    Class for sizing maximum margin amount relative to net worth.
    Margin trading allows buying more than available cash using leverage. Positive cash is required
    thus margin trading can only happen one asset at a time since orders are submitted asset by asset.
    Initial_margin = 1 means no margin, namely entire purchase should me made with available cash only.
    leverage = 1/inital_margin. initial_margin = 0.1 means only 10% of purchase value needs to be present
    in account as cash thus maximum of 1/0.1 = 10 times cash, worth of assets can be purchased.
    overall margin will have nominal effect on portfolio performance unless large purchase
    of a single asset is involved.

    Attributes:
        short_ratio (float): Represents the maximum short ratio allowed. A value of 0 means no shorting,
                            while a value of 0.2 means the maximum short position can be 20% of net worth.
                            
    Methods:
        __init__(self, env: Env, short_ratio: float = 0.2) -> None: Constructor for the class.
        _set_short_budget(self) -> None: Sets the short budget based on net worth and short_ratio.
        action(self, actions) -> np.array: Processes and modifies actions to respect short sizing limits.
    """


    def __init__(self, env: Env, initial_margin: float = 1) -> None:

        """
        Initializes the class with the given trading environment and initial_margin.
        Args:
            env (Env): The trading environment.
            initial_margin (float): The initial margin required for each trade. Default is 1.
        """

        super().__init__(env)

        assert 0 < initial_margin <= 1, "Initial margin must be a float in (0, 1]."

        self.initial_margin = initial_margin
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape= (self.n_symbols,), dtype= GLOBAL_DATA_TYPE)

        return None


    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        """
        Processes the given actions without applying the effects, and proactively modifies actions that
        would lead to short sizing limit violations. It considers the available cash and the initial_margin
        to calculate the maximum amount that can be bought on margin.

        Args:
            actions (np.array): The actions to process.

        Returns:
            np.array: The modified actions respecting the margin requirements.

        Comments:
            1. Margin requires available cash. No cash means no margin.
            2. Sell actions are ignored due to having no effect on margin.        
        """



        cash = self.market_metadata_wrapper.cash

        for asset, action in enumerate(actions):
            
            if action <= 0:
                continue

            leverage = 1/self.initial_margin
            
            buy = min(action, leverage * cash)
            cash -= buy
            actions[asset] = buy

        return actions



@observation
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
            action_ (float): The action value to parse.

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
            actions (list): The actions to perform.

        Returns:
            list: The parsed actions.

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

    """A wrapper that enforces directional trading by zeroing either positive 
    action values (no long) or negative values (no short).
    Serves as an upstream action wrapper that modifies the actions 
    for downstream trade sizing wrappers.

    Args:
    env (gym.Env): The environment to wrap.
    long (bool): if True only long is allowed, otherwise only short.

    """

    def __init__(self, env: Env, long: bool = True) -> None:

        super().__init__(env)

        self.long = long
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.action_space = spaces.Box(-np.inf,
                                       np.inf, shape=(self.n_symbols,))

        return None

    def action(self, actions: np.ndarray[float]) -> np.ndarray[float]:

        """Modify the actions to enforce directional trading.
        
        Args:
        actions (np.ndarray): The array of actions to enforce directional trading.
        
        Returns:
        list: The modified array of actions.
        
        """
        if self.long:
            actions[actions < 0] = 0
        else:
            actions[actions > 0] = 0

        return actions
    


@metadata
@observation
class ActionClipperWrapper(ActionWrapper):


    """A wrapper that clips actions to expected range of downstream position sizing wrappers.
    Actions received are usually immediate output of model. Serves as an upstream action
    wrapper that enforces action values to be in a certain range for downstream trade sizing
    wrappers.

    Args:
    env (gym.Env): The environment to wrap.
    low (float, optional): The minimum value for the actions. Defaults to -1.
    high (float, optional): The maximum value for the actions. Defaults to 1.

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

        """Clip the actions to be within the given low and high values.
        
        Args:
        actions (np.ndarray): The array of actions to clip.
        
        Returns:
        list: The clipped array of actions.
        
        """

        for asset, action in actions:
            actions[asset] = np.clip(action, self.low, self.high)

        return actions
    

@observation
@metadata
class PositionsFeatureEngineeringWrapper(ObservationWrapper):

    """
    A wrapper for OpenAI Gym trading environments that augments observations such that,
    instead of asset quantities held, the notional USD value of assets (positions) is 
    used. This is useful to reflect the value of investments in each asset.

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    n_symbols : int
        The number of assets in the environment.
    n_features : int
        The number of additional features included in each observation after augmentation.
    """

    def __init__(self, env: Env) -> None:

        """
        Initializes a new instance of the PositionsFeatureEngineeringWrapper class.

        Parameters:
        -----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.n_features = self.market_metadata_wrapper.n_features
        self.expected_observation_type = spaces.Dict({

            'cash': spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),

            'asset_quantities': spaces.Box(
                low=-np.inf, high=np.inf, shape=(
                    self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
                low=0, high=np.inf, shape=(
                    self.n_symbols,), dtype=np.int32),

            'features': spaces.Box(
                low=-np.inf, high=np.inf, shape=(
                    self.n_features,), dtype=np.float32)})

        
        self.observation_space = spaces.Dict({

            'cash': spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),

            'positions': spaces.Box(
                low=-np.inf, high=np.inf, shape=(
                    self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
                low=0, high=np.inf, shape=(
                    self.n_symbols,), dtype=np.int32),

            'features': spaces.Box(
                low=-np.inf, high=np.inf, shape=(
                    self.n_features,), dtype=np.float32)})
        
        return None
        

    def observation(
        self, 
        observation: Dict[str, np.ndarray[float]]
        ) -> Dict[str, np.ndarray[float]]:

        """
        Augments the observation such that, instead of asset quantities held, the USD value of assets (positions) is used.

        Parameters:
        -----------
        observation : dict
            A dictionary containing the original observation.

        Returns:
        --------
        dict
            A dictionary containing the augmented observation, where the 'positions' key contains the USD value of each asset.
        """

        asset_prices= self.market_metadata_wrapper.asset_prices
        asset_quantities = observation.pop('asset_quantities')

        observation['positions'] = asset_prices * asset_quantities

        return observation
    


@observation
@metadata
class WealthAgnosticFeatureEngineeringWrapper(ObservationWrapper):

    """
    A wrapper for OpenAI Gym trading environments that augments observations 
    such that net worth sensitive features are now independent of net worth. 
    This wrapper should be applied immediately after the PositionsFeatureEngineeringWrapper.

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    initial_cash : float
        The initial amount of cash in the environment.
    n_symbols : int
        The number of assets in the environment.
    n_features : int
        The number of additional features included in each observation after augmentation.
    """

    def __init__(self, env: Env) -> None:

        """
        Initializes a new instance of the WealthAgnosticFeatureEngineeringWrapper class.

        Parameters:
        -----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)

        self.initial_cash = self.market_metadata_wrapper.initial_cash
        self.n_symbols = self.market_metadata_wrapper.n_symbols
        self.n_features = self.market_metadata_wrapper.n_features

        self.expected_observation_type = spaces.Dict({

            'cash':spaces.Box(
            low=-np.inf, high=np.inf, shape = (1,), dtype=np.float32),

            'positions': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.int32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})

        self.observation_space = spaces.Dict({

            'cash':spaces.Box(
            low=-np.inf, high=np.inf, shape = (1,), dtype=np.float32),

            'positions': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.int32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32),

            'return': spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),})
        
        return None
    

    def observation(self, observation: Dict[str, np.ndarray[float]]) -> Dict[str, np.ndarray[float]]:

        """
        Augments the observation such that net worth sensitive 
        features now have net worth independent values.

        Parameters:
        -----------
        observation : dict
            A dictionary containing the original observation.

        Returns:
        --------
        dict
            A dictionary containing the augmented observation, where the 
            'features' key contains net worth sensitive features that are now independent of net worth.
        """

        net_worth = self.market_metadata_wrapper.net_worth

        observation['positions'] /= net_worth
        observation['cash'] /= net_worth
        observation['return'] = net_worth/self.initial_cash - 1

        return observation



class ObservationBufferWrapper(ObservationWrapper):

    """
    A wrapper for OpenAI Gym trading environments that provides a temporary buffer of observations for subsequent wrappers 
    that require this form of information. Autofills itself with the first observation received from the environment to 
    maintain a constant buffer size at all times.

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    buffer_size : int
        The maximum number of observations to be stored in the buffer.
    observation_buffer : deque
        A deque object that stores the last n observations, where n is equal to the buffer_size.
    """


    def __init__(self, env: Env, buffer_size: int = 10) -> None:

        """
        Initializes a new instance of the ObservationBufferWrapper class.

        Parameters:
        -----------
        env : Env
            The trading environment to be wrapped.
        buffer_size : int, optional
            The maximum number of observations to be stored in the buffer. Defaults to 10.
        """

        super().__init__(env)
        
        assert buffer_size > 0, "The buffer size must be greater than 0."

        self.buffer_size = buffer_size
        self.observation_buffer = FillDeque(buffer_size=buffer_size)

        return None


    def reset(self):

        """
        Resets the environment and clears the observation buffer.
        """

        observation = self.env.reset()
        self.observation_buffer.clear()
        self.observation_buffer.append(observation)

        return observation
    

    def observation(self, observation):

        """
        Adds the observation to the buffer and returns the buffer as the new observation.

        Parameters:
        -----------
        observation : dict
            A dictionary containing the current observation.

        Returns:
        --------
        deque
            A deque object containing the last n observations, where n is equal to the buffer_size.
        """

        self.observation_buffer.append(observation)

        return observation
    


@observation
class FlattenToNUmpyObservationWrapper(ObservationWrapper):

    """
    A wrapper for OpenAI Gym trading environments that flattens the observation space to a 1D numpy array.

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    """

    def __init__(self, env: Env) -> None:

        """
        Initializes a new instance of the FlattenToNUmpyObservationWrapper class.

        Parameters:
        -----------
        env : Env
            The trading environment to be wrapped.
        """

        super().__init__(env)
        self.expected_observation_type = [dict, np.ndarray]
        self.observation_space = self.flattened_space(env.observation_space)


        return None


    def flattened_space(self, space: Space) -> spaces.Box:

        """
        Returns a flattened observation space.

        Parameters:
        -----------
        env : Env
            The trading environment.

        Returns:
        --------
        spaces.Box
            The flattened observation space.
        """        

        # self.observation_space at constructor is set equal to of self.env.observation_space
        # i.e. observation_spcae of enclosed wrapper due to inheritance from ObservationWrapper
        sample_observation = space.sample()

        if isinstance(sample_observation, np.ndarray):
            
            flattened_shape = sample_observation.flatten().shape
            flattened_low = space.low.flatten() if not np.isscalar(
                space.low) else np.full(flattened_shape, space.low)
            flattened_high = space.high.flatten() if not np.isscalar(
                space.high) else np.full(flattened_shape, space.high)

            flattened_observation_space = spaces.Box(
                low=flattened_low,
                high=flattened_high,
                shape=flattened_shape, 
                dtype=GLOBAL_DATA_TYPE)
            
            return flattened_observation_space

        elif isinstance(sample_observation, dict):

            flattened_size = 0
            flattened_observation_space  = dict()
            
            for key, space in space.items():
                flattened_observation_space[key] = self.flattened_space(space)

            flattened_low = np.concatenate(
                [flattened_observation_space[key].low for key in flattened_observation_space])

            flattened_high = np.concatenate(
                [flattened_observation_space[key].high for key in flattened_observation_space])
            
            flattened_size = sum(
                shape[0] for shape in flattened_observation_space.values())
            
            flattened_shape = (flattened_size, )

            return spaces.Box(
                low=flattened_low, 
                high=flattened_high, 
                shape=flattened_shape, 
                dtype=GLOBAL_DATA_TYPE)


    def observation(self, observation):

        """
        Flattens the observation space to a 1D numpy array.

        Parameters:
        -----------
        observation : dict or ndarray
            The observation space.

        Returns:
        --------
        ndarray
            The flattened observation space.
        """

        if isinstance(observation, dict):
            observation = np.concatenate([
                array.flatten() for array in observation.values()])
            
        elif isinstance(observation, np.ndarray):
            observation = observation.flatten()

        return observation



@buffer
class ObservationStackerWrapper(ObservationWrapper):

    """
    A wrapper for OpenAI Gym trading environments that stacks the last n observations in the buffer.
    If observation is changed between buffer and stacker, all changes will be lost as this wrapper's
    point of reference is the buffer.

    Attributes:
    -----------
    env : Env
        The trading environment to be wrapped.
    stack_size : int
        The number of observations to be concatenated.
    """

    def __init__(self, env: Env, stack_size: Optional[int] = None) -> None:

        """
        Initializes a new instance of the ObservationStackerWrapper class.

        Parameters:
        -----------
        env : Env
            The trading environment to be wrapped.
        stack_size : int, optional
            The number of observations to be concatenated. Defaults to 4.
        """

        super().__init__(env)

        buffer_size = self.observation_buffer_wrapper.buffer_size
        self.stack_size = (
            stack_size if stack_size is not None else buffer_size)
        
        assert (self.stack_size <= buffer_size, 
            f"Stack size {self.stack_size} cannot exceed buffer size {buffer_size}.")

        self.observation_space = None

        return None
    

    def infer_stacked_observation_space(
        self, 
        observation_space: spaces.Box
        ) -> spaces.Box:

        # TODO: Add support for other observation spaces
        assert isinstance(observation_space, spaces.Box), (
            f"currently {ObservationStackerWrapper.__name__} only supports Box observation spaces")

        observation = observation_space.sample()
        stacked_shape = np.stack([observation] * self.stack_size).shape

        stacked_low = (np.full(stacked_shape, observation_space.low) 
            if np.isscalar(observation_space.low) 
            else np.stack(self.stack_size * [observation_space.low]))
        
        stacked_high = (np.full(stacked_shape, observation_space.high) 
            if np.isscalar(observation_space.high) 
            else np.stack(self.stack_size * [observation_space.high]))

        return spaces.Box(
            low=stacked_low,
            high=stacked_high,
            shape=stacked_shape, 
            dtype=GLOBAL_DATA_TYPE)
        

    def stacked_observation_space(self) -> Space:

        """
        Returns a flattened observation space.

        Parameters:
        -----------
        env : Env
            The trading environment.

        Returns:
        --------
        spaces.Box
            The flattened observation space.
        """        

        # self.observation_space at constructor is set equal to of self.env.observation_space
        # i.e. observation_spcae of enclosed wrapper due to inheritance from ObservationWrapper

        buffer_observation_space = self.observation_buffer_wrapper.observation_space

        if isinstance(buffer_observation_space, spaces.Box):
            stacked_observation_space =  self.infer_stacked_observation_space(buffer_observation_space)
        
        elif isinstance(buffer_observation_space, dict):
            stacked_observation_space = dict()
            for key, space in buffer_observation_space.items():
                stacked_observation_space[key] = self.infer_stacked_observation_space(space)

        return stacked_observation_space
        

    def observation(self, observation: Dict[str, np.ndarray[float]] | np.ndarray[float]):

        """
        Returns the last n stacked observations in the buffer. Note observation in argument
        is discarded and only elemetns in buffer are used, thus if observation is changed
        between buffer and stacker, all changes will be lost as this wrapper's point of
        reference is the buffer.

        Parameters:
        -----------
        observation : dict or ndarray
            A dictionary or ndarray containing the current observation.

        Returns:
        --------
        ndarray or dict of ndarrays
            An ndarray or dict of ndarrays containing the stacked observations.
        """

        stack = self.observation_buffer_wrapper.observation_buffer[-self.stack_size:]
        
        # Check if the observations are ndarrays or dictionaries of ndarrays
        if isinstance(stack[0], np.ndarray):
            stacked_observation = np.stack(stack) # default axis=0

        elif isinstance(stack[0], dict):
            stacked_observation = {}
            for key in stack[0].keys():
                key_stack = [observation[key] for observation in stack]
                key_stack = np.stack(key_stack)
                stacked_observation[key] = key_stack
        
        return stacked_observation



@observation
class RunningMeanSandardDeviationObservationWrapper(ObservationWrapper):

    """
    Gym environment wrapper that tracks the running mean and standard deviation
    of the observations using the RunningMeanStandardDeviation class.

    Parameters:
        env: The environment to wrap.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.expected_observation_type = [dict, np.ndarray]
        self.observation_rms = self.initialize_observation_rms(
            env.observation_space.sample())


    def initialize_observation_rms(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        )-> RunningMeanStandardDeviation | Dict[str, RunningMeanStandardDeviation]: 

        if isinstance(observation, np.ndarray):
            observation_rms = RunningMeanStandardDeviation(observation.shape)

        elif isinstance(observation, dict):

            observation_rms = dict()
            for key, array in observation.items():
                observation_rms[key] = RunningMeanStandardDeviation(array.shape)
            
        return observation_rms
    

    def update(
        self, 
        observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> None:
        
        if isinstance(observation, np.ndarray):
            self.observation_rms.update(observation)

        elif isinstance(observation, dict):
            for key, array in observation.items():
                self.observation_rms[key].update(array)

        return None
    
    def observation(self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]):
        
        self.update(observation)

        return observation



@observation
class NormalizeObservationWrapper(RunningMeanSandardDeviationObservationWrapper):

    """
    Gym environment wrapper that normalizes observations using the running mean and standard deviation
    tracked by the RunningMeanStandardDeviation class.
    """

    def __init__(self, env: Env, epsilon: float = 1e-8, clip: float = 10) -> None:
        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
    

    def observation(
        self, observation: np.ndarray[float] | Dict[str, np.ndarray[float]]
        ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:

        observation = super().observation(observation)

        if isinstance(observation, np.ndarray):
            observation = self.observation_rms.normalize(
                observation, self.epsilon, self.clip).astype(GLOBAL_DATA_TYPE)

        elif isinstance(observation, dict):

            observation = dict()
            
            for key, rms in self.observation_rms.items():
                observation[key] = rms.normalize(
                    observation, self.epsilon, self.clip).astype(GLOBAL_DATA_TYPE)

        return observation



class NormalizeReward(RewardWrapper):

    """
    This wrapper will normalize immediate rewards. 
    Usage:
        env = NormalizeReward(env, epsilon=1e-8, clip_threshold=10)
    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    Methods:
        reward(reward: float) -> float
            Normalize the reward.
    """

    def __init__(
        self,
        env: Env,
        epsilon: float = 1e-8,
        clip_threshold: float = np.inf,
        ) -> None:

        """
        This wrapper normalizes immediate rewards so that rewards have mean 0 and standard deviation 1.

        Args:
            env (Env): The environment to apply the wrapper.
            epsilon (float, optional): A small constant to avoid divide-by-zero errors when normalizing data. Defaults to 1e-8.
            clip_threshold (float, optional): A value to clip normalized data to, to prevent outliers 
            from dominating the statistics. Defaults to np.inf.
        """

        super().__init__(env)

        self.reward_rms = RunningMeanStandardDeviation()
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold


    def reward(self, reward: float) -> float:

        """Normalize the reward.

        Args:
            reward (float): The immediate reward to normalize.

        Returns:
            float: The normalized reward.
        """

        self.reward_rms.update(reward)
        normalized_reward = self.reward_rms.normalize(reward, self.epsilon, self.clip_threshold)

        return normalized_reward



class AbstractRewardShaperWrapper(Wrapper, ABC):

    # highly useful for pretraining an agent with many degrees of freedom
    # in actions. Apply relevant reward shaping wrappers to def ine and restrict unwanted
    # actions.

    pass

class PenalizeShortRatioRewardWrapper(RewardWrapper):
    pass

class PenalizeCashRatioRewardWrapper(RewardWrapper):
    pass



@buffer
class FinancialIndicatorsWrapper(ObservationWrapper):

    # computes running financial indicators such as CCI, MACD
    # etc. Requires an observations buffer containing a window 
    # of consecutive observations.

    def observation(self, observation):
        return None



class ExperienceRecorderWrapper(Wrapper):

    # saves experiences in buffer and writes to hdf5 when 
    # buffer reaches a certain size.

    pass
