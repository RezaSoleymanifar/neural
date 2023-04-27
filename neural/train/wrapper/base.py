from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Type, Dict, Tuple
from datetime import datetime

import numpy as np
from gym import Env, Wrapper

from neural.common.log import logger
from neural.common.exceptions import IncompatibleWrapperError
from neural.meta.env.base import AbstractMarketEnv, TrainMarketEnv, TradeMarketEnv

from neural.tools.base import sharpe_ratio, tabular_print



def market(wrapper_class: Type[Wrapper]) -> Type[Wrapper]:

    """
    A decorator that augments an existing Gym wrapper class by adding a
    pointer to the underlying market environment, if the base environment
    is a market environment.

    Parameters:
    wrapper_class (Type[Wrapper]): The base Gym wrapper class to be augmented.
                                   This should be a subclass of `gym.Wrapper`.

    Returns:
        Type[Wrapper]: A new wrapper class that augments the input wrapper class with
                    a pointer to the underlying market environment, if applicable.

    Class Attributes:
        market_env (AbstractMarketEnv): The market environment.

    Raises:
        TypeError: If the `wrapper_class` is not a subclass of `Wrapper`.
        IncompatibleWrapperError: If the unwrapped base environment 
        is not of type AbstractMarketEnv.

    Methods:
        __init__(self, env: Env, *args, **kwargs) -> None:
            Initializes the MarketEnvDependentWrapper instance.

        check_unwrapped(self, env: Env) -> AbstractMarketEnv:
            Checks if the unwrapped base env is a market env.

    """

    if not issubclass(wrapper_class, Wrapper):

        raise TypeError(
            f"{wrapper_class.__name__} must be a subclass of {Wrapper}")

    class MarketEnvDependentWrapper(wrapper_class):

        """
        A wrapper that creates a pointer to the base market environment and checks
        if the unwrapped base env is a market env. If the check fails, an error is raised.

        Parameters:
            env (gym.Env): The environment being wrapped.

        Raises:
            IncompatibleWrapperError: If the unwrapped base environment is 
            not of type AbstractMarketEnv.
        """

        def __init__(self, env: Env, *args, **kwargs) -> None:

            """
            Initializes the MarketEnvDependentWrapper instance.

            Parameters:
                env (gym.Env): The environment being wrapped.
                *args: Optional arguments to pass to the wrapper.
                **kwargs: Optional keyword arguments to pass to the wrapper.
            """

            self.market_env = self.check_unwrapped(env)
            super().__init__(env, *args, **kwargs)

        def check_unwrapped(self, env: Env) -> AbstractMarketEnv:

            """
            Checks if the unwrapped base env is a market env.

            Parameters:
                env (gym.Env): The environment being wrapped.

            Raises:
                IncompatibleWrapperError: If the unwrapped base environment
                is not of type AbstractMarketEnv.

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

        """
        Initialize the AbstractMarketEnvMetadataWrapper.

        Parameters
        ----------
        env : gym.Env
            The base environment being wrapped.

        Returns
        -------
        None

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
        Update metadata.

        This method should be implemented by subclasses to update metadata as necessary
        during an episode. The method takes an arbitrary number of arguments and keyword
        arguments, depending on the specific metadata that needs to be updated.

        Parameters
        ----------
        *args
            Arbitrary arguments to be passed to the method.
        **kwargs
            Arbitrary keyword arguments to be passed to the method.

        Returns
        -------
        None
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
        Update metadata.

        This method should be implemented by subclasses to update metadata as necessary
        during an episode. The method takes an arbitrary number of arguments and keyword
        arguments, depending on the specific metadata that needs to be updated.

        Parameters
        ----------
        *args
            Arbitrary arguments to be passed to the method.
        **kwargs
            Arbitrary keyword arguments to be passed to the method.

        Returns
        -------
        None
        """

        if isinstance(self.market_env, TrainMarketEnv):
            self._upadate_train_env_metadata()

        elif isinstance(self.market_env, TradeMarketEnv):
            self._update_trade_env_metadata()

        self._cache_metadata_history()

        return None


    def _cache_metadata_history(self) -> None:

        """
        Cache metadata.

        This method should be implemented by subclasses to cache metadata as necessary
        during an episode. The method takes an arbitrary number of arguments and keyword
        arguments, depending on the specific metadata that needs to be cached.

        Parameters
        ----------
        *args
            Arbitrary arguments to be passed to the method.
        **kwargs
            Arbitrary keyword arguments to be passed to the method.

        Returns
        -------
        None
        """

        self.history['net_worth'].append(self.net_worth)

        return None
    


@market
@metadata
class ConsoleTearsheetRenderWrapper(Wrapper):


    """
    A wrapper that prints a tear sheet to console showing trading metadata.

    Args:
    env (gym.Env): The environment to wrap.
    verbosity (int, optional): Controls the frequency at which the tear sheet is printed. Defaults to 20.
        Set to 1 if you want to only show the final performance in episode.
    """

    def __init__(
        self, env: Env,
        verbosity: int = 20
        ) -> None:

        """
        Initializes the ConsoleTearsheetRenderWrapper instance.

        Args:
            env (gym.Env): The environment to wrap.
            verbosity (int, optional): Controls the frequency at which the tear sheet is 
            printed. Defaults to 20. Set to 1 if you want to only show the final performance in episode.
        """

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
        
        """
        Takes a step in the environment and updates the tear sheet if necessary.

        Args:
            actions (np.ndarray | Dict[str, np.ndarray]): The actions to be taken by the agent.

        Returns:
            Tuple: A tuple containing the new observation, the reward obtained, a boolean indicating whether the
            episode has ended, and a dictionary containing additional information.
        """


        observation, reward, done, info = self.env.step(actions)
        self.index += 1

        if self.index % self.render_every == 0 or done:
            self.render()

        return observation, reward, done, info


    def render(self, mode='human') -> None:

        """
        Prints the trading metadata tear sheet to console.
        """

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
        sharpe = sharpe_ratio(
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



class ExperienceRecorderWrapper(Wrapper):

    """
    A wrapper that records experiences and writes them to an HDF5 file.

    This wrapper maintains a buffer of experiences and writes them to an HDF5 file
    when the buffer reaches a certain size. Each experience is a tuple containing
    the observation, the action taken, the reward obtained, and the new observation.

    Args:
        env (gym.Env): The environment to wrap.
        buffer_size (int): The maximum number of experiences to store in the buffer.
        path (str): The path to the HDF5 file.
    """

    def __init__(self, env: Env, buffer_size: int, path: str) -> None:
        super().__init__(env)

        self.buffer_size = buffer_size
        self.path = path

        self.buffer = deque(maxlen=buffer_size)
        self.hdf5_file = None


    def reset(self) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:

        """
        Resets the environment and returns the initial observation.

        Returns:
            observation (np.ndarray[float] | Dict[str, np.ndarray[float]]): The initial observation.
        """

        return self.env.reset()


    def step(self, action: np.ndarray[float] | Dict[str, np.ndarray[float]]) -> Tuple[
        np.ndarray[float] | Dict[str, np.ndarray[float]], float, bool, dict()]:

        """
        Performs a step in the environment and saves the experience.

        Args:
            action (np.ndarray[float] | Dict[str, np.ndarray[float]]): The action to be taken by the agent.

        Returns:
            Tuple[np.ndarray[float] | Dict[str, np.ndarray[float]], float, bool, dict()]: A tuple containing the new
            observation, the reward obtained, a boolean indicating whether the episode has ended, and a dictionary
            containing additional information.
        """

        observation, reward, done, info = self.env.step(action)

        experience = (self.env.observation_space.sample(), action, reward, observation)
        self.buffer.append(experience)

        if len(self.buffer) == self.buffer_size:
            self.write_to_hdf5()

        return observation, reward, done, info


    def close(self) -> None:

        """
        Closes the HDF5 file.
        """

        if self.hdf5_file is not None:
            self.write_to_hdf5()
            self.hdf5_file.close()
            self.hdf5_file = None


    def write_to_hdf5(self) -> None:

        """
        Writes the buffer of experiences to an HDF5 file.
        """

        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.path, 'w')

        if '/experiences' not in self.hdf5_file:
            self.hdf5_file.create_group('experiences')

        experiences = np.array(self.buffer)
        dataset = self.hdf5_file['experiences'].get('data')
        if dataset is None:
            self.hdf5_file.create_dataset('experiences/data', data=experiences, maxshape=(None, 4))
        else:
            dataset.resize((dataset.shape[0] + experiences.shape[0]), axis=0)
            dataset[-experiences.shape[0]:] = experiences

        self.buffer.clear()

