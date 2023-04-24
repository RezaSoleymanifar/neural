from __future__ import annotations

from typing import Tuple, Dict, TYPE_CHECKING, Optional, Iterable

import numpy as np
from gym import spaces, Env

from abc import ABC, abstractmethod
from neural.core.data.ops import StaticDataFeeder, AsyncDataFeeder
from neural.core.data.enums import ColumnType

if TYPE_CHECKING:
    from neural.core.trade.ops import AbstractTrader



class AbstractMarketEnv(Env, ABC):

    """
    Abstract base class for market environments.

    Subclasses should implement the following abstract methods:
        - update_env(): update the state of the environment after taking actions.
        - construct_observation(): construct a new observation of the environment state.
        - place_orders(actions): execute orders in the environment based on given actions.

    Note:
    ----
    Market environments should be designed to interact with trading algorithms or other agents
    in order to simulate the behavior of a real-world market. This base class defines the minimum
    interface required for an environment to be used for algorithmic trading. 
    """

    @abstractmethod
    def update_env(self):

        """
        Abstract method for updating the market environment.
        Should be implemented by subclasses.
        """

        raise NotImplementedError
    

    @abstractmethod
    def construct_observation(self):

        """
        Abstract method for constructing the market observation.
        Should be implemented by subclasses.
        """

        raise NotImplementedError


    @abstractmethod
    def place_orders(self, actions):

        """
        Abstract method for placing orders in the market environment.
        Should be implemented by subclasses.
        """

        raise NotImplementedError



class TrainMarketEnv(AbstractMarketEnv):

    """A bare metal market environment with no market logic. Natively allowes cash and asset quantities to be negative, 
    accommodating short/margin trading by default. Use action wrappers to impose market logic.

    Parameters:
    -----------
    market_data_feeder: StaticDataFeeder
        The StaticDataFeeder instance providing the data to the environment
    initial_cash: float, optional
        The initial amount of cash to allocate to the environment. Default is 1e6.
    initial_asset_quantities: np.ndarray, optional
        The initial quantity of assets to allocate to the environment. Default is None.

    Attributes:
    -----------
    action_space: gym.spaces.Box
        The action space for the environment
    observation_space: gym.spaces.Dict
        The observation space for the environment
    data_feeder: StaticDataFeeder
        The StaticDataFeeder instance providing the data to the environment
    dataset_metadata: DatasetMetadata
        The metadata object describing the dataset
    column_schema: Dict[ColumnType, List[str]]
        A dictionary mapping column types to a list of column names
    asset_price_mask: List[str]
        A list of column names for the asset price data
    n_steps: int
        The number of timesteps in the dataset
    n_features: int
        The number of features in the dataset
    n_symbols: int
        The number of symbols in the dataset
    index: int
        The current index in the dataset
    initial_cash: float
        The initial amount of cash allocated to the environment
    initial_asset_quantities: np.ndarray
        The initial quantity of assets allocated to the environment
    cash: float
        The current amount of cash in the environment
    net_worth: float
        The current net worth of the environment
    asset_quantities: np.ndarray
        The current quantity of assets in the environment
    holds: np.ndarray
        The number of timesteps each asset has been held
    features: np.ndarray
        The current features in the environment
    asset_prices: np.ndarray
        The current asset prices in the environment
    info: Dict
        Additional information about the environment
    """

    def __init__(
        self, 
        market_data_feeder: StaticDataFeeder,
        initial_cash: float = 1e6,
        initial_asset_quantities: Optional[np.ndarray] = None
        ) -> None:

        self.data_feeder = market_data_feeder
        self.dataset_metadata = self.data_feeder.dataset_metadata
        self.column_schema = self.dataset_metadata.column_schema
        self.asset_price_mask = self.dataset_metadata.column_schema[ColumnType.CLOSE]
        self.n_steps = self.dataset_metadata.n_rows
        self.n_features = self.dataset_metadata.n_columns
        self.n_symbols = len(self.dataset_metadata.symbols)

        self.index = None
        self.initial_cash = initial_cash
        self.initial_asset_quantities = initial_asset_quantities
        self.cash = None
        self.net_worth = None
        self.asset_quantities = None
        self.holds = None # steps asset was held
        self.features = None
        self.asset_prices = None
        self.info = None

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(
            self.n_symbols,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'cash':spaces.Box(
            low=-np.inf, high=np.inf, shape = (1,), dtype=np.float32),

            'asset_quantities': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})
        
        return None
    

    def update_env(self) -> None:

        """
        Updates the environment state by moving to the next time step and updating 
        the environment variables such as features, asset prices, holds, and net worth.

        Returns:
            None
        """

        self.features = next(self.row_generator)
        self.asset_prices = self.features[self.asset_price_mask]

        self.index += 1
        self.holds[self.asset_quantities != 0] += 1
        self.net_worth = self.cash + self.asset_quantities @ self.asset_prices

        return None


    def construct_observation(self) -> Dict:

        """
        Constructs the current observation from the environment's state variables.
        The observation includes the current cash balance, asset quantities, holds (time steps an asset has been held),
        and features of the current time step.
        
        Returns:
        -------
        observation: dict
            The observation dictionary containing the current cash balance, asset quantities, holds, and features.
        """

        observation = {
            'cash': np.array([self.cash], dtype=np.float32),
            'asset_quantities': self.asset_quantities,
            'holds': self.holds,
            'features': self.features
        }
        
        return observation


    def place_orders(self, actions: Iterable[float]) -> None:

        """
        Places orders on the assets based on the given actions.
        
        Args:
        - actions: Iterable containing the actions to take for each asset.
        
        Returns:
        - None
        
        Action values correspond to the amount of asset to buy or sell. A zero value means no action is taken.
        Buys are represented as positive values, while sells are represented as negative values.
        """

        for asset, action in enumerate(actions):

            if action == 0:
                continue

            quantity = action/self.asset_prices[asset]

            self.asset_quantities[asset] += quantity
            self.cash -= action
            self.holds[asset] = 0
        
        return None
        


    def reset(self) -> Dict:

        """
        Resets the market environment to its initial state.

        Returns:
            observation (Dict): The initial observation space of the market environment containing:
                - 'cash': the available cash in the account.
                - 'asset_quantities': the quantities of assets held.
                - 'holds': the number of consecutive steps an asset has been held.
                - 'features': the current features of the environment.
        """

        self.row_generator = self.data_feeder.reset()

        self.index = -1
        self.holds = np.zeros((self.n_symbols,), dtype=np.float32)

        self.cash = self.initial_cash

        self.asset_quantities = (
            np.zeros((self.n_symbols,), dtype=np.float32)
            if self.initial_asset_quantities is None
            else self.initial_asset_quantities)

        self.update_env()

        observation = self.construct_observation()

        return observation


    def step(
        self, 
        actions: Iterable[float]
        ) -> Tuple[Dict, np.float32, bool, Dict]:

        """
        Execute a step in the trading environment.

        Args:
        - actions (numpy.ndarray): The actions to be taken for each asset in the current step.
        
        Returns:
        - observation (Dict): A dictionary containing the current observation.
        - reward (numpy.float32): The reward achieved in the current step.
        - done (bool): A boolean value indicating whether the current episode is finished.
        - info (Dict): Additional information related to the current step.

        """

        net_worth_ = self.net_worth
        self.place_orders(actions)


        short_ratio = self.asset_prices[self.asset_quantities < 0] @ self.asset_quantities[self.asset_quantities < 0] / self.net_worth



        self.update_env()
        
        reward = self.net_worth - net_worth_
        self.observation = self.construct_observation()

        # report terminal state
        done = self.index == self.n_steps - 1
            
        return self.observation, reward, done, self.info
        


class TradeMarketEnv(AbstractMarketEnv):

    """
    A market environment used for trading. Inherits from AbstractMarketEnv.

    Attributes:
    - trader (AbstractTrader): The trader that will be using this environment
    - dataset_metadata (DatasetMetadata): Metadata about the dataset used
    - data_feeder (AsyncDataFeeder): The data feeder used to feed the environment
    - column_schema (Dict[ColumnType, List[str]]): A dictionary mapping column types to their column names
    - asset_price_mask (List[str]): A list of column names representing the asset prices
    - n_steps (float): The number of steps in the environment (set to infinity)
    - n_features (int): The number of features in the dataset
    - n_symbols (int): The number of symbols in the dataset
    - initial_cash (float): The initial amount of cash for the trader
    - cash (float): The current amount of cash for the trader
    - net_worth (float): The current net worth of the trader
    - asset_quantities (np.ndarray): An array representing the quantities of each asset held by the trader
    - holds (np.ndarray): An array representing the number of steps each asset has been held by the trader
    - features (np.ndarray): An array representing the current features of the environment
    - info (dict): A dictionary for storing additional information (unused)

    Methods:
    - __init__(self, trader: AbstractTrader): Initializes the TradeMarketEnv object
    - update_env(self) -> None: Updates the environment
    - construct_observation(self) -> Dict: Constructs an observation of the current environment state
    - place_orders(self, actions) -> None: Places orders based on the given actions
    - reset(self) -> Dict: Resets the environment to its initial state and returns an initial observation
    - step(self, actions) -> Tuple[Dict, np.float32, bool, Dict]: Performs a step in the environment given the given actions and returns an observation, reward, done flag, and additional information
    """

    def __init__(
        self,
        trader: AbstractTrader
        ) -> None:
    
        self.trader = trader
        self.dataset_metadata = self.trader.dataset_metadata
        self.data_feeder = AsyncDataFeeder(self.dataset_metadata)

        self.column_schema = self.dataset_metadata.column_schema
        self.asset_price_mask = self.dataset_metadata.column_schema[ColumnType.CLOSE]
        self.n_steps = float('inf')
        self.n_features = self.dataset_metadata.n_columns
        self.n_symbols = len(self.dataset_metadata.symbols)

        self.initial_cash = self.trader.initial_cash
        self.cash = None
        self.net_worth = None
        self.asset_quantities = None
        self.holds = None
        self.features = None
        self.info = None

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(
            self.n_symbols,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'cash':spaces.Box(
            low=0, high=np.inf, shape = (1,), dtype=np.float32),

            'asset_quantities': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})
        
        return None
            

    def update_env(self) -> None:

        """
        Updates the environment with the latest information from the trader.

        This method updates the environment's cash, asset quantities, net worth, and
        holds to match those of the trader's. The features of the environment are
        also updated with the latest market data from the trader's dataset.

        Returns:
            None
        """

        # this step will take time equal to dataset resolution to aggregate data stream
        self.features = next(self.row_generator)
        self.holds[self.asset_quantities != 0] += 1

        self.cash = self.trader.cash
        self.asset_quantities = self.trader.asset_quantities
        self.net_worth = self.trader.net_worth

        return None


    def construct_observation(self) -> Dict:

        """
        Construct the observation space for the trading environment.
        The observation space is a dictionary containing the following key-value pairs:
        - 'cash': The amount of cash available to the trader.
        - 'asset_quantities': The quantities of assets currently held by the trader.
        - 'holds': The number of time steps for which the assets have been held by the trader.
        - 'features': The current features of the market data stream.
        
        Returns:
        observation (Dict): A dictionary containing the current observation space.
        """

        observation = {
            'cash': np.array([self.cash], dtype= np.float32),
            'asset_quantities': self.asset_quantities,
            'holds': self.holds,
            'features': self.features
        }

        return observation
    

    def place_orders(self, actions: Iterable[float]) -> None:

        """
        Places orders through the connected trader instance based on the provided actions.

        Args:
            actions (np.ndarray): An array of actions to be taken for each asset in the environment.

        Returns:
            None
        """

        self.trader.place_orders(actions)

        return None


    def reset(self) -> Dict:

        """
        Resets the market environment to the initial state and returns the observation.

        Returns:
            A dictionary containing the following items:
                * 'cash': A float representing the available cash in the market environment.
                * 'asset_quantities': A numpy array representing the number of assets held by the agent.
                * 'holds': A numpy array representing the number of consecutive steps for which each asset has been held.
                * 'features': A numpy array representing the market data used as features in the observation.

        """

        # resetting input state
        self.row_generator = self.data_feeder.reset()

        self.holds = np.zeros(
            (self.n_symbols,), dtype=np.float32)
        
        self.update_env()

        # compute state
        self.observation = self.construct_observation()

        return self.observation


    def step(
        self,
        actions: Iterable[float]
        ) -> Tuple[Dict, np.float32, bool, Dict]:

        """
        Advances the environment one step by applying the given actions to the current state.

        Args:
        - actions (Iterable[float]): an iterable containing actions for each asset in the environment.

        Returns:
        - observation (Dict): a dictionary containing the current observation of the environment.
        - reward (float): the reward received from the last action.
        - done (bool): whether the episode has ended or not (always False for trading environments).
        - info (Dict): a dictionary containing additional information about the step.
        """
        
        net_worth_ = self.net_worth
        self.place_orders(actions)
        self.update_env()

        reward = self.net_worth - net_worth_

        # returns states using env vars
        self.observation = self.construct_observation()


        return self.observation, reward, False, self.info
