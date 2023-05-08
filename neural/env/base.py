from __future__ import annotations

from typing import Tuple, Dict, TYPE_CHECKING, Optional, Iterable
from abc import ABC, abstractmethod

import numpy as np
from gym import spaces, Env

from neural.common.constants import GLOBAL_DATA_TYPE
from neural.data.base import StaticDataFeeder

if TYPE_CHECKING:
    from neural.trade.alpaca import AbstractTrader


class AbstractMarketEnv(Env, ABC):
    """
    Abstract base class for market environments. In case a custom market
    environment is needed, it should inherit from this class. It
    preserves the logic of sequentially updating the environment based
    on resolution of the dataset and placing orders based on actions
    received from the agent.

    Subclasses should implement the following abstract methods:

    Parameters:
    -----------
    Env (Env): OpenAI Gym Environment. ABC (ABC): Abstract Base Class
    from the abc module.

    Methods:
    --------
    update_env():
        Abstract method for updating the market environment.
    construct_observation():
        Abstract method for constructing the market observation.
    place_orders(actions):
        Abstract method for placing orders in the market environment.

    Notes:
    ------
    Market environments should be designed to interact with trading
    algorithms or other agents in order to simulate the behavior of a
    real-world market. This base class defines the minimum interface
    required for an environment to be used for algorithmic trading.
    """

    @abstractmethod
    def update_env(self):
        """
        Abstract method for updating the market environment. Should be
        implemented by subclasses. This method should update the
        environment state by moving to the next time step and updating
        the environment internalvariables such as features, asset
        prices, holds, and equity.
        """

        raise NotImplementedError

    @abstractmethod
    def construct_observation(self):
        """
        Abstract method for constructing the market observation. Should
        be implemented by subclasses. This method should construct the
        current observation from the environment's state variables. The
        observation may include the current cash balance, asset
        quantities, holds (time steps an asset has been held), and
        features of the current time step.
        """

        raise NotImplementedError

    @abstractmethod
    def place_orders(self, actions: np.ndarray[float]):
        """
        Abstract method for placing orders in the market environment.
        Should be implemented by subclasses. This method should place
        orders on the assets based on the given actions.
        The exact effect on the environment is left to the
        implementation of the subclass.
        """

        raise NotImplementedError


class TrainMarketEnv(AbstractMarketEnv):
    """
    A bare metal market environment with no market logic. Natively
    allowes cash and asset quantities to be negative, accommodating
    short/margin trading by default. Use action wrappers to impose
    market logic such as margin account initial and maintenance margins.
    Note that by default high frequency trading is only feasible with a
    margin account. Cash accounts incur delays in depositing cash and
    settling trades that makes high frequency trading infeasible.
    Inherits from AbstractMarketEnv. This class is intended to be used
    for training agents. For trading, use TradeMarketEnv. This class is
    not intended to be used directly. Instead, use the pipes in
    neural.meta.env.pipe to augment the environment with additional
    features. This enables the environment to respect real market
    constraints such as short selling and margin trading.

    Attributes:
    -----------
        data_feeder: StaticDataFeeder
            The StaticDataFeeder instance providing the data to the
            environment
        initial_cash: float, optional
            The initial amount of cash to allocate to the environment.
            Default is 1e6.
        initial_asset_quantities: np.ndarray, optional
            The initial quantity of assets to allocate to the
            environment. Default is None.
        data_metadata: DatasetMetadata
            Metadata about the dataset used
        feature_schema: Dict[FeatureType, List[bool]] 
            A dictionary mapping feature types to their respective
            boolean masks. Used to extract features from the dataset.
        asset_price_mask: List[bool]
            A list of boolean values representing the asset price
            features in the dataset.
        n_steps: int
            The number of steps in the environment
        n_features: int
            The number of features in the dataset
        n_assets: int
            The number of assets in the dataset
        index: int
            The current index of the environment
        cash: float
            The current amount of cash in the environment
        equity: float
            The current net worth of the environment
        asset_quantities: np.ndarray
            An array representing the quantities of each asset held by
            the environment. A positive value means the asset is held
            long, while a negative value means the asset is held short.
            Asset quantities can be fractional, allowing for partial
            shares, or integer, allowing for only whole shares.
        holds: np.ndarray
            An integer array representing the number of steps each asset
            has been held by the environment. As soon as a trade is
            placed, the holds for that asset is reset to zero.
        features: np.ndarray
            An array representing the current features of the
            environment. Features can include a wide swath of possible
            data, including market data such as open, high, low, close,
            volume, and other features such as sentiment scores, text
            embeddings, or raw text.
        _asset_prices: np.ndarray
            An array representing the current asset prices of the
            environment.
        info: dict
            A dictionary for storing additional information (unused)
        action_space: gym.spaces.Box. Number of actions is equal to
            the number of assets in the dataset. Each action is the
            notional value of the asset to buy or sell. A zero value
            means no action is taken (hold). Buys are represented as
            positive values, while sells are represented as negative
            values. Notional value means the face value of the asset
            (e.g. 100 means buy 100 dollars worth of the asset, while
            -100 means sell 100 dollars worth of the asset, given
            currency is USD).
        observation_space: gym.spaces.Dict. The observation space is a
        dictionary containing the following keys:
            - 'cash' (numpy.ndarray): A numpy array representing the
            available cash in the account. - 'asset_quantities'
            (numpy.ndarray): A numpy array representing the quantities
            of assets held. - 'holds' (numpy.ndarray): A numpy array
            representing the number of consecutive steps an asset has
            been held. - 'features' (numpy.ndarray): A numpy array
            representing the current features of the environment. This
            can including market data such as open, high, low, close,
            volume, and other features such as sentiment scores, text
            embeddings, or raw text.

    Methods:
    --------
    update_env():
        Updates the environment state by moving to the next time step
        and updating the environment variables such as features, asset
        prices, holds, and net worth.
    construct_observation():
        Constructs the current observation from the environment's state
        variables. The observation includes the current cash balance,
        asset quantities, holds (time steps an asset has been held), and
        features of the current time step.
    place_orders(actions):
        Places orders on the assets based on the given actions. Updates
        internal variables such as cash, asset quantities, and holds
        accordingly.
    reset():
        Resets the market environment to its initial state. Sets initial
        values for cash, asset quantities, and holds. Returns the
        initial observation. This is consistent with gym.Env.reset()
    step(actions):
        Executes a step in the trading environment. Updates the market
        environment state by moving to the next time step and updating
        the environment variables such as features, asset prices, holds,
        and net worth. This is consistent with gym.Env.step() interface.

    Examples:
    ---------
    >>> from neural.data.base import StaticDataFeeder
    >>> from neural.env.base import TrainMarketEnv
    >>> data_feeder = StaticDataFeeder(path='data.h5')
    >>> env = TrainMarketEnv(data_feeder=data_feeder)
    """

    def __init__(
        self,
        data_feeder: StaticDataFeeder,
        initial_cash: float = 1e6,
        initial_asset_quantities: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the TrainMarketEnv class.

        Args:
        -----------
        data_feeder: StaticDataFeeder
            The StaticDataFeeder instance providing the data to the
            environment
        initial_cash: float, optional
            The initial amount of cash to allocate to the environment.
            Default is 1e6.
        initial_asset_quantities: np.ndarray, optional
            The initial quantity of assets to allocate to the
            environment. Default is None.
        """

        self.data_feeder = data_feeder
        self.initial_cash = initial_cash
        self.initial_asset_quantities = initial_asset_quantities

        self.data_metadata = self.data_feeder.dataset_metadata
        self.feature_schema = self.data_metadata.feature_schema
        self.asset_price_mask = self.data_metadata.asset_price_mask
        self.n_steps = self.data_metadata.n_rows
        self.n_features = self.data_metadata.n_columns
        self.n_assets = len(self.data_metadata.assets)

        self.index = None
        self.cash = None
        self.equity = None
        self.asset_quantities = None
        self.holds = None
        self.features = None
        self._asset_prices = None
        self.info = None

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets, ),
            dtype=GLOBAL_DATA_TYPE,
        )

        self.observation_space = spaces.Dict({
            "cash":
            spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(1, ),
                       dtype=GLOBAL_DATA_TYPE),
            "asset_quantities":
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_assets, ),
                dtype=GLOBAL_DATA_TYPE,
            ),
            "holds":
            spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.n_assets, ),
                dtype=GLOBAL_DATA_TYPE,
            ),
            "features":
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.n_features, ),
                dtype=GLOBAL_DATA_TYPE,
            ),
        })

        return None

    @property
    def asset_prices(self) -> np.ndarray:
        """
        An array representing the current asset prices of the
        environment.
        
        Returns:
        --------
            asset_prices: np.ndarray
                An array representing the current asset prices of the
                environment.
        """
        self._asset_prices = self.features[self.asset_price_mask]

        return self._asset_prices

    def update_env(self) -> None:
        """
        Updates the environment state by moving to the next time step and updating
        the environment variables such as features, holds, and equity.

        Returns:
            None
        """

        self.features = next(self.row_generator)

        self.index += 1
        self.holds[self.asset_quantities != 0] += 1

        return None

    def construct_observation(self) -> Dict[str, np.ndarray[float]]:
        """
        Constructs the current observation from the environment's state
        variables. The observation includes the current cash balance,
        asset quantities, holds (time steps an asset has been held), and
        features of the current time step.

        Returns:
        -------
        observation: Dict[str, np.ndarray[float]]
            The observation dictionary containing the current cash
            balance, asset quantities, holds, and features.
        """

        observation = {
            "cash": self.cash,
            "asset_quantities": self.asset_quantities,
            "holds": self.holds,
            "features": self.features,
        }

        return observation

    def place_orders(self, actions: np.ndarray[float]) -> None:
        """
        Places orders on the assets based on the given actions. Orders
        are placed sequentially, meaning that the order of the assets
        matters. For example, if the agent wants to buy 100 dollars

        Args: - actions: Iterable containing the actions to take for
        each asset.

        Returns: - None

        Action values correspond to the notional value of asset to buy
        or sell. A zero value means no action is taken. Buys are
        represented as positive values, while sells are represented as
        negative values.
        """

        for asset, action in enumerate(actions):
            if action == 0:
                continue

            quantity = action / self.asset_prices[asset]

            self.asset_quantities[asset] += quantity
            self.cash -= action
            self.holds[asset] = 0

        return None

    def reset(self) -> Dict:
        """
        Resets the market environment to its initial state.

        Returns:
        -------
        Dict:
            A dictionary representing the initial observation space of
            the market environment containing the following keys: -
            'cash' (numpy.ndarray): A numpy array representing the
            available cash in the account. - 'asset_quantities'
            (numpy.ndarray): A numpy array representing the quantities
            of assets held. - 'holds' (numpy.ndarray): A numpy array
            representing the number of consecutive steps an asset has
            been held. - 'features' (numpy.ndarray): A numpy array
            representing the current features of the environment.
        """

        self.row_generator = self.data_feeder.get_row_generator()

        self.index = -1
        self.cash = np.array([self.initial_cash], dtype=GLOBAL_DATA_TYPE)

        self.asset_quantities = (np.zeros(
            (self.n_assets, ), dtype=GLOBAL_DATA_TYPE)
                                 if self.initial_asset_quantities is None else
                                 self.initial_asset_quantities)

        self.holds = np.zeros((self.n_assets, ), dtype=GLOBAL_DATA_TYPE)

        self.update_env()

        observation = self.construct_observation()

        return observation

    def step(
        self, actions: np.ndarray[float]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute a step in the trading environment.

        Parameters
        ----------
        actions : Iterable[float]
            The actions to be taken for each asset in the current step.

        Returns
        -------
        Tuple[Dict, float, bool, Dict]
            observation : dict
                A dictionary containing the current observation.
            reward : float
                The reward achieved in the current step.
            done : bool
                A boolean value indicating whether the current episode
                is finished.
            info : dict
                Additional information related to the current step.

        Notes
        -----
        `actions` are represented as the notional value of assets to buy
        or sell. A zero value means no action is taken. Buys are
        represented as positive values, while sells are represented as
        negative values. action = 200 means buy 200 dollars worth of the
        asset, while action = -200 means sell 200 dollars worth of the
        asset, given currncy is USD.

        The `step()` method updates the environment state by moving to
        the next time step and updating the environment variables such
        as features, asset prices, holds, and net worth. It then places
        orders on the assets based on the given actions, and updates the
        environment state again. Finally, it constructs the current
        observation from the environment's state variables and computes
        the reward. If the current episode is finished, it sets `done`
        to `True`.

        """

        net_worth_ = self.equity
        self.place_orders(actions)

        self.update_env()

        reward = self.equity - net_worth_
        observation = self.construct_observation()

        # report terminal state
        done = self.index == self.n_steps - 1

        return observation, reward, done, self.info


class TradeMarketEnv(TrainMarketEnv):
    """
    A market environment used for trading. Inherits from
    AbstractMarketEnv.

    Attributes: - trader (AbstractTrader): The trader that will be using
    this environment - dataset_metadata (DatasetMetadata): Metadata
    about the dataset used - data_feeder (AsyncDataFeeder): The data
    feeder used to feed the environment 
    - column_schema (Dict[ColumnType, List[str]]): A dictionary mapping 
    column types to their column names 
    - asset_price_mask (List[str]): A list of column
    names representing the asset prices 
    - n_steps (float): The number of steps in the environment (set to
    infinity) 
    - n_features (int): The number of features in the dataset 
    - n_symbols (int): The number of symbols in the dataset 
    - initial_cash (float): The initial amount of
    cash for the trader - cash (float): The current amount of cash for
    the trader - net_worth (float): The current net worth of the trader
    - asset_quantities (np.ndarray): An array representing the
    quantities of each asset held by the trader - holds (np.ndarray): An
    array representing the number of steps each asset has been held by
    the trader - features (np.ndarray): An array representing the
    current features of the environment - info (dict): A dictionary for
    storing additional information (unused)

    Methods: - __init__(self, trader: AbstractTrader):
        Initializes the TradeMarketEnv object

    - update_env(self) -> None:
        Updates the environment with the latest information from the
        trader.

        This method updates the environment's cash, asset quantities,
        net worth, and holds to match those of the trader's. The
        features of the environment are also updated with the latest
        market data from the trader's dataset.

        Returns:
            None

    - construct_observation(self) -> Dict:
        Constructs an observation of the current environment state

        Returns: observation (Dict):
            A dictionary containing the current observation space.

    - place_orders(self, actions) -> None:
        Places orders based on the given actions

        Args:
            actions (np.ndarray):
                An array of actions to be taken for each asset in the
                environment.

        Returns:
            None

    - reset(self) -> Dict:
        Resets the environment to its initial state and returns an
        initial observation

        Returns:
            A dictionary containing the following items:
                * 'cash': A float representing the available cash in the
                  market environment.
                * 'asset_quantities': A numpy array representing the
                  number of assets held by the agent.
                * 'holds': A numpy array representing the number of
                  consecutive steps for which each asset has been held.
                * 'features': A numpy array representing the market data
                  used as features in the observation.

    - step(self, actions) -> Tuple[Dict, float, bool, Dict]:
        Performs a step in the environment given the given actions and
        returns an observation, reward, done flag, and additional
        information

        Args:
            actions (Iterable[float]):
                an iterable containing actions for each asset in the
                environment.

        Returns: - observation (Dict):
            a dictionary containing the current observation of the
            environment.
        - reward (float):
            the reward received from the last action.
        - done (bool):
            whether the episode has ended or not (always False for
            trading environments).
        - info (Dict):
            a dictionary containing additional information about the
            step.
    """

    def __init__(
        self,
        trader: AbstractTrader,
    ) -> None:
        """
        Initializes the TradeMarketEnv object.

        Parameters
        ----------
        trader : AbstractTrader
            The trader object that will use this environment
        """

        self.trader = trader
        self.trade_client = self.trader.trade_client

        data_feeder = self.trader.data_streamer
        data_metadata = self.trader.stream_metadata
        initial_cash = self.trader.cash
        initial_asset_quantities = self.trader.asset_quantities

        super().__init__(
            data_feeder=data_feeder,
            data_metadata=data_metadata,
            initial_cash=initial_cash,
            initial_asset_quantities=initial_asset_quantities,
        )

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

        # this step will take time equal to dataset resolution to aggregate
        # data stream
        self.features = next(self.row_generator)

        self.holds[self.asset_quantities != 0] += 1
        self.cash = self.trader.trade_client.cash
        self.asset_quantities = [
            self.trader.trade_client.asset_quantities.get(symbol, 0)
            for symbol in self.symbols
        ]

        self.net_worth = self.trader.net_worth

        return None

    def place_orders(self, actions: np.ndarray[float]) -> None:
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
        Resets the market environment to the initial state and returns
        the observation.

        Returns:
            A dictionary containing the following items:
                * 'cash': A float representing the available cash in the
                  market environment.
                * 'asset_quantities': A numpy array representing the
                  number of assets held by the agent.
                * 'holds': A numpy array representing the number of
                  consecutive steps for which each asset has been held.
                * 'features': A numpy array representing the market data
                  used as features in the observation.

        """

        self.row_generator = self.data_feeder.get_row_generator()

        self.holds = np.zeros((self.n_assets, ), dtype=GLOBAL_DATA_TYPE)

        self.update_env()

        # compute state
        self.observation = self.construct_observation()

        return self.observation

    def step(self, actions: Iterable[float]) -> Tuple[Dict, float, bool, Dict]:
        """
        Advances the environment one step by applying the given actions
        to the current state.

        Args: - actions (Iterable[float]): an iterable containing
        actions for each asset in the environment.
        Returns: - observation (Dict): a dictionary containing the
        current observation of the environment.
        - reward (float): the
        reward received from the last action.
        - done (bool): whether the
        episode has ended or not (always False for trading
        environments).
        - info (Dict): a dictionary containing additional
        information about the step.
        """

        net_worth_ = self.net_worth
        self.place_orders(actions)
        self.update_env()

        reward = self.net_worth - net_worth_
        observation = self.construct_observation()

        done = False

        return observation, reward, done, self.info
