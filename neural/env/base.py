"""
base.py

This module defines the base classes for the market environment needed
for training and trading with deep reinforcement learning algorithms.
The base classes are designed to be used with the OpenAI Gym
interface.

Classes:
--------
    AbstractMarketEnv:
        Abstract base class for market environments. In case a custom
        market environment is needed, it should inherit from this
        class. It preserves the logic of sequentially updating the
        environment based on resolution of the dataset and placing
        orders based on actions received from the agent.
    TrainMarketEnv:
        A bare metal market environment with no market logic. Natively
        allowes cash and asset quantities to be negative, accommodating
        short/margin trading by default. Use action wrappers to impose
        market logic such as margin account initial and maintenance
        margins. Note that by default high frequency trading is only
        feasible with a margin account. Cash accounts incur delays in
        depositing cash and settling trades that makes high frequency
        trading infeasible.
    TradeMarketEnv:
        This is a subclass of TrainMarketEnv. It is intended to be used
        for trading. It is identical to TrainMarketEnv except that it
        is connected to a trader instance. This allows the environment
        to interact with the trader and place orders in the market.
        This class is not intended to be used directly. Instead, use
        the pipes in neural.meta.env.pipe to augment the environment
        with additional features. Typically the same pipe used for
        training is used for trading. Pipe is saved as an attribute of
        the agent that was used for training. The agent can then be
        loaded and used for trading using this environment.

Notes:
------
    Market environments should be designed to interact with trading
    algorithms or other agents in order to simulate the behavior of a
    real-world market. This module defines the minimum interface
    required for an environment to be used for high frequency trading.
"""

from __future__ import annotations

from typing import Tuple, Dict, TYPE_CHECKING, Optional
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
        the environment internal variables such as features, asset
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
    constraints for short selling and margin trading.

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
        assets: List[str]
            A list of asset names in the dataset.
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
        _cash: float
            The current amount of cash in the environment
        _asset_quantities: np.ndarray
            An array representing the quantities of each asset held by
            the environment. A positive value means the asset is held
            long, while a negative value means the asset is held short.
            Asset quantities can be fractional, allowing for partial
            shares, or integer, allowing for only whole shares.
        _asset_prices: np.ndarray
            An array representing the current asset prices of the
            environment.
        features_generator: Iterator[np.ndarray]
            An iterator that yields the next row of the dataset. This
            iterator is used to update the environment state by moving
            to the next time step and updating the environment
            variables.
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
        observation_space: gym.spaces.Dict. 
            The observation space is a dictionary containing the
            following keys:
                - 'cash' (numpy.ndarray): 
                    A numpy array representing the available cash in the
                    account. 
                - 'asset_quantities' (numpy.ndarray): 
                    A numpy array representing the quantities of assets
                    held. 
                - 'holds' (numpy.ndarray): 
                    A numpy array representing the number of consecutive
                    steps an asset has been held. 
                - 'features' (numpy.ndarray): 
                    A numpy array representing the current features of
                    the environment. This can including market data such
                    as open, high, low, close, volume, and other
                    features such as sentiment scores, text embeddings,
                    or raw text.

    Properties:
    -----------
        cash: float
            The current amount of cash in the environment.
        asset_quantities: np.ndarray
            An array representing the quantities of each asset held by
            the environment. A positive value means the asset is held
            long, while a negative value means the asset is held short.
            Asset quantities can be fractional, allowing for partial
            shares, or integer, allowing for only whole shares.
        asset_prices: np.ndarray
            An array representing the current asset prices of the
            environment.

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
        self.assets = self.data_metadata.assets

        self.n_steps = self.data_feeder.n_rows
        self.n_features = self.data_metadata.n_columns
        self.n_assets = len(self.assets)

        self.index = None
        self.holds = None
        self.features = None

        self._cash = None
        self._asset_quantities = None
        self._asset_prices = None

        self.features_generator = None
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
    def cash(self) -> float:
        """
        The current amount of cash in the environment.

        Returns:
        --------
            cash: float
                The current amount of cash in the environment.
        """
        return self._cash
    
    @property
    def asset_quantities(self) -> np.ndarray:
        """
        An array representing the quantities of each asset held by the
        environment. A positive value means the asset is held long,
        while a negative value means the asset is held short. Asset
        quantities can be fractional, allowing for partial shares, or
        integer, allowing for only whole shares.

        Returns:
        --------
            asset_quantities: 
                np.ndarray An array representing the quantities of each
                asset held by the environment.
        """
        return self._asset_quantities
    
    @property
    def asset_prices(self) -> np.ndarray:
        """
        An array representing the current asset prices of the
        environment.
        
        Returns:
        --------
            asset_prices: np.ndarray
                An array representing the current asset prices of the
                assets.
        """
        asset_price_mask = self.data_metadata.asset_price_mask
        self._asset_prices = self.features[asset_price_mask]

        return self._asset_prices


    def update_env(self) -> None:
        """
        Updates the environment state by moving to the next time step
        and updating the environment variables such as features, holds,
        and index.

        Returns:
            None
        """

        self.features = next(self.features_generator)

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
        Places orders on the assets based on the given actions. Updates
        internal variables such as cash, asset quantities, and holds
        accordingly.

        Args:
        ------
            actions: np.ndarray[float]
                An array of actions to be taken for each asset in the
                current step. If positive, the agent buys the asset. If
                negative, the agent sells the asset. If zero, the agent
                holds the asset. Value of action shows the notional
                value of asset to buy or sell, namely the face value of
                the asset (e.g. 100 means buy 100 dollars worth of the
                asset, while -100 means sell 100 dollars worth of the
                asset, given currency is USD).
        """
        for asset, action in enumerate(actions):
            if action == 0:
                continue

            quantity = action / self.asset_prices[asset]

            self._asset_quantities[asset] += quantity
            self._cash -= action
            self.holds[asset] = 0

        return None

    def reset(self) -> Dict[str, np.ndarray[float]]:
        """
        Resets the market environment to its initial state. Sets initial
        values for cash, asset quantities, and holds. Returns the
        initial observation. 

        Returns:
        --------
        observation: Dict[str, np.ndarray[float]]
            The initial observation dictionary containing the current
            cash balance, asset quantities, holds, and features.

        Notes:
        ------
            This is consistent with gym.Env.reset()
        """
        self.features_generator = self.data_feeder.get_features_generator()
        self.index = -1
        self.holds = np.zeros((self.n_assets, ), dtype=GLOBAL_DATA_TYPE)

        self._cash = np.array([self.initial_cash], dtype=GLOBAL_DATA_TYPE)
        self._asset_quantities = (np.zeros(
            (self.n_assets, ), dtype=GLOBAL_DATA_TYPE)
                                 if self.initial_asset_quantities is None
                                 else self.initial_asset_quantities)

        self.update_env()
        observation = self.construct_observation()

        return observation

    def step(
        self, actions: np.ndarray[float]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute a step in the trading environment.

        Args:
        ----------
        actions : Iterable[float]
            The actions to be taken for each asset in the current step.
            If positive, the agent buys the asset. If negative, the
            agent sells the asset. If zero, the agent holds the asset.
            Value of action shows the notional value of asset to buy or
            sell, namely the face value of the asset (e.g. 100 means buy
            100 dollars worth of the asset, while -100 means sell 100
            dollars worth of the asset, given currency is USD).

        Returns
        -------
        Tuple[Dict, float, bool, Dict]
            observation : dict
                A dictionary containing the current observation.
            reward : float
                The reward achieved in the current step. Reward is not 
                generated in env. Instead enclosing wrappers impolemnt
                the logic of reward generation. The output reward thus
                is None, which later will be modified by wrappers.
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
        as features, asset prices, and holds. It then places
        orders on the assets based on the given actions, and updates the
        environment state again. Finally, it constructs the current
        observation from the environment's state variables and computes
        the reward. If the current episode is finished, it sets `done`
        to `True`.

        """

        self.place_orders(actions)

        self.update_env()

        reward = None
        observation = self.construct_observation()

        # report terminal state
        done = self.index == self.n_steps - 1

        return observation, reward, done, self.info


class TradeMarketEnv(TrainMarketEnv):
    """
    This is a subclass of TrainMarketEnv. It is intended to be used for
    trading. It is identical to TrainMarketEnv except that it is
    connected to a trader instance. This allows the environment to
    interact with the trader and place orders in the market. This class
    is not intended to be used directly. Instead, use the pipes in
    neural.meta.env.pipe to augment the environment with additional
    features. Typically the same pipe used for training is used for
    trading. Pipe is saved as an attribute of the agent that was
    used for training. The agent can then be loaded and used for
    trading using this environment.
    """

    def __init__(
        self,
        trader: AbstractTrader,
    ) -> None:
        """
        Initialize the TradeMarketEnv class.

        Args:
        -----------
            trader: AbstractTrader
                The trader instance to connect to the environment.
        """
        self.trader = trader

        data_feeder = self.trader.data_feeder
        initial_cash = self.trader.cash
        initial_asset_quantities = self.trader.asset_quantities

        super().__init__(
            data_feeder=data_feeder,
            initial_cash=initial_cash,
            initial_asset_quantities=initial_asset_quantities,
        )

        return None

    def update_env(self) -> None:
        """
        This is identical to the TrainMarketEnv.update_env() method
        except that it also updates the cash and asset quantities. In 
        training cash and asset quantities were computed locally due to
        full control over market simulation. In trading, however, cash
        and asset quantities are computed by the trader instance. This
        is to consider the effect of slippage unfulfilled orders or 
        other market conditions that may affect the trader's cash and
        asset quantities beyond what is simulated in the environment.
        """
        super().update_env()
        self._cash = self.trader.cash
        self._asset_quantities = [
            self.trader.asset_quantities(self.assets)
        ]

        return None

    def place_orders(self, actions: np.ndarray[float]) -> None:
        """
        Places orders through the connected trader instance based on the
        provided actions. Trader may impose additional constraints on
        the actions before placing orders in the market.

        Args:
        ------
            actions (np.ndarray): 
                An array of actions to be taken for each asset in the
                environment. action = 200 means buy 200 dollars worth of
                the asset, while action = -200 means sell 200 dollars
                worth of the asset, given currncy is USD. Trader may
                place orders in the notional amount of the action or
                convert to quantity and place orders, depending on the
                facilities that the associated trade client provides.
        """

        self.trader.place_orders(actions)

        return None
