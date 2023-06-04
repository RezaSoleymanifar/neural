"""
base.py

This module defines the base classes for the market environment needed
for training and trading with deep reinforcement learning algorithms.
The base classes are designed to be used with the OpenAI Gym interface.

Classes:
--------
    AbstractMarketEnv:
        Abstract base class for market environments. In case a custom
        market environment is needed, it should inherit from this class.
        It preserves the logic of sequentially updating the environment
        and placing orders based on actions received from the agent.
    TrainMarketEnv:
        A bare metal market environment with no market logic. Natively
        allowes cash and asset quantities to be negative, accommodating
        short/margin trading by default. Use action wrappers to impose
        market logic.
    TradeMarketEnv:
        This is a subclass of TrainMarketEnv. It is intended to be used
        for trading. It is identical to TrainMarketEnv except that it is
        connected to a trader instance. This allows the environment to
        interact with the trader and place orders in the market. Use the
        pipes in `neural.meta.env.pipe` to augment the environment with
        additional features. Typically the same pipe used for training
        is used for trading. Pipe is saved as an attribute of the agent
        that was used for training. The agent can then be loaded and
        used for trading using this environment.

Notes:
------
    Market environments should be designed to interact with agents in
    order to simulate the behavior of a real-world market.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, TYPE_CHECKING, Optional

from gym import spaces, Env
import numpy as np

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
    update(self):
        Abstract method for updating the market environment. Thismethod
        uses the generator returned by the data feeder to update the 
        environment state.
    construct_observation(self):
        Abstract method for constructing the market observation.
    place_orders(actions):
        Abstract method for placing orders in the market environment.
        Actions are notional values of assets to buy/sell. The sign of
        the action determines whether to buy or sell. Example:
            Buy 100 shares of AAPL at $100 per share:
                action = 100 * 100 = 10000
            Sell 100 shares of AAPL at $100 per share:
                action = -100 * 100 = -10000

    Notes:
    ------
    Market environments should be designed to interact with agents in
    order to simulate the behavior of a real-world market for training.
    This base class defines the minimum interface required for an
    environment to be used for high frequency trading.

    There are typically two types of market environments:
        - Training environments
        - Trading environments

    Training environments are used for training agents with historical
    data and performs accounting for financial operations internally.
    Trading environments are used with trained agents to stream live
    market data and place orders in the market using a trading API. In
    this case financial accounting is performed by the broker and
    accounting data is read from the broker's API.
    """

    @abstractmethod
    def update(self):
        """
        Abstract method for updating the market environment. Should be
        implemented by subclasses. This method should update the
        environment state by moving to the next time step and updating
        the environment internal variables such as features, and holds.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
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
        implementation of the subclass. actions are notional values
        of assets to buy/sell. The sign of the action determines
        whether to buy or sell. The magnitude of the action determines
        the notional value of the order. The notional value of the
        order is the product quantity and price. 

        Example:
        --------
        Buy 100 shares of AAPL at $100 per share:
            action = 100 * 100 = 10000
        Sell 100 shares of AAPL at $100 per share:
            action = -100 * 100 = -10000
        """
        raise NotImplementedError


class TrainMarketEnv(AbstractMarketEnv):
    """
    A bare metal market environment with no market logic for training
    agents. Natively allowes cash and asset quantities to be negative,
    accommodating short/margin trading by default. The default
    environment allows cash and asset quantities to grow indefinitely in
    any direction (positive or negative). Use action wrappers to impose
    market logic such as margin logic (e.g. account initial and
    maintenance margins). For trading, use TradeMarketEnv. Use the pipes
    in `neural.meta.env.pipe` to augment the environment with additional
    features. This enables the environment to respect real market
    constraints (e.g. short selling and margin trading constraints).

    Attributes:
    -----------
        data_feeder (StaticDataFeeder):
            The StaticDataFeeder instance providing the data to the
            environment. This is used to update the environment state
            and construct the observation.
        initial_cash (float, optional):
            The initial amount of cash to allocate to the environment.
            Default is 1e6.
        initial_asset_quantities (np.ndarray, optional):
            The initial quantity of assets to allocate to the
            environment. Default is None.
        metadata (DatasetMetadata):
            Metadata about the dataset used. This includes the feature
            schema, asset names, and asset price mask.
        assets (List[AbstractAsset]):
            A list of assets in the dataset.
        n_assets (int):
            The number of assets in the dataset.
        n_features (int):
            The number of features in the dataset.
        holds (np.ndarray):
            An integer array representing the number of steps each asset
            has been held by the environment. As soon as a trade
            (buy/sell) is placed, the holds for that asset is reset to
            zero. If asset is held (long/short) then hold is incremented
            at each time step. If asset is not held, hold stays at zero.
        features (np.ndarray):
            An array representing the current features of the
            environment. Features can include a wide swath of possible
            data, including market data such as open, high, low, close,
            volume, and other features such as sentiment scores, text
            embeddings, or raw text. Features is typically a long vector
            with shape (n_features,). In order to extract individual
            features, feature_schema is used to apply boolean masks and
            retrieve the desired features.
        _cash (float):
            The current amount of cash in the environment. Cash can be
            negative, allowing for margin trading.
        _asset_quantities (np.ndarray):
            An array representing the quantities of each asset held by
            the environment. A positive value means the asset is held
            long, while a negative value means the asset is held short.
            Asset quantities can be fractional, allowing for partial
            shares, or integer, allowing for only whole shares.
        _asset_prices (np.ndarray):
            An array representing the current asset prices of the
            environment.
        features_generator (Iterator[np.ndarray]):
            An iterator that yields the next feature row of the dataset.
            This iterator is used to update the environment state by
            moving to the next time step and updating the environment
            variables.
        info (Dict):
            A dictionary for storing additional information (unused for
            now)
        action_space (gym.spaces.Box):
            Number of actions is equal to the number of assets in the
            dataset. Each action is the notional value of the asset to
            buy or sell. A zero value means no action is taken (hold).
            Buys are represented as positive values, while sells are
            represented as negative values. Notional value means the
            face value of the asset (e.g. 100 means buy 100 dollars
            worth of the asset, while -100 means sell 100 dollars worth
            of the asset, given currency is USD).
        observation_space (gym.spaces.Dict): 
            The observation space is a dictionary containing the
            following keys:
                - 'cash' (numpy.ndarray): 
                    A scalar numpy array representing the available cash
                    in the account. 
                - 'asset_quantities' (numpy.ndarray): 
                    A numpy array representing the quantities of assets
                    held. 
                - 'holds' (numpy.ndarray): 
                    A numpy array representing the number of consecutive
                    steps an asset has been held. 
                - 'features' (numpy.ndarray): 
                    A numpy array representing the current features of
                    the environment.

    Properties:
    -----------
        done (bool):
            A boolean value indicating whether the current episode is
            finished.
        cash (float):
            The current amount of cash in the environment.
        asset_quantities (np.ndarray):
            An array representing the quantities of each asset held by
            the environment. A positive value means the asset is held
            long, while a negative value means the asset is held short.
            Asset quantities can be fractional, allowing for partial
            shares, or integer, allowing for only whole shares.
        asset_prices (np.ndarray):
            An array representing the current asset prices of the
            environment.

    Methods:
    --------
    update(self) -> None:
        Uses features_generator to update the environment state by
        moving to the next time step.
    construct_observation(self) -> Dict[str, np.ndarray[float]):
        Constructs the current observation from the environment's state
        variables. The observation includes:
            - cash
            - asset_quantities
            - holds
            - features
    place_orders(actions) -> None:
        Places orders on the assets based on the given actions. Performs
        the book keeping internally by updating variables such as cash,
        asset quantities, and holds.
    reset(self) -> Dict[str, np.ndarray]:
        Resets the market environment to its initial state. Sets initial
        values for cash, asset quantities, and holds. Returns the
        initial observation. This is consistent with gym.Env.reset()
        from OpenAI gym API.
    step(actions: np.ndarray[float]) -> Tuple[Dict[str, np.ndarray],
    float, bool, Dict]:
        Executes a step in the trading environment. It places orders
        based on the given actions, updates the environment state, and
        constructs the observation. Returns the observation, reward,
        done, and info. This is consistent with gym.Env.step() from
        OpenAI gym API.

    Examples:
    ---------
    >>> from neural.data.base import StaticDataFeeder
    >>> from neural.env.base import TrainMarketEnv
    >>> data_feeder = StaticDataFeeder(path=...)
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
        data_feeder (StaticDataFeeder):
            The StaticDataFeeder instance providing the data to the
            environment
        initial_cash (float, optional):
            The initial amount of cash to allocate to the environment.
            Default is 1e6.
        initial_asset_quantities (np.ndarray, optional):
            The initial quantity of assets to allocate to the
            environment. Default is None.
        """
        self.data_feeder = data_feeder
        self.initial_cash = initial_cash
        self.initial_asset_quantities = initial_asset_quantities

        self.metadata = self.data_feeder.metadata
        self.assets = self.metadata.assets
        self.n_assets = len(self.assets)
        self.n_features = self.metadata.n_features

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
    def done(self) -> bool:
        """
        Returns a boolean value indicating whether the current episode
        is finished.

        Returns:
        --------
            done (bool):
                A boolean value indicating whether the current episode
                is finished.
        """
        return self.data_feeder.done
    
    @property
    def cash(self) -> float:
        """
        The current amount of cash in the environment.

        Returns:
        --------
            cash (float)
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
            asset_quantities (np.ndarray):
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
            asset_prices (np.ndarray):
                An array representing the current asset prices of the
                assets.
        """
        asset_prices_mask = self.metadata.asset_prices_mask
        self._asset_prices = self.features[asset_prices_mask]

        return self._asset_prices


    def update(self) -> None:
        """
        Updates the environment state by moving to the next time step
        and updating the environment variables such as features and
        holds.
        """
        self.features = next(self.features_generator)
        self.holds[self.asset_quantities != 0] += 1

        return None

    def get_observation(self) -> Dict[str, np.ndarray[float]]:
        """
        Constructs the current observation from the environment's state
        variables. The observation includes the current cash balance,
        asset quantities, holds (time steps an asset has been held), and
        features of the current time step.

        Returns:
        -------
        observation: 
            Dict[str, np.ndarray[float]] The observation dictionary
            containing the current cash balance, asset quantities,
            holds, and features.
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
        self.holds = np.zeros((self.n_assets, ), dtype=GLOBAL_DATA_TYPE)

        self._cash = np.array([self.initial_cash], dtype=GLOBAL_DATA_TYPE)
        self._asset_quantities = (np.zeros(
            (self.n_assets, ), dtype=GLOBAL_DATA_TYPE)
                                 if self.initial_asset_quantities is None
                                 else self.initial_asset_quantities)

        self.update()
        observation = self.get_observation()

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
        """

        self.place_orders(actions)

        self.update()

        reward = None
        observation = self.get_observation()

        return observation, reward, self.done, self.info


class TradeMarketEnv(TrainMarketEnv):
    """
    This is a subclass of TrainMarketEnv. It is intended to be used for
    trading. It is almost identical to TrainMarketEnv except that it is
    connected to a trader instance. This allows the environment to
    interact with the trader and place orders in the market. This class
    is not intended to be used directly. Instead, use the pipes in
    neural.meta.env.pipe to augment the environment with additional
    features. Typically the same pipe used for training is used for
    trading. Pipe is saved as an attribute of the agent that was used
    for training. The agent can then be loaded and used for trading
    using this environment.

    Attributes:
    -----------
        trader: AbstractTrader
            The trader instance to connect to the environment.
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

    def update(self) -> None:
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
        super().update()
        self._cash = self.trader.cash
        self._asset_quantities = self.trader.asset_quantities(self.assets)

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
                worth of the asset, given currncy is USD.
        """
        self.trader.place_orders(actions)

        return None
