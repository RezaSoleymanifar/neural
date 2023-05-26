"""
base.py
"""
from abc import ABC
from typing import List
import numpy as np

from neural.client.alpaca import AbstractTradeClient, AbstractDataClient
from neural.data.base import AsyncDataFeeder, AlpacaAsset
from neural.env.base import TradeMarketEnv
from neural.meta.agent import Agent


class AbstractTrader(ABC):
    """
    Abstract base class for defining a trader that can execute orders
    using a trading client and a model to produce actions. This trader
    requires a trading client to connect to a trading environment, a
    data client to stream the live data and an agent to perform the
    decision making. The agent has a model to generate actions, a data
    pipe to modify base environment, and metadata for the dataset used
    to train the agent.  Metadata will be used to create aggregated data
    stream matching the training data. 
    
    TODO: support or multiple data clients for streaming from multiple
    data stream sources.

    Attributes:
    ----------
        trade_client (AbstractTradeClient):
            An instance of the trading client to connect to the trading
            platform.
        data_client (AbstractDataClient):
            An instance of the data client to stream data.
        agent (Agent):
            An instance of the agent to perform decision making.
        data_feeder (AsyncDataFeeder):
            An instance of the data feeder to stream data.
        market_env (TradeMarketEnv):
            An instance of the trading environment.
    
    Properties:
    -----------
        cash (float):
            Cash available in the trading account. Cash can be positive
            or negative depending on the amount of cash borrowed from
            the broker.
        asset_quantities (np.ndarray[float]):
            The current quantity of each asset held by the trader. Asset
            quantities can be positive or negative. Negative quantities
            indicate that the trader has shorted the asset, namely the
            trader owes the asset to the broker.
    """

    def __init__(self, trade_client: AbstractTradeClient,
                 data_client: AbstractDataClient, agent: Agent):
        """
        Initializes an AbstractTrader object.

        Args:
        ------
            trade_client (AbstractTradeClient):
                An instance of the trading client to connect to the
                trading platform.
            data_client (AbstractDataClient):
                An instance of the data client to stream data.
            agent (Agent):
                An instance of the agent to perform decision making.
        """

        self.trade_client = trade_client
        self.data_client = data_client
        self.agent = agent

        self._data_feeder = None
        self._trade_market_env = None

        return None

    @property
    def cash(self) -> float:
        """
        Cash available in the trading account. Cash can be positive or
        negative depending on the amount of cash borrowed from the
        broker.
        """
        return self.trade_client.cash

    @property
    def asset_quantities(self) -> np.ndarray[float]:
        """
        A numpy array of current quantity of each asset held by the
        trader. Asset quantities can be positive or negative. Negative
        quantities indicate that the trader has shorted the asset,
        namely the trader owes the asset to the broker.

        Returns:
        --------
            asset_quantities (np.ndarray[float]):
                The current quantity of each asset held by the trader.
        """
        return self.trade_client.asset_quantities

    @property
    def asset_prices(self) -> np.ndarray[float]:
        """
        A numpy array of current price of each asset held by the
        trader.

        Returns:
        --------
            asset_prices (np.ndarray[float]):
                The current price of each asset held by the trader.
        """
        return self.trade_market_env.asset_prices

    @property
    def assets(self) -> List[AlpacaAsset]:
        """
        A numpy array of assets held by the trader.

        Returns:
        --------
            assets (np.ndarray[str]):
                The assets held by the trader.
        """
        return self.trade_market_env.assets

    @property
    def data_feeder(self) -> AsyncDataFeeder:
        """
        The data feeder used to stream data from the data client.

        Returns:
        --------
            data_feeder (AsyncDataFeeder):
                An instance of the data feeder to stream data.
        """
        if self._data_feeder is None:
            stream_metadata = self.agent.dataset_metadata.stream
            self._data_feeder = AsyncDataFeeder(stream_metadata,
                                                self.data_client)
        return self._data_feeder

    @property
    def trade_market_env(self) -> TradeMarketEnv:
        """
        The trading environment used to execute orders.
        """
        if self._trade_market_env is None:
            self._trade_market_env = TradeMarketEnv(trader=self)
        return self._trade_market_env

    def place_orders(self, actions: np.ndarray, *args, **kwargs):
        """
        
        """
        for action, asset, price, quantity in zip(actions, self.assets,
                                                  self.asset_prices,
                                                  self.asset_quantities):
            new_quantity = action / price

            if not asset.fractionable:
                new_quantity = int(new_quantity)
            
            if not asset.shortable:
                if quantity + new_quantity < 0:
                    new_quantity = quantity

            if (quantity > 0 and quantity + new_quantity < 0 or
                    quantity < 0 and quantity + new_quantity > 0):
                new_quantity = quantity

            if quantity == 0 and new_quantity < 0:
                new_quantity = int(new_quantity)
            
            self.trade_client.place_order(*args, **kwargs)
            
    def trade(self):
        """
        Starts the trading process by creating a trading environment and
        executing actions from the model.

        Raises:
            NotImplementedError: This method must be implemented by a
            subclass.
        """

        self.trade_market_env = self._get_trade_market_env(self)
        observation = self.trade_market_env.reset()

        while True:
            action = self.model(observation)
            observation, reward, done, info = self.trade_market_env.step(action)
            if done:
                self.trade_market_env.reset()
