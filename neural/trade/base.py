"""
base.py
"""
from abc import ABC

import numpy as np

from neural.client.alpaca import AbstractTradeClient, AbstractDataClient
from neural.data.base import AsyncDataFeeder
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
    """

    def __init__(
        self, 
        trade_client: AbstractTradeClient,
        data_client: AbstractDataClient,
        agent: Agent
    ):
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
        
        Attributes:
        ----------
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

        return None

    @property
    def cash(self):
        return self.trade_client.cash
    
    @property
    def asset_quantities(self):
        return self.trade_client.asset_quantities
    
    
    def apply_rules(self, *args, **kwargs):
        """
        Applies trading rules to the trades. Override this method to
        apply custom rules before placing orders. This allows rule based
        trading to complement the model based trading. For example, a
        rule could be to only buy a stock if it has a positive sentiment
        score. Or execute a techinical analysis strategy whenever a
        condition is met to override the normal behavior of the model.

        Raises:
            NotImplementedError: This method must be implemented by a
            subclass.
        """

        raise NotImplementedError

    def integer_shorts(self, *args, **kwargs):
        """
        Considers constraint on shorting stocks that quantity of stocks
        to be shorted must be an integer always.
        TODO: experiment with scenario that actions leads to shortting.
        How the residual quantity is handled in that case?
        """
        raise notImplementedError

    def place_orders(self, actions: np.ndarray, *args, **kwargs):
        """
        Takes actions from the model and places relevant orders.

        Args:
            actions (np.ndarray): A 2D numpy array of actions generated
            by the model.

        Raises:
            NotImplementedError: This method must be implemented by a
            subclass.
        """
        # Get the list of symbols from the dataset metadata

        symbols = self.agent.dataset_metadata.assets

        # Loop over the symbols and actions and place orders for each
        # symbol
        for symbol, quantity in zip(symbols, actions):
            self.trade_client.place_order(symbol, actions, *args, **kwargs)

    def _get_trade_market_env(self, trader):
        stream_metadata = self.agent.dataset_metadata.stream
        data_feeder = AsyncDataFeeder(stream_metadata, self.data_client)
        self._get_data_feeder()
        self.trade_market_env = TradeMarketEnv(trader=self)

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
