from neural.core.data.enums import DatasetMetadata
from neural.core.data.ops import AsyncDataFeeder
from neural.connect.client import AbstractClient
from neural.meta.env.base import TradeMarketEnv, TrainMarketEnv
from neural.meta.env.pipe import AbstractPipe
from neural.connect.client import AlpacaMetaClient
from torch import nn
import os
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class AbstractTrader(ABC):

    def __init__(self,
        client: AbstractClient,
        model: nn.Module,
        pipe: AbstractPipe,
        dataset_metadata: DatasetMetadata):

        self.client = client
        self.model = model
        self.pipe = pipe
        self.dataset_metadata = dataset_metadata

        self._cash = None
        self._asset_quantities = None
        self._net_worth = None

        return None
    
    @property
    @abstractmethod
    def cash(self):

        raise NotImplementedError
    
    @property
    @abstractmethod
    def asset_quantities(self):

        raise NotImplementedError
    
    @property
    @abstractmethod
    def net_worth(self):

        raise NotImplementedError
    
    # this method is responsible for starting the trading process.
    @abstractmethod
    def trade(self, *args, **kwargs):

        raise NotImplementedError

    # takes actions from model and places relevant orders
    @abstractmethod
    def place_orders(self, actions, *args, **kwargs):

        raise NotImplementedError



class AlpacaTrader(AbstractTrader):

    def __init__(self, 
        client: AlpacaMetaClient, 
        model : nn.Module, 
        pipe: AbstractPipe, 
        dataset_metadata: DatasetMetadata):

        super().__init__(
            client,
            model,
            pipe,
            dataset_metadata)

    @property
    def cash(self):
        self._cash = self.client.account.cash
        return self._cash
    
    @property
    def asset_quantities(self):
        self._asset_quantities = self.client.account.equity
        return self._asset_quantities
    
    @property
    def net_worth(self):
        self._net_worth = self.client.account.portfolio_value
        return self._net_worth


    def place_orders(actions):
        pass


    def trade(self):

        self.trade_market_env = TradeMarketEnv(trader = self)


        piped_trade_env = self.pipe(self.trade_market_env)
        observation = piped_trade_env.reset()

        while True:
            action = self.model(observation)
            observation, reward, done, info = piped_trade_env.step(action)


class CustomAlpacaTrader(AlpacaTrader):

    def __init__(
            self, client: 
            AlpacaMetaClient, 
            model: nn.Module, 
            pipe: AbstractPipe, 
            dataset_metadata: DatasetMetadata
            ) -> None:
        
        super().__init__(
            client, 
            model, 
            pipe, 
            dataset_metadata)

        self.warumup_dataset_path = None


    def record_experiences(self):
        raise NotImplementedError
    

    def warumup_pipe(warmup_env: TrainMarketEnv, n_episode: Optional[int] = None):
        raise NotImplementedError


    def place_orders(actions):
        raise NotImplementedError


    def trade(self):

        trade_market_env = TradeMarketEnv(trader=self)

        if self.warumup_dataset_path is not None:

            self.warmup_pipe(n_episodes=3)

        piped_trade_env = self.pipe(trade_market_env)
        obs = piped_trade_env.reset()

        while True:
            action = self.model(obs)
            observation, reward, done, info = piped_trade_env.step(action)
            self.record_experiences(observation, reward, done)
            if done:
                break