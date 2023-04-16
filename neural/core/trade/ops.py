from __future__ import annotations
from abc import ABC, abstractmethod
from torch import nn
from typing import TYPE_CHECKING

from neural.common.constants import PATTERN_DAY_TRADER_MINIMUM_NET_WORTH
from neural.connect.client import AlpacaMetaClient, AbstractClient
from neural.core.data.enums import DatasetMetadata
from neural.meta.env.pipe import AbstractPipe

if TYPE_CHECKING:
    from neural.meta.env.base import TradeMarketEnv



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

        self.symbols = self.dataset_metadata.symbols
        self._initial_cash = None
        self._cash = None
        self._asset_quantities = None
        self._net_worth = None
        self._longs = None
        self._shorts = None

        return None

    @property
    @abstractmethod
    def initial_cash(self):

        raise NotImplementedError
        
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
    def positions(self):

        raise NotImplementedError    

    @property
    @abstractmethod
    def net_worth(self):

        raise NotImplementedError
    
    @property
    @abstractmethod
    def longs(self):

        raise NotImplementedError
    
    @property
    @abstractmethod
    def shorts(self):

        raise NotImplementedError
        
    # this method is responsible for starting the trading process.
    @abstractmethod
    def trade(self, *args, **kwargs):

        raise NotImplementedError

    # takes actions from model and places relevant orders
    @abstractmethod
    def place_orders(self, actions, *args, **kwargs):

        raise NotImplementedError



class AlpacaTraderTemplate(AbstractTrader):


    def __init__(self,
        client: AlpacaMetaClient,
        model: nn.Module,
        pipe: AbstractPipe,
        dataset_metadata: DatasetMetadata):


        super().__init__(
            client,
            model,
            pipe,
            dataset_metadata)


    @property
    def initial_cash(self):

        if self._initial_cash is None:
            self._initial_cash = self.client.cash

        return self._initial_cash


    @property
    def cash(self):
        self._cash = self.client.account.cash
        return self._cash


    @property
    def asset_quantities(self):

        symbols = self.dataset_metadata.symbols
        positions = self.client.positions
        asset_quantities = []

        for symbol in symbols:
            # check if symbol is present in dataframe
            if symbol in positions['symbol'].values:

                # calculate quantity * sign
                row = positions[positions['symbol'] == symbol].iloc[0]
                if row['side'] == 'long':
                    quantity = row['qty']
                elif row['side'] == 'short':
                    quantity = -1 * row['qty']
            else:
                # set quantity to 0 if symbol is missing
                quantity = 0

            asset_quantities.append(quantity)
        return asset_quantities


    @property
    def positions(self):

        symbols = self.dataset_metadata.symbols
        positions = self.client.positions
        positions = []

        for symbol in symbols:

            # check if symbol is present in dataframe
            if symbol in positions['symbol'].values:

                # calculate quantity * sign
                row = positions[positions['symbol'] == symbol].iloc[0]
                if row['side'] == 'long':
                    position = row['market_value']
                elif row['side'] == 'short':
                    position = -1 * row['qty']

            else:
                # set position to 0 if symbol is missing
                position = 0

            positions.append(position)
        return positions

    @property
    def net_worth(self):

        self._net_worth = self.client.account.portfolio_value

        return self._net_worth


    @property
    def longs(self):

        self._longs = self.client.account.long_market_value

        return self._longs


    @property
    def shorts(self):

        self._shorts = self.client.account.short_market_value

        return self._shorts


    def place_orders(self, action, *args, **kwargs):
        raise NotImplemented


    def check_trade_constraints(self, *args, **kwargs):

        # pattern day trader constraint
        patttern_day_trader = self.client.account.pattern_day_trader
        net_worth = self.net_worth

        pattern_day_trader_constraint = True if not patttern_day_trader \
            else net_worth > PATTERN_DAY_TRADER_MINIMUM_NET_WORTH

        # margin trading
        margin = abs(self.cash) if self.cash < 0 else 0
        maintenance_margin = 1.00

        margin_constraint = margin * maintenance_margin <= self.porftfolio_value

        return pattern_day_trader_constraint and margin_constraint


    def trade(self):

        self.trade_market_env = TradeMarketEnv(trader = self)

        piped_trade_env = self.pipe(self.trade_market_env)
        observation = piped_trade_env.reset()

        while True:

            action = self.model(observation)
            observation, reward, done, info = piped_trade_env.step(action)


class CustomAlpacaTrader(AlpacaTraderTemplate):

    # inherit and override apply rules to 
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
    

    def apply_rules(self, *args, **kwargs):
        # any user
        raise NotImplementedError
    

    def customize_place_orders(self, place_order_func):

        def custom_place_order(action):

            self.check_trade_constraints()
            self.apply_rules()

            place_order_func(action)

            return None

        return custom_place_order
    

    @customize_place_orders
    def place_orders(self, action, *args, **kwargs):

        return super().place_orders(action, *args, **kwargs)