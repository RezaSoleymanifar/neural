from __future__ import annotations
from abc import ABC, abstractmethod
from torch import nn
from typing import TYPE_CHECKING, List

from neural.common.constants import PATTERN_DAY_TRADER_MINIMUM_NET_WORTH
from neural.common.exceptions import TradeConstraintViolationError
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
            row = positions[positions['symbol'] == symbol]

            if not row.empty:
                row_data = row.iloc[0]
                quantity = (row_data['qty'] 
                    if row_data['side'] == 'long' 
                    else - 1 * row_data['qty'])
            else:
                quantity = 0

            asset_quantities.append(quantity)
        return asset_quantities


    @property
    def positions(self) -> List[float]:

        symbols = self.dataset_metadata.symbols
        positions = self.client.positions
        positions = []

        for symbol in symbols:
            row = positions[positions['symbol'] == symbol]

            if not row.empty:
                row_data = row.iloc[0]
                position = (row_data['market_value'] 
                    if row_data['side'] == 'long' 
                    else - 1 * row_data['market_value'])
            else:
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
        # IOC order 
        raise NotImplementedError


    def check_trade_constraints(self, *args, **kwargs):

        # pattern day trader constraint
        patttern_day_trader = self.client.account.pattern_day_trader
        net_worth = self.net_worth

        pattern_day_trader_constraint = True if not patttern_day_trader \
            else net_worth > PATTERN_DAY_TRADER_MINIMUM_NET_WORTH

        if not pattern_day_trader_constraint:
            raise TradeConstraintViolationError(
                'Pattern day trader constraint violated.')


        # margin trading
        margin = abs(self.cash) if self.cash < 0 else 0
        maintenance_margin = 1.00

        margin_constraint = margin * maintenance_margin <= self.porftfolio_value

        if not margin_constraint:
            raise TradeConstraintViolationError(
                'Margin constraint violated.')

        return None


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

        # hard rules like min trade size
        # integer asset quantity
        # no short etc. are applied here.

        raise NotImplementedError
    

    def custom(self, place_order_func):

        def custom_place_order(action):

            self.check_trade_constraints()
            self.apply_rules()

            place_order_func(action)

            return None

        return custom_place_order
    

    @custom
    def place_orders(self, action, *args, **kwargs):

        return super().place_orders(action, *args, **kwargs)