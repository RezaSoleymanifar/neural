from typing import Callable

import numpy as np
from torch import nn

from neural.client.alpaca import AlpacaTradeClient
from neural.common.exceptions import TradeConstraintViolationError
from neural.common.constants import PATTERN_DAY_TRADER_MINIMUM_NET_WORTH
from neural.data.base import DatasetMetadata
from neural.trade.base import AbstractTrader, Agent
from neural.meta.pipe import AbstractPipe



class AlpacaTraderFactory(AbstractTrader):


    def __init__(self,
        trade_client: AlpacaTradeClient,
        agent: Agent) -> None:


        super().__init__(
            trade_client = trade_client,
            agent = agent)


    def check_trade_constraints(self, *args, **kwargs):
        """
        Checks if all trade constraints are met before placing orders.

        Raises:
            TradeConstraintViolationError: If any trade constraint is violated.
        """

        # pattern day trader constraint
        patttern_day_trader = self.client.account.pattern_day_trader
        net_worth = self.net_worth

        pattern_day_trader_constraint = (
            True if not patttern_day_trader else net_worth >
            PATTERN_DAY_TRADER_MINIMUM_NET_WORTH)

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



class AlpacaTrader(AlpacaTraderFactory):

    """
    A custom implementation of the AlpacaTraderTemplate that allows for custom order placing and rule application.

    Args:
        client (AlpacaMetaClient): An instance of the Alpaca client.
        model (nn.Module): The PyTorch model to use for trading.
        pipe (AbstractPipe): The data pipe to use for feeding the model with data.
        dataset_metadata (DatasetMetadata): The metadata for the dataset used for training the model.

    Attributes:
        client (AlpacaMetaClient): An instance of the Alpaca client.
        model (nn.Module): The PyTorch model to use for trading.
        pipe (AbstractPipe): The data pipe to use for feeding the model with data.
        dataset_metadata (DatasetMetadata): The metadata for the dataset used for training the model.

    """

    def __init__(self, 
        client: AlpacaTradeClient, 
        agent: Agent) -> None:

        super().__init__(
            client,
            model,
            pipe,
            dataset_metadata)


    def constraints(self, place_orders_func: Callable):


        def customized_place_order(action):

            self.check_trade_constraints()

            place_orders_func(action)

        return customized_place_order
    

    def rules(self, place_orders_func: Callable):

        """
        Decorator factory that returns a new function `custom_place_order`.
        The purpose of `custom_place_order` is to wrap around a given `place_order_func` function and enforce certain 
        constraints and rules before calling it.
        
        The `check_trade_constraints` method is used to check if any trade constraints are violated, and the `apply_rules` 
        method is used to apply additional rules. Once these constraints and rules have been checked and applied, 
        `place_order_func` is called with the `action` argument.
        
        The `custom` method is designed to allow users to customize the `place_orders` method while still enforcing the 
        necessary constraints and rules. It can be used by defining a custom `place_orders` function and then decorating 
        it with `custom` to ensure that the necessary checks are performed before the orders are placed.
        """

        def customized_place_order(action):

            try:
                self.apply_rules()

            except NotImplementedError:
                pass

            place_orders_func(action)

        return customized_place_order
    

    @rules
    def place_orders(self, action, *args, **kwargs):

        return super().place_orders(action, *args, **kwargs)