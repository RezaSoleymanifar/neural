"""
alpaca.py
"""
import numpy as np

from neural.client.alpaca import AlpacaTradeClient
from neural.common.exceptions import TradeConstraintViolationError
from neural.common.constants import PATTERN_DAY_TRADER_MINIMUM_EQUITY
from neural.meta.agent import Agent
from neural.trade.base import AbstractTrader, AbstractDataClient


class AlpacaTraderFactory(AbstractTrader):

    def __init__(self, trade_client: AlpacaTradeClient,
                 data_client: AbstractDataClient, agent: Agent) -> None:

        super().__init__(trade_client=trade_client,
                         data_client=data_client,
                         agent=agent)


class AlpacaTrader(AlpacaTraderFactory):
    """
    A custom implementation of the AlpacaTraderTemplate that allows for
    custom order placing and rule application.

    Args:
        client (AlpacaMetaClient): An instance of the Alpaca client.
        agent (Agent): An instance of the agent to perform decision
        making.

    Attributes:
        trade_client (AlpacaMetaClient): An instance of the Alpaca
        client. agent (Agent): An instance of the agent to perform
        decision making.
    """

    def __init__(self, trade_client: AlpacaTradeClient, agent: Agent) -> None:

        super().__init__(trade_client, agent=agent)

    def check_constraints(self, delta = 0.2):

        if self.trade_market_env.equity < (
                1 + self.min_equity_ratio) * PATTERN_DAY_TRADER_MINIMUM_EQUITY:
            raise TradeConstraintViolationError(
                'Trader does not meet the equity requirement to trade.')
        
    def place_orders(
        self,
        actions: np.ndarray[float],
        quantity_precision: int = 5,
        min_equity_ratio: float = 0.25,
    ):
        """
        This method places orders based on the actions provided by the
        agent. The actions are the notional values to be traded for each
        asset. The notional value is calculated as the product of the
        price and the quantity. The quantity is calculated as the action
        divided by the price. The quantity is rounded to the nearest
        integer if the asset is not fractionable. If asset is not
        shortable, and position is being shorted the action is set to
        zero. The quantity is rounded to the nearest integer if the
        asset is being shorted. (Alpaca API does not allow fractional
        shorting). If position flips from long to short or vice versa,
        the quantity is modified so that only the position is closed.
        (Alpaca API does not allow for continuous flipping of
        positions).

        Args:
        ----------
            actions (np.ndarray[float]):
                The actions to be taken for each asset.
            quantity_precision (int, optional):
                The precision to which the quantity is rounded. Defaults
                to 5. 
        
        Notes:
        ---------
            Assets can have maximum quantity precision constraints. Set 
            the quantity_precision high enough to avoid this triggering
            this constraint.
        """

        for action, asset, price, quantity in zip(actions, self.assets,
                                                  self.asset_prices,
                                                  self.asset_quantities):
            new_quantity = round(action / price, quantity_precision)

            if not asset.fractionable:
                new_quantity = int(new_quantity)
            if not asset.shortable:
                if quantity == 0 and new_quantity < 0:
                    new_quantity = 0
            if quantity == 0 and new_quantity < 0:
                new_quantity = int(new_quantity)
            if (quantity > 0 and quantity + new_quantity < 0
                    or quantity < 0 and quantity + new_quantity > 0):
                new_quantity = quantity

            self.trade_client.place_order(
                asset=asset,
                quantity=new_quantity,
                time_in_force='ioc',
            )
