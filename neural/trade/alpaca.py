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

    def check_trade_constraints(self, *args, **kwargs):
        """
        The PDT rule is a regulation enforced by the U.S. Securities and
        Exchange Commission (SEC) that applies to traders who execute
        four or more day trades within a five-business-day period using
        a margin account. A day trade is the act of buy and sell of the
        same asset on the same day in a margin account. Same applies to
        shorting and then covering the same asset on the same day in a
        margin account. If a trader meets this threshold, they are
        classified as a pattern day trader and must maintain a minimum
        equity balance of $25,000 in their account to continue day
        trading. v

        If margin falls bellow maintenance margin then a margin call is
        issued. A margin call is a broker's demand on an investor using
        margin to deposit additional money or securities so that the
        margin account is brought up to the minimum maintenance margin.
        Margin calls occur when the account value depresses to a value
        calculated by the broker's particular formula. If the investor
        fails to bring the account back into line,

        Alpaca API has automatic margin call and pattern day trader
        protection in place. This facility is provided to avoid
        triggering the Alpaca's protection mechanism in a more
        conservative way.

        Raises:
        ---------
            TradeConstraintViolationError: If any trade constraint is
            violated.
        
        Notes:
        ---------
            This is only valid in margin accounts. If using cash account
            the PDT rule does not apply. Also if using 
        """

        # pattern day trader constraint
        patttern_day_trader = self.client.account.pattern_day_trader
        net_worth = self.net_worth

        pattern_day_trader_constraint = (True if not patttern_day_trader else
                                         net_worth
                                         > PATTERN_DAY_TRADER_MINIMUM_EQUITY)

        if not pattern_day_trader_constraint:
            raise TradeConstraintViolationError(
                'Pattern day trader constraint violated.')

        # margin trading
        margin = abs(self.cash) if self.cash < 0 else 0
        maintenance_margin = 1.00
        margin_constraint = margin * maintenance_margin <= self.porftfolio_value

        if not margin_constraint:
            raise TradeConstraintViolationError('Margin constraint violated.')

        return None


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

    def place_orders(self,
                     actions: np.ndarray[float],
                     quantity_decimals: int = 5,
                     *args,
                     **kwargs):

        for action, asset, price, quantity in zip(actions, self.assets,
                                                  self.asset_prices,
                                                  self.asset_quantities):
            new_quantity = round(action / price, quantity_decimals)

            if not asset.fractionable:
                new_quantity = int(new_quantity)
            if not asset.shortable:
                if quantity + new_quantity < 0:
                    new_quantity = quantity
            if (quantity > 0 and quantity + new_quantity < 0
                    or quantity < 0 and quantity + new_quantity > 0):
                new_quantity = quantity
            if quantity == 0 and new_quantity < 0:
                new_quantity = int(new_quantity)

            self.trade_client.place_order(
                asset=asset,
                quantity=new_quantity,
                time_in_force='ioc',
            )
