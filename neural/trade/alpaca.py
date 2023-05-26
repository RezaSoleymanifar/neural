"""
alpaca.py
"""
import numpy as np

from neural.client.alpaca import AlpacaTradeClient, AlpacaDataClient
from neural.common.exceptions import TradeConstraintViolationError
from neural.common.constants import PATTERN_DAY_TRADER_MINIMUM_EQUITY
from neural.meta.agent import Agent
from neural.trade.base import AbstractTrader


class AlpacaTrader(AbstractTrader):
    """
    A concrete implementation of the AbstractTrader class. This class
    implements the place_orders method to place orders using the Alpaca
    trading client. This class also implements the check_constraints
    method to check if the trader meets the constraints to trade. The
    constraints are:
        - The trader must have at least 120% of the pattern day trader
            minimum equity if delta = 0.20. Pattern day trader minimum
            equity is $25,000.
        - The trader must have a positive excess margin.
    
    Methods:
    --------
        check_constraints:
            Check if the trader meets the constraints to trade.
        place_orders:
            Place orders using the Alpaca trading client.
    
    Raises:
    -------
        TradeConstraintViolationError:
            If the trader does not meet the constraints to trade.

    Notes
    -----
    If manual liquidation of assets is not performed, the trader may
    receive a margin call. If trader violates the pattern day trader
    equity requirement, the trader will be restricted from day trading
    for 90 days. Set delta high enough to avoid this.
    """

    def __init__(self,
                 trade_client: AlpacaTradeClient,
                 data_client: AlpacaDataClient,
                 agent: Agent) -> None:

        super().__init__(trade_client=trade_client,
                         data_client=data_client,
                         agent=agent)

    def check_constraints(self, delta=0.2):

        if self.trade_market_env.equity < (1 + self.min_equity_ratio) * (
                1 + delta) * PATTERN_DAY_TRADER_MINIMUM_EQUITY:
            raise TradeConstraintViolationError(
                'Trader does not meet the equity requirement to trade.')

        if self.trade_market_env.excess_margin < 0:
            raise TradeConstraintViolationError(
                'Trader may receive a margin call if no action is taken.')

        return None

    def place_orders(
        self,
        actions: np.ndarray[float],
        quantity_precision: int = 5,
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
        self.check_constraints()

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
