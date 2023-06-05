"""
alpaca.py

Description:
------------
    This module contains the AlpacaTrader class. This class implements the
    place_orders method to place orders using the Alpaca trading client. This
    class also implements the check_constraints method to check if the trader
    meets the constraints to trade. The constraints are:
        - The trader must have at least 120% of the pattern day trader minimum
            equity (if delta = 0.20). Pattern day trader minimum equity is
            $25,000.
        - The trader must have a positive excess margin.    
        - The trader must satisfy a certain return on equity (ROE) threshold.   
            This threshold is set by the agent.
        
License:
--------
    MIT License. See LICENSE.md file.

Author(s):
-------
    Reza Soleymanifar, Email: Reza@Soleymanifar.com

Class(es):
---------
    AlpacaTrader:
        A concrete implementation of the AbstractTrader class.
"""
import numpy as np

from neural.client.alpaca import AlpacaTradeClient, AlpacaDataClient
from neural.common.constants import PATTERN_DAY_TRADER_MINIMUM_EQUITY
from neural.common.exceptions import TradeConstraintViolationError
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
            minimum equity (if delta = 0.20). Pattern day trader minimum
            equity is $25,000.
        - The trader must have a positive excess margin.
        - The trader must satisfy a certain return on equity (ROE) 
            threshold. This threshold is set by the agent.

    Attributes:
    ----------
        trade_client (AbstractTradeClient):
            An instance of the trading client to connect to the trading
            platform.
        data_client (AbstractDataClient):
            An instance of the data client to stream data.
        agent (Agent):
            An instance of the agent to perform decision making.
        _data_feeder (AsyncDataFeeder):
            An instance of the asynchronous data feeder to stream data.
        _trade_market_env (TradeMarketEnv):
            An instance of the trading environment.
        _model (nn.Module):
            The PyTorch neural network model used by the agent.
        handle_non_trade_time (bool):
            A flag to indicate if the trader should handle non-trade
            time. Non-trade time is defined as the time outside of the
            trading schedule. If the flag is set to True, then the
            trader will cancel all open orders and wait for the market
            to open.
    
    Properties:
    -----------
        data_feeder (AsyncDataFeeder):
            The data feeder used to stream data from the data client.
        trade_market_env (TradeMarketEnv):
            The trading environment used to execute orders.
        market_metadata_wrapper (AbstractMarketEnvMetadataWrapper):
            The metadata wrapper used by the trading environment.
        schedule (pd.DataFrame):
            The schedule of the trading environment. The schedule is a
            pandas DataFrame with two columns, start and end and date as
            the index.
        assets (List[AlpacaAsset]):
            A numpy array of assets held by the trader.
        cash (float):
            Cash available in the trading account. Cash can be positive
            or negative depending on the amount of cash borrowed from
            the broker.
        asset_quantities (np.ndarray[float]):
            A numpy array of current quantity of each asset held by the
            trader. Asset quantities can be positive or negative.
            Negative quantities indicate that the trader has shorted the
            asset, namely the trader owes the asset to the broker.
        asset_prices (np.ndarray[float]):
            A numpy array of current price of each asset held by the
            trader.
        model (nn.Module):
            Returns the model used by the agent to generate actions.
        equity (float):
            The current equity of the trader. Equity is the sum of cash 
            and the value of all assets owned by the trader. Equity = L 
            + C - S where L is the value of long positions, C is the
            cash and S is the value of short positions. Cash can be
            positive or negative.

    Methods:
    --------
        _check_time(self) -> bool:
            A method to check if the current time is within the trading
            schedule. If the current time is not within the trading
            schedule, then all open orders are cancelled.
        trade (self) -> None:
            Starts the trading process by creating a trading environment
            and executing actions from the model.
        place_orders(self, actions: np.ndarray, *args, **kwargs) ->
        None:
            Abstract method for placing an order for a single asset. The
            restrictions of the API should be enforced in this method.
        check_constraints(self, *args, **kwargs) -> None:
            Check if the trader meets the constraints to trade.
        place_orders(self, actions: np.ndarray[float], *args, **kwargs)
        -> None:
            Places orders based on the actions provided by the agent.
    
    Raises:
    -------
        TradeConstraintViolationError:
            If pattern day trader equity requirement is not met.
        TradeConstraintViolationError:
            If excess margin is negative.
        TradeConstraintViolationError:
            If return on equity is below threshold.
    """

    def __init__(self, trade_client: AlpacaTradeClient,
                 data_client: AlpacaDataClient, agent: Agent) -> None:

        super().__init__(trade_client=trade_client,
                         data_client=data_client,
                         agent=agent)

    @property
    def equity(self) -> float:
        """
        The current equity of the trader, read from the API. Equity is
        the sum of cash and the value of all assets owned by the trader.
        Equity = L + C - S where L is the value of long positions, C is
        the cash and S is the value of short positions. Cash can be
        positive or negative.

        Returns:
        --------
            equity (float):
                The current equity of the trader.
        """
        return self.trade_client.equity

    def check_constraints(self, delta=0.2, return_threshold=-0.1):
        """
        Checks trading constraints. The constraints are:
            - The trader must have at least 120% of the pattern day
                trader minimum equity if delta = 0.20. Pattern day trader
                minimum equity is $25,000.
            - The trader must have a positive excess margin.
            - The trader must satisfy a certain return on equity (ROE)
                threshold. This threshold is set by the agent.

        Args:
        ------
            delta (float, optional):
                A cushion around the pattern day trader minimum equity. If
                delta = 0.20, the trader must have at least 120% of the pattern
                day trader minimum equity. Defaults to 0.2.
            return_threshold (float, optional):
                The return on equity threshold. Defaults to -0.1.
            
        Raises:
        -------
        TradeConstraintViolationError:
            If pattern day trader equity requirement is not met.
        TradeConstraintViolationError:
            If excess margin is negative.
        TradeConstraintViolationError:
            If return on equity is below threshold.

        Notes:
        ------
            If manual liquidation of assets is not performed, the trader may
            receive a margin call. If trader violates the pattern day trader
            equity requirement, the trader will be restricted from day trading
            for 90 days. Set delta high enough to avoid this. More info here:
            https://www.finra.org/investors/investing/investment-products/stocks/day-trading
        """
        if self.equity < (1 + delta) * PATTERN_DAY_TRADER_MINIMUM_EQUITY:
            raise TradeConstraintViolationError(
                'Trader does not meet the equity requirement to trade.')

        if self.market_metadata_wrapper.excess_margin < 0:
            raise TradeConstraintViolationError(
                'Trader may receive a margin call if no action is taken.')

        if self.market_metadata_wrapper.return_ < return_threshold:
            raise TradeConstraintViolationError(
                'Trader does not meet the return threshold to continue to trade.')
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
            if quantity <= 0 and new_quantity < 0:
                new_quantity = int(new_quantity)
            if (quantity > 0 and quantity + new_quantity < 0
                    or quantity < 0 and quantity + new_quantity > 0):
                new_quantity = quantity

            self.trade_client.place_order(
                asset=asset,
                quantity=new_quantity,
                time_in_force='ioc',
            )
