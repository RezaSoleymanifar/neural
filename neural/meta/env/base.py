from typing import Tuple, Dict

import numpy as np
from gym import spaces, Env

from abc import ABC, abstractmethod
from neural.common.log import logger
from neural.core.data.ops import StaticDataFeeder, AsyncDataFeeder
from neural.core.data.enums import ColumnType
from neural.core.trade.ops import CustomAlpacaTrader, AbstractTrader


class AbstractMarketEnv(Env, ABC):


    @abstractmethod
    def update_env(self):

        raise NotImplementedError
    
    @abstractmethod
    def construct_observation(self):

        raise NotImplementedError



class TrainMarketEnv(AbstractMarketEnv):
    # bare metal market environment with no market logic
    # allows short/margin by default 
    # natively allowes cash and asset_quantities to be negative.
    # use action wrappers to enforce no margin, no short or neither
    def __init__(
        self, 
        market_data_feeder: StaticDataFeeder,
        initial_cash: float = 1e6,
        ) -> None:
        

        self.data_feeder = market_data_feeder
        self.dataset_metadata = self.data_feeder.dataset_metadata
        self.column_schema = self.dataset_metadata.column_schema
        self.asset_price_mask = self.dataset_metadata.column_schema[ColumnType.CLOSE]
        self.n_steps = self.dataset_metadata.n_rows
        self.n_features = self.dataset_metadata.n_columns
        self.n_symbols = len(self.dataset_metadata.symbols)

        self.index = None
        self.initial_cash = initial_cash
        self.cash = None
        self.net_worth = None # cash + assets (stock/crypto)
        self.asset_quantities = None # quantity of each assetheld
        self.holds = None # steps asset did not have trades
        self.features = None
        self.asset_prices = None
        self.info = None

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(
            self.n_symbols,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'cash':spaces.Box(
            low=-np.inf, high=np.inf, shape = (1,), dtype=np.float32),

            'asset_quantities': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.int32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})
    

    def update_env(self) -> np.ndarray:

        self.features = next(self.row_generator)
        self.asset_prices = self.features[self.asset_price_mask]

        return None

    def construct_observation(self) -> np.ndarray:
        
        observation = {
            'cash': self.cash,
            'asset_quantities': self.asset_quantities,
            'holds': self.holds,
            'features': self.features
            }
        
        return observation

    def reset(
            self
            ) -> Dict:
                
        # resetting input state
        self.row_generator = self.data_feeder.reset()

        self.update_env()

        self.index = 0
        self.holds = np.zeros((self.n_symbols,), dtype=np.int32)

        self.cash = self.initial_cash
        self.asset_quantities = np.zeros((self.n_symbols,), dtype=np.float32)
        self.net_worth = self.cash

        self.observation = self.construct_observation()

        return self.observation

    def step(
        self, 
        actions
        ) -> Tuple[Dict, np.float32, bool, Dict]:

        # iterates over actions
        for asset, action in enumerate(actions):

            if action > 0: # buy
                
                buy = action
                quantity = buy/self.asset_prices[asset]

                self.asset_quantities[asset] += quantity
                self.cash -= buy

            elif action < 0:
                
                sell = abs(action)
                quantity = sell/self.asset_prices[asset]

                self.asset_quantities[asset] -= quantity
                self.cash += sell
                self.holds[asset] = 0

        self.update_env()

        self.index += 1
        self.holds[self.asset_quantities != 0] += 1

        # new asset value
        new_net_worth = self.cash + self.asset_quantities @ self.asset_prices
        reward = new_net_worth - self.net_worth
        self.net_worth = new_net_worth
        
        # returns states using env vars
        self.observation = self.construct_observation()

        # report terminal state
        done = self.index == self.n_steps - 1
            
        return self.observation, reward, done, self.info
        


class TradeMarketEnv(AbstractMarketEnv):

    def __init__(
        self,
        trader: AbstractTrader
        ) -> None:
    
        self.trader = trader
        self.dataset_metadata = self.trader.dataset_metadata
        self.data_feeder = AsyncDataFeeder(self.dataset_metadata)

        self.column_schema = self.dataset_metadata.column_schema
        self.asset_price_mask = self.dataset_metadata.column_schema[ColumnType.CLOSE]
        self.n_features = self.dataset_metadata.n_columns
        self.n_symbols = len(self.dataset_metadata.symbols)

        self.initial_cash = self.trader.initial_cash
        self.cash = None
        self.net_worth = None # cash + assets (stock/crypto)
        self.asset_quantities = None # quantity of each assetheld
        self.holds = None # steps asset did not have trades
        self.features = None
        self.info = None

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(
            self.n_symbols,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'cash':spaces.Box(
            low=0, high=np.inf, shape = (1,), dtype=np.float32),

            'asset_quantities': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.int32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})
            
    def update_env(self):

        # this step is delayed to aggregate stream data
        self.features = next(self.row_generator)
        self.cash = self.trader.cash
        self.asset_quantities = self.trader.asset_quantities
        self.net_worth = self.trader.net_worth

        return None

    def construct_observation(self) -> np.ndarray:

        observation = {
            'cash': self.cash,
            'asset_quantities': self.asset_quantities,
            'holds': self.holds,
            'features': self.features
        }

        return observation
    

    def reset(
        self
        ) -> Dict:

        # resetting input state
        self.row_generator = self.data_feeder.reset()

        self.update_env()

        self.holds = np.zeros(
            (self.n_symbols,), dtype=np.int32)

        # compute state
        self.observation = self.construct_observation()

        return self.observation


    def step(
        self,
        actions: np.ndarray
        ) -> Tuple[Dict, np.float32, bool, Dict]:


        previous_net_worth = self.net_worth
        self.trader.place_orders(actions)

        # takes time equal to trading resolution
        self.update_env()

        self.holds[self.asset_quantities != 0] += 1

        reward = self.net_worth - previous_net_worth

        # returns states using env vars
        self.observation = self.construct_observation()


        return self.observation, reward, False, self.info
