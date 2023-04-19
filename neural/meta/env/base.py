from __future__ import annotations

from typing import Tuple, Dict, TYPE_CHECKING, Optional

import numpy as np
from gym import spaces, Env

from abc import ABC, abstractmethod
from neural.core.data.ops import StaticDataFeeder, AsyncDataFeeder
from neural.core.data.enums import ColumnType

if TYPE_CHECKING:
    from neural.core.trade.ops import AbstractTrader



class AbstractMarketEnv(Env, ABC):

    # abstract class for market envs.

    @abstractmethod
    def update_env(self):

        raise NotImplementedError
    

    @abstractmethod
    def construct_observation(self):

        raise NotImplementedError


    @abstractmethod
    def place_orders(self, actions):

        raise NotImplementedError



class TrainMarketEnv(AbstractMarketEnv):

    # bare metal market environment with no market logic
    # natively allowes cash and asset_quantities to be negative
    # allowing short/margin trading by default. Use actions wrappers
    # to impose market logic.

    def __init__(
        self, 
        market_data_feeder: StaticDataFeeder,
        initial_cash: float = 1e6,
        initial_asset_quantities: Optional[np.ndarray] = None
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
        self.initial_asset_quantities = initial_asset_quantities
        self.cash = None
        self.net_worth = None
        self.asset_quantities = None
        self.holds = None # steps asset was held
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
            self.n_symbols,), dtype=np.float32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})
        
        return None
    

    def update_env(self) -> np.ndarray:

        self.features = next(self.row_generator)
        self.asset_prices = self.features[self.asset_price_mask]

        self.index += 1
        self.holds[self.asset_quantities != 0] += 1
        self.net_worth = self.cash + self.asset_quantities @ self.asset_prices

        return None


    def construct_observation(self) -> np.ndarray:
        
        observation = {
            'cash': self.cash,
            'asset_quantities': self.asset_quantities,
            'holds': self.holds,
            'features': self.features
            }
        
        return observation


    def place_orders(self, actions) -> None:

        for asset, action in enumerate(actions):

            if action == 0:
                continue

            quantity = action/self.asset_prices[asset]

            self.asset_quantities[asset] += quantity
            self.cash -= action
            self.holds[asset] = 0
        
        return None
        

    def reset(self) -> Dict:
                
        self.row_generator = self.data_feeder.reset()

        self.index = -1
        self.holds = np.zeros((self.n_symbols,), dtype=np.int32)

        self.cash = self.initial_cash

        self.asset_quantities = (
            np.zeros((self.n_symbols,), dtype=np.float32)
            if self.initial_asset_quantities is None
            else self.initial_asset_quantities)

        self.update_env()

        observation = self.construct_observation()

        return observation


    def step(
        self, 
        actions
        ) -> Tuple[Dict, np.float32, bool, Dict]:
        
        net_worth_ = self.net_worth
        self.place_orders(actions)
        self.update_env()
        
        reward = self.net_worth - net_worth_
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
        self.n_steps = float('inf')
        self.n_features = self.dataset_metadata.n_columns
        self.n_symbols = len(self.dataset_metadata.symbols)

        self.initial_cash = self.trader.initial_cash
        self.cash = None
        self.net_worth = None
        self.asset_quantities = None
        self.holds = None
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
            self.n_symbols,), dtype=np.float32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})
        
        return None
            

    def update_env(self) -> None:

        # this step will take time equal to dataset resolution to aggregate data stream
        self.features = next(self.row_generator)
        self.holds[self.asset_quantities != 0] += 1

        self.cash = self.trader.cash
        self.asset_quantities = self.trader.asset_quantities
        self.net_worth = self.trader.net_worth

        return None


    def construct_observation(self) -> Dict:

        observation = {
            'cash': self.cash,
            'asset_quantities': self.asset_quantities,
            'holds': self.holds,
            'features': self.features
        }

        return observation
    

    def place_orders(self, actions) -> None:

        self.trader.place_orders(actions)

        return None


    def reset(self) -> Dict:

        # resetting input state
        self.row_generator = self.data_feeder.reset()

        self.holds = np.zeros(
            (self.n_symbols,), dtype=np.int32)
        
        self.update_env()

        # compute state
        self.observation = self.construct_observation()

        return self.observation


    def step(
        self,
        actions: np.ndarray
        ) -> Tuple[Dict, np.float32, bool, Dict]:


        net_worth_ = self.net_worth
        self.place_orders(actions)
        self.update_env()

        reward = self.net_worth - net_worth_

        # returns states using env vars
        self.observation = self.construct_observation()


        return self.observation, reward, False, self.info
