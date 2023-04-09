from collections import defaultdict
from typing import Tuple

import numpy as np
from gym import spaces, Env

from neural.common.log import logger
from neural.tools.ops import sharpe, tabular_print
from neural.core.data.ops import StaticDataFeeder, AsyncDataFeeder
from neural.core.data.enums import ColumnType


class BaseEnv(Env):
    def __init__(
        self, 
        data_feeder: StaticDataFeeder | AsyncDataFeeder,
        init_cash: float = 1e6,
        min_trade: float = 1, 
        verbose: bool = True
        ) -> None:
        
        if data_feeder is None:
            logger.error(
                ("data_feeder must be provided."))
            raise TypeError
        else:
            self.data_feeder = self.data_feeder

        self.dataset_metadata = self.data_feeder.dataset_metadata
        self.column_schema = self.dataset_metadata.column_schema
        self.asset_price_mask = self.dataset_metadata.column_schema[ColumnType.CLOSE]
        self.steps = self.dataset_metadata.n_rows
        self.n_features = self.dataset_metadata.n_columns
        self.n_symbols = len(self.dataset_metadata.symbols)

        self.index = None
        self.init_cash = init_cash
        self.cash = None
        self.net_worth = None # cash + assets (stock/crypto)
        self.asset_quantities = None # quantity of each asset held
        self.holds = None # steps asset did not have trades
        self.min_trade = min_trade

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(
            self.n_symbols,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'cash':spaces.Box(
            low=0, high=np.inf, shape = (1,), dtype=np.float32),

            'positions': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.int32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32)})
        
        self.history = defaultdict(list)
        self.verbose = verbose
        self.render_every = self.steps//20
    
    def prices_and_features_generator(
        self
        ) -> Tuple[np.ndarray, np.ndarray]:

        features = next(self.row_generator)
        features.astype(np.float32)
        prices = features[self.asset_price_mask]
        
        yield prices, features

    def _get_env_state(self) -> np.ndarray:
        
        state = {
            'cash': self.cash,
            'asset_quantities': self.asset_quantities,
            'holds': self.holds,
            'features': self.features
            }
        
        return state

    def _cache_hist(self):
        self.history['assets'].append(self.net_worth)

    def reset(self):
        # resetting input state
        self.row_generator = self.data_feeder.reset()
        
        logger.info(
            f'Steps: {self.steps}, symbols: {self.n_symbols}, features: {self.n_features}')
        
        self.index = 0
        self.prices, self.features = next(
            self.prices_and_features_generator())
        self.cash = self.init_cash
        self.asset_quantities = np.zeros((self.n_symbols,), dtype=np.float32)
        self.holds = np.zeros((self.n_symbols,), dtype=np.int32)
        self.net_worth = self.cash

        # cache env history
        self._cache_hist()
        # compute state
        self.state = self._get_env_state()
        return self.state

    def step(self, actions):
        # iterates over actions iterable
        for asset, action in enumerate(actions):

            if action > 0 and self.cash > 0: # buy

                buy = min(self.cash, action)
                buy = max(self.min_trade, buy)
                quantity = buy/self.prices[asset]

                self.asset_quantities[asset] += quantity
                self.cash -= buy

            elif action < 0 and self.asset_quantities[asset] > 0: # sell
                
                sell = min(
                    self.asset_quantities[asset] * self.prices[asset], abs(action))
                quantity = sell/self.prices[asset]

                self.asset_quantities[asset] -= quantity
                self.cash += sell
                self.holds[asset] = 0

        self.prices, self.features = next(
            self.prices_and_features_generator())
        
        # increase hold time of purchased stocks
        self.holds[self.asset_quantities > 0] += 1
        # new asset value
        assets = self.cash + self.asset_quantities @ self.prices
        reward = assets - self.net_worth
        self.net_worth = assets
        # comoputes states using env vars
        self.state = self._get_env_state()
        self._cache_hist()

        # report terminal state
        done = self.index == self.steps - 1

        # tabulate decision maker performance
        if (self.verbose and self.index % self.render_every == 0) or done:
            self.render(done)
            
        return self.state, reward, done, self.history
        
    def render(self, done:bool = False) -> None:

        # print header at first render
        if self.index == self.render_every:

            # print results in a tear sheet format
            print(tabular_print(
                ['Progress', 'Return','Sharpe ratio',
                'Assets', 'Positions', 'Cash'], header = True))
        
        # total value of positions in portfolio
        positions_ = self.asset_quantities @ self.prices
        return_ = (self.net_worth-self.init_cash)/self.init_cash

        # sharpe ratio filters volatility to reflect investor skill
        sharpe_ = sharpe(self.history['assets'])
        progress_ = self.index/self.steps

        metrics = [f'{progress_:.0%}', f'{return_:.2%}', f'{sharpe_:.4f}',
            f'${self.net_worth:,.2f}', f'${positions_:,.2f}', f'${self.cash:,.2f}']
        
        # add performance metrics to tear sheet
        tabular_print(metrics)
                
        if done:
            logger.info('Episode terminated.')
            logger.info(*metrics)
        return None