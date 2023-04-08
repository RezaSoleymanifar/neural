from collections import defaultdict
from typing import Tuple

import numpy as np
from gym import spaces, Env

from neural.common import log
from neural.tools.ops import sharpe, tabular_print
from neural.core.data.ops import RowGenerator
from neural.core.data.enums import ColumnType


class BaseEnv(Env):
    def __init__(
        self, 
        data_generator: RowGenerator, 
        init_cash: float = 1e6,
        min_trade: float = 1, 
        verbose: bool = True
        ) -> None:
        
        if data_generator is None:
            log.logger.error(
                ("data_generator must be provided."))
            raise ValueError
        else:
            self.data_generator = data_generator

        self.dataset_metadata = self.deta_generator.dataset_metadata
        self.column_schema = self.dataset_metadata.column_schema
        self.asset_price_mask = self.dataset_metadata.column_schema[ColumnType.CLOSE]
        self.index = None
        self.init_cash = init_cash
        self.cash = None
        self.assets = None # cash + stocks/crypto
        self.steps = self.deta_generator.n_rows
        self.n_features = self.data_generator.n_columns
        self.n_symbols = len(self.dataset_metadata.symbols)   
        self.stocks = None # quantity of each stock held
        self.holds = None # steps stock did not have trades
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
        self.render_every = self.step//20
    
    def prices_and_features_generator(
        self
        ) -> Tuple[np.ndarray, np.ndarray]:

        features = next(self.deta_generator)
        features.astype(np.float32)
        prices = features[self.asset_price_mask]
        
        yield prices, features

    def _get_env_state(self) -> np.ndarray:

        positions = self.stocks * self.prices
        state = np.hstack(
            [self.cash, positions, self.holds, self.features])
        return state

    def _cache_hist(self):
        self.history['assets'].append(self.assets)

    def reset(self):
        # resetting input state
        self.deta_generator.reset()
        
        log.logger.info(
            f'Steps: {self.steps}, symbols: {self.n_symbols}, features: {self.n_features}')
        
        self.index = 0
        self.prices, self.features = next(
            self.prices_and_features_generator())
        self.cash = self.init_cash
        self.stocks = np.zeros((self.n_symbols,), dtype=np.int32)
        self.holds = np.zeros((self.n_symbols,), dtype=np.int32)
        self.assets = self.cash

        # cache env history
        self._cache_hist()
        # compute state
        self.state = self._get_env_state()
        return self.state

    def step(self, actions):
        # iterates over actions iterable
        for stock, action in enumerate(actions):

            if action > 0 and self.cash > 0: # buy

                buy = min(self.cash, action)
                buy = max(self.min_trade, buy)
                quantity = buy/self.prices[stock]

                self.stocks[stock] += quantity
                self.cash -= buy

            elif action < 0 and self.stocks[stock] > 0: # sell

                sell = min(
                    self.stocks[stock] * self.prices[stock], abs(action))
                quantity = sell/self.prices[stock]

                self.stocks[stock] -= quantity
                self.cash += sell
                self.holds[stock] = 0

        self.prices, self.features = next(
            self.prices_and_features_generator())
        
        # increase hold time of purchased stocks
        self.holds[self.stocks > 0] += 1
        # new asset value
        assets = self.cash + self.stocks @ self.prices
        reward = assets - self.assets
        self.assets = assets
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
        positions_ = self.stocks @ self.prices
        return_ = (self.assets-self.init_cash)/self.init_cash

        # sharpe ratio filters volatility to reflect investor skill
        sharpe_ = sharpe(self.history['assets'])
        progress_ = self.index/self.steps

        metrics = [f'{progress_:.0%}', f'{return_:.2%}', f'{sharpe_:.4f}',
            f'${self.assets:,.2f}', f'${positions_:,.2f}', f'${self.cash:,.2f}']
        
        # add performance metrics to tear sheet
        tabular_print(metrics)
                
        if done:
            log.logger.info('Episode terminated.')
            log.logger.info(*metrics)
        return None