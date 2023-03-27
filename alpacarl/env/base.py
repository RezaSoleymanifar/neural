import pandas as pd
from gym import spaces, Env
import numpy as np
from alpacarl.meta import log
from alpacarl.aux.tools import sharpe, print_
from collections import defaultdict


class BaseEnv(Env):
    def __init__(self, prices: pd.DataFrame, features: np.ndarray,\
                  init_cash: float = 1e6, min_trade = 1, trade_ratio: float = 2e-2) -> None:
        self.prices = prices
        self.features = features
        self.index = None
        self.init_cash = init_cash
        self.cash = None
        self.assets = None
        self.steps, self.n_stocks = prices.shape
        self.positions = None #quantity
        self.holds = None
        self.min_trade = min_trade
        self.max_trade = self._max_trade(trade_ratio)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32)
        # state = (cash, self.positions * self.prices[self.index], self.holds, self.features[self.index])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (1 + self.n_stocks + self.n_stocks\
                                                       + self.features.shape[1], ), dtype=np.float32)
        self.history = defaultdict(list)
        
    def _parse_action(self, action: float, threshold = 0.15) -> float:
        # action value in (-threshold, +threshold) is parsed as hold
        fraction = (abs(action) - threshold)/(1- threshold)
        return fraction * self.max_trade * np.sign(action) if fraction > 0 else 0
   
    def _state(self) -> np.ndarray:
        # scaling that is agnosic to initial_cash value.
        cash_ = self.cash / self.init_cash

        # value of positions scaled by initial cash
        positions_ = (self.positions * self.prices.iloc[self.index]) / self.init_cash

        # scaling integer hold values. saturates after 1e3 steps of holding.
        holds_ = np.tanh(self.holds * 1e-3)
        state = np.hstack([cash_, positions_, holds_, self.features[self.index]])
        return state
    
    def _max_trade(self, trade_ratio: float) -> float:
        # sets value for self.max_trade
        # Default: 2% of initial cash per trade per stocks
        # Recommended initial_cash >= n_stocks/trade_ratio. Trades bellow $1 is clipped to 1 (API constraint).
        max_trade = (trade_ratio * self.init_cash)/self.n_stocks
        return max_trade

    def _update_hist(self):
        self.history['assets'].append(self.assets)

    def reset(self):
        log.logger.info(f'Initializing environment. Steps in environment: {self.steps:,}, symbols: {self.n_stocks}')

        # initializing env vars
        self.index = 0
        self.cash = self.init_cash
        self.positions = np.zeros(self.n_stocks, dtype=np.float64)
        self.holds = np.zeros(self.n_stocks, dtype=np.int32)
        self.assets = self.cash

        # update env vars history
        self._update_hist()
        self.render()
        return self._state()

    def step(self, actions):
        # parsed action is 0 or some value in (-self.max_trade, +self.max_trade)
        parsed_actions = [self._parse_action(action) for action in actions]

        # iterate over parsed actions and execute.
        for stock, action in enumerate(parsed_actions):
            if action > 0 and self.cash > 0: # buy
                buy = min(self.cash, action)
                buy = max(self.min_trade, buy)
                quantity = buy/self.prices.iloc[self.index][stock]
                self.positions[stock] += quantity
                self.cash -= buy
            elif action < 0 and self.positions[stock] > 0: # sell
                sell = min(self.positions[stock] * self.prices.iloc[self.index][stock], abs(action))
                quantity = sell/self.prices.iloc[self.index][stock]
                self.positions[stock] -= quantity
                self.cash += sell
                self.holds[stock] = 0

        # next state
        self.index += 1
        # increase hold time of purchased stocks
        self.holds[self.positions > 0] += 1
        # new asset value
        assets = self.cash + self.positions @ self.prices.iloc[self.index]
        # rewards are agnostic to initial cash value
        reward = (assets - self.assets)/self.init_cash
        self.assets = assets
        self._update_hist()

        # report terminal state
        done = self.index == self.steps - 1

        # log strategy performance
        if not self.index % (self.steps//20) or done:
            self.render(done)
        return self._state(), reward, done, self.history
        
    def render(self, done:bool = False) -> None:
        # print header at env start
        if not self.index:
            # print results in a tear sheet format
            print_(['Progress', 'Return','Sharpe ratio', 'Assets', 'Positions', 'Cash'], header = True)
            return None
        
        # value of positions in portfolio
        positions_ = self.positions @ self.prices.iloc[self.index]
        return_ = (self.assets-self.init_cash)/self.init_cash

        # sharpe ratio filters volatility to reflect investor skill
        sharpe_ = sharpe(self.history['assets'])
        progress_ = self.index/self.steps

        # add performance metrics to tear sheet
        print_([f'{progress_:.0%}', f'{return_:.2f}', f'{sharpe_:.2f}',\
                                f'${self.assets:,.2f}', f'${positions_:,.2f}', f'${self.cash:,.2f}'])
                
        if done:
            log.logger.info('Episode terminated.')
            log.logger.info(f'Progress: {progress_:<.0%}, return: {return_:<.2f}, Sharpe ratio: {sharpe_:<.2f} ' \
                            f'assets: ${self.assets:<,.2f}, positions: ${positions_:<,.2f},  cash: ${self.cash:<,.2f}')
        return None