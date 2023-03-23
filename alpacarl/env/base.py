import pandas as pd
import gym
import numpy as np
from alpacarl.meta import log
from alpacarl.aux.tools import sharpe, return_

class TrainEnv(gym.Env):
    def __init__(self, prices: np.ndarray, features: np.ndarray,\
                  cash: float = 1e6, min_trade = 1, trade_ratio: float = 2e-2) -> None:
        self.prices = prices
        self.features = features
        self.index = None
        self.init_cash = None
        self.cash = None
        self.assets = None
        self.steps, self.n_stocks = prices.shape
        self.stocks = None #quantity
        self.holds = None
        self.min_trade = min_trade
        self.max_trade = self._max_trade(trade_ratio)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32)
        # state = (cash, self.stocks * self.prices[self.index], self.holds, self.features[self.index])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (1 + self.n_stocks + self.n_stocks\
                                                       + self.features.shape[1], ), dtype=np.float32)
        
    def _parse_action(self, action: float, threshold = 0.15) -> float:
        #  action value in (-threshold, +threshold) is parsed as hold
        fraction = (abs(action) - threshold)/(1- threshold)
        return fraction * self.max_trade * np.sign(action) if fraction > 0 else 0
   
    def _state(self) -> np.ndarray:
        # scaling that is agnosic to initial_cash value.
        cash_ = self.cash / self.init_cash

        # value of stocks scaled by initial cash
        stocks_ = (self.stocks * self.prices[self.index]) / self.init_cash

        # sacling ordinal holds. saturates after 1e3 steps of holding.
        holds_ = np.tanh(self.holds * 1e-3)
        state = np.hstack(cash_, stocks_, holds_, self.features[self.index])
        return state
    
    def _max_trade(self) -> float:
        # Default: 2% of initial cash per trade per stocks
        # Recommended initial_cash >= n_stocks/trade_ratio. Trades bellow $1 is clipped to 1 (API constraint).
        max_trade = (self.trade_ratio * self.init_cash)/self.n_stocks
        return max_trade

    def reset(self):
        self.idex = 0
        self.cash = self.init_cash
        self.stocks = np.zeros(self.n_symbols, dtype=np.float64)
        self.holds = np.zeros(self.n_symbols, dtype=np.int32)
        self.assets = self.cash
        return self._state()

    def step(self, actions):
        # 0 or some value in (-self.max_trade, +self.max_trade)
        parsed_actions = [self._parse_action(action) for action in actions]
        self.index += 1

        # increase hold time of purchased stocks
        self.holds[self.stocks > 0] += 1

        # iterate over parsed actions and execute.
        for stock, action in enumerate(parsed_actions):
            if action > 0 and self.cash > 0: # buy
                buy = min(self.cash, action)
                buy = min(self.min_trade, buy)
                quantity = buy/self.prices[self.index][stock]
                self.stocks[stock] += quantity
                self.cash =- buy
            elif action < 0 and self.stocks[stock] > 0: # sell
                sell = min(self.stocks[stock] * self.prices[self.index][stock], abs(action))
                quantity = sell/self.prices[self.index][stock]
                self.stocks[stock] =- quantity
                self.cash =+ sell
                self.holds[stock] = 0

        assets = self.cash + self.stocks @ self.prices[self.index]

        # rewards are agnostic to initial cash value
        reward = (assets - self.assets)/self.init_cash
        self.assets = assets

        # report terminal state
        done = self.index == self.steps - 1

        # log portfolio performance
        if self.index % (self.steps//10):
            self.render()
        return self._state(), reward, done, dict()
        
    def render():
        # sharpe ratio filters volatility to reflect investor skill
        sharpe = None
        return_ = None
        log.logger.info(f'Return: {return_:>3.2f}, Sharpe ratio: {sharpe:>3.2f},\
                         assets: ${self.assets:>3,.2f}, cash: ${self.cash:>3.2f}')