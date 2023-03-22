import pandas as pd
import gym
import numpy

class BaseEnv(gym.Env):
    def __init__(self, prices: np.ndarray, features: np.ndarray, cash: float = 1e6, trade_ratio: float = 2e-2) -> None:
        self.prices = prices
        self.features = features
        self.index = 0
        self.init_cash = cash
        self.cash = cash
        self.assets = cash
        self.n_stocks, self.steps = prices.shape
        self.stocks = np.zeros(self.n_stocks)
        self.holds = np.zeros(self.n_stocks)
        self.max_trade = self.get_max_trade(trade_ratio)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32)
        # state = (cash, self.stocks * self.prices, self.holds, self.features) normalized relatively
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,\
                                             shape = (1 + len(self.stocks) + len(self.holds)\
                                                       + self.n_features.shape[1], ), dtype=np.float32)

    def parse_action(self, action: float, threshold = 0.15) -> float:
        #action values inside (-threshold, threshold) are parsed as hold
        fraction = (abs(action) - threshold)/(1- threshold)
        return fraction * self.max_trade * np.sign(action) if fraction > 0 else 0

        
    def state(self):
        state = np.hstack(self.cash / self.init_cash,\
                           (self.stocks * self.prices) / self.init_cash, self.holds / self.steps, self.features)
        return state
    
    def get_max_trade(self):
        # usually 2% of initial cash per trade divided by number of stocks
        max_trade = (self.trade_ratio * self.init_cash)/self.n_stocks
        return max_trade

    def reset():
        self.idex = 0
        self.cash = self.init_cash
        self.stocks = np.zeros(self.n_symbols)
        self.holds = np.zeros(self.n_symbols)
        self.assets = cash
        return self.state_mapper.states(self.cash, self.stocks, self.holds, self.features)

    def step(self, actions):
        parsed_actions = [self.parse_action(action) for action in actions]
        self.index += 1
        self.holds[self.stocks > 0] += 1
        for stock, action in enumerate(parsed_actions):
            if action > 0 and self.cash > 0: # buy
                buy = min(self.cash, action)
                quantity = buy/self.prices[stock]
                self.stocks[stock] += quantity
                self.cash =- buy
            elif action < 0 and self.stocks[stock] > 0: # sell
                sell = min(self.stocks[stock] * self.prices[stock], abs(action))
                quantity = sell/self.prices[stock]
                self.stocks[stock] =- quantity
                self.cash =+ sell
                self.holds[stock] = 0

        assets = self.cash + self.stocks @ self.prices[self.index]
        reward = (assets - self.assets)/self.init_cash
        self.assets = assets
        done = self.index == self.steps
        return self.state(), reward, done, dict()

    def render():
        pass

class BaseTradeEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

class ActionParser:
    def __init__(self, max_trade: float, threshold: float = 0.15) -> None:
        self.max_trade = max_trade
        self.threshold = threshold

class Scaler:
    def __init__(self,  cash_scale = 3e-6, stock_scale = 1e-6, hold_scale = 5e-4, feature_scale = 1e-3, reward_scale = 1e-3) -> None:
        self.cash_scale = cash_scale
        self.stock_scale = stock_scale
        self.hold_scale = hold_scale
        self.feature_scale = feature_scale


    
    def reward(self, reward)

class RewardShaper:
    def __init__(self) -> None:
        pass