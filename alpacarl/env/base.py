import pandas as pd
import gym
import numpy

class BaseEnv(gym.Env):
    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, cash: float = 1e6, max_buy = None) -> None:
        self.prices = prices
        self.features = features
        self.index = 0
        self.cash = cash
        self.n_symbols, self.steps = prices.shape
        self.n_features = features.shape[1]
        self.stocks = np.zeros(self.n_symbols)
        self.holds = np.zeros(self.n_symbols)
        self.max_buy = min((0.02 * capital)/len(self.assets), 1) if not max_buy else max_sell
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_symbols,), dtype=np.float32)
        # state = (capital, self.stocks, self.holds, self.features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,\
                                             shape = (1 + len(self.assets) + len(self.holds) + self.n_features, ), dtype=np.float32)

    def reset():
        pass
    def step():
        pass
    def render():
        pass
    def states():
        pass

class ActionParser:
    def __init__(self, max_allowed: ) -> None:
        pass

    def fractional(action: float, max_buy: float, max_sell: float, threshold: float = 0.15) -> float:
        fraction = (abs(action) - threshold)/(1- threshold)
        # if fraction > 0:
        #     # simulates buy/sell action
        #     return fraction * max_buy if action > 0 else fraction * max_sell
        # else:
        #     # simulates a hold action
        #     return 0
