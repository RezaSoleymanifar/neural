from gym import ActionWrapper
import numpy as np


class FractionalActionWrapper(ActionWrapper):
    # maps actions in (-1, 1) to buy/sell/hold
    def __init__(self, env: Env, ratio = 0.02):
        super().__init__(env)
        base_env = env.unwrapped
        self.max_trade = self._set_max_trade(ratio)
        self.n_symbols = base_env.n_symbols
        self.init_cash = base_env.init_cash
        self.action_space = spaces.Box(
            low = -1, high = 1, shape = (self.n_symbols, ))
        self.threshold = 0.15
    
    def _set_max_trade(self, trade_ratio: float) -> float:
        # sets value for self.max_trade
        # Default: 2% of initial cash per trade per stocks
        # Recommended initial_cash >= n_stocks/trade_ratio. Trades bellow $1 is clipped to 1 (API constraint).
        max_trade = (trade_ratio * self.init_cash)/self.n_symbols
        return max_trade

    def action(self, action: float, threshold = 0.15) -> float:
        # action value in (-threshold, +threshold) is parsed as hold
        fraction = (abs(action) - self.threshold)/(1- self.threshold)
        return fraction * self.max_trade * np.sign(action) if fraction > 0 else 0
        