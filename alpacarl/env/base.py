import pandas as pd
from gym import spaces, Env, ActionWrapper
import numpy as np
from alpacarl.meta import log
from alpacarl.aux.tools import sharpe, print_
from alpacarl.core.data import PeekableDataWrapper, row_generator
from collections import defaultdict


class BaseEnv(Env):
    def __init__(self, data: pd.DataFrame = None, dir: str = None, chunk: int = None,\
                  init_cash: float = 1e6, min_trade = 1, trade_ratio: float = 2e-2) -> None:

        if data is not None and dir is not None:
            log.logger.error(("Cannot provide both file path and file object simultaneously."))
            raise ValueError
        elif data is None and dir is None:
            log.logger.error(("Either file path or a file object must be provided."))
            raise ValueError
        elif self.dir is not None:
            self.data = PeekableDataWrapper(row_generator(dir, chunk))
        else:
            self.data = PeekableDataWrapper(data)
        
        self.index = None
        self.init_cash = init_cash
        self.cash = None
        self.assets = None
        self.steps, self.n_stocks, self.n_features = self.summarize()
        self.positions = None #quantity
        self.holds = None
        self.min_trade = min_trade
        self.max_trade = self._max_trade(trade_ratio)
        # abs(action) = sell/buy amount, sign(action) = +1 buy, -1 sell. Action = 0 is hold.
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_stocks,), dtype=np.float32)
        # state = (cash, self.positions, self.holds, self.features[self.index])
        self.observation_space = spaces.Dict({
            'cash':spaces.Box(low=0, high=np.inf, shape = (1,), dtype=np.float32),
            'positions': spaces.Box(low=0, high=np.inf, shape = (self.n_stocks,), dtype=np.int32),
            'holds': spaces.Box(low=0, high=np.inf, shape = (self.n_stocks,), dtype=np.int32),
            'features': spaces.Box(low=-np.inf, high=np.inf, shape = (self.n_features,), dtype=np.float32)
        })

        self.history = defaultdict(list)

    def summarize(self):
        steps = len(self.data)
        row = self.data.peek()
        n_stocks = len(row['close'])
        n_features = len(row)
        return steps, n_stocks, n_features
   
    def _state(self) -> np.ndarray:
        
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
        if self.data is not None:
            if dir is not None:
                ra
            self.data = DataWrapper(self.data)
        
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
    
    class FractionalActionWrapper(ActionWrapper):
        # maps actions in (-1, 1) to buy/sell/hold actions
        def __init__(self, env: Env, threshold = 0.15):
            super().__init__(env)
            self.threshold = threshold

        def _parse_action(self, action: float) -> float:
            # action value in (-threshold, +threshold) is parsed as hold
            fraction = (abs(action) - self.threshold)/(1- self.threshold)
            return fraction * self.max_trade * np.sign(action) if fraction > 0 else 0
        
        def action(self, action):
            pass