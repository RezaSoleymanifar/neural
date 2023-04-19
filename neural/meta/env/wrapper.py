from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Type, Callable, Optional
from datetime import datetime

import numpy as np
from gym import (ActionWrapper, Env, Wrapper, 
    ObservationWrapper, RewardWrapper, spaces)

from neural.common.log import logger
from neural.common.exceptions import IncompatibleWrapperError
from neural.meta.env.base import AbstractMarketEnv, TrainMarketEnv, TradeMarketEnv

from neural.tools.ops import get_sharpe_ratio, tabular_print



def market(wrapper_class: Type[Wrapper]):

    # augments a wrapper class so that it checks if unwrapped base env
    # is a market env and creates a pointer to it. If search fails an error
    # is raised.

    if not issubclass(wrapper_class, Wrapper):

        raise TypeError(
            f"{wrapper_class} must be a subclass of {Wrapper}")

    class MarketEnvDependentWrapper(wrapper_class):

        def __init__(self, env: Env, *args, **kwargs) -> None:

            self.market_env = self.check_unwrapped(env)
            super().__init__(env, *args, **kwargs)

        def check_unwrapped(self, env):

            if not isinstance(env.unwrapped, AbstractMarketEnv):

                raise IncompatibleWrapperError(
                    f'{wrapper_class} requires unwrapped env to be of type {AbstractMarketEnv}.')
            
            return env.unwrapped
        
    return MarketEnvDependentWrapper



def metadata(wrapper_class: Type[Wrapper]):

    # augments a wrapper class so that it recursively checks for a metadata wrapper
    # in enclosed wrappers and creates a pointer to it. If search fails an error
    # is raised.

    if not issubclass(wrapper_class, Wrapper):

        raise TypeError(
            f"{wrapper_class} must be a subclass of {Wrapper}")

    class MarketMetadataWrapperDependentWrapper(wrapper_class):

        def __init__(self, env: Env, *args, **kwargs) -> None:

            self.market_metadata_wrapper = self.find_metadata_wrapper(env)
            super().__init__(env, *args, **kwargs)

            return None


        def find_metadata_wrapper(self, env):

            if isinstance(env, AbstractMarketEnvMetadataWrapper):
                return env

            if hasattr(env, 'env'):
                return self.find_metadata_wrapper(env.env)
            
            else:
                raise IncompatibleWrapperError(
                f'{wrapper_class} requires a wrapper of type '
                f'{AbstractMarketEnvMetadataWrapper} in enclosed wrappers.')


    return MarketMetadataWrapperDependentWrapper



class AbstractMarketEnvMetadataWrapper(Wrapper, ABC):

    # a blueprint class for market metadata wrappers.

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.history = defaultdict(list)
        self.initial_cash = self.market_env.initial_cash
        self.n_symbols = self.market_env.n_symbols
        self.cash = None
        self.asset_quantities = None
        self.positions = None
        self.net_wroth = None
        self.portfolio_value = None
        self.longs = None
        self.shorts = None
        self.profit = None
        self.return_ = None
        self.progress = None


    @abstractmethod
    def update_metadata(self, *args, **kwargs):

        raise NotImplementedError
        

    @abstractmethod
    def _cache_hist(self, *args, **kwargs):

        raise NotImplementedError


    def reset(self):

        observation = self.env.reset()
        self.update_metadata()

        return observation


    def step(self, action):

        observation, reward, done, info = self.env.step(action)
        self.update_metadata()

        return observation, reward, done, info



@market
class MarketEnvMetadataWrapper(AbstractMarketEnvMetadataWrapper):

    # Wraps a market env to track env metadata
    # enclosing wrappers can utilize this metadata

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        if isinstance(self.market_env, TradeMarketEnv):
            self.trader = self.market_env.trader


    def _upadate_train_env_metadata(self):

        self.asset_quantities = self.market_env.asset_quantities
        self.asset_prices = self.market_env.asset_prices
        self.net_worth = self.market_env.net_worth
        self.initial_cash = self.market_env.initial_cash
        self.cash = self.market_env.cash
        self.positions = self.asset_quantities * self.asset_prices

        short_mask = self.asset_quantities < 0
        long_mask = self.asset_quantities > 0

        self.longs = self.asset_quantities[
            long_mask] @ self.asset_prices[long_mask]
        self.shorts = self.asset_quantities[
            short_mask] @ self.asset_prices[short_mask]
        
        self.progress = self.market_env.index/self.market_env.n_steps


    def _update_trade_env_metadata(self):
            
            self.history = defaultdict(list)
            self.asset_quantities = self.trader.asset_quantities
            self.net_worth = self.trader.net_worth
            self.initial_cash = self.market_env.initial_cash
            self.cash = self.trader.cash

            self.positions = self.trader.positions
            self.longs = self.trader.longs
            self.shorts = self.trader.shorts

            now = datetime.now()
            self.progress = now.strftime("%Y-%m-%d %H:%M")


    def update_metadata(self):

        if isinstance(self.market_env, TrainMarketEnv):
            self._upadate_train_env_metadata()

        elif isinstance(self.market_env, TradeMarketEnv):
            self._update_trade_env_metadata()
        

        # common between trade and train market envs.
        self.profit = self.net_worth - self.initial_cash
        self.return_ = (self.net_worth - self.initial_cash)/self.initial_cash

        self._cache_hist()

        return None


    def _cache_hist(self):

        self.history['net_worth'].append(self.net_worth)

        return None
    


@market
@metadata
class ConsoleTearsheetRenderWrapper(Wrapper):

    # prints a tear sheet to console showing trading metadata.

    def __init__(
        self, env: Env,
        verbosity: int = 20
        ) -> None:

        super().__init__(env)
        self.verbosity = verbosity
        self.index = None

        if isinstance(self.market_env, TrainMarketEnv):
            self.render_every = self.market_env.n_steps//self.verbosity

        elif isinstance(self.market_env, TradeMarketEnv):
            self.render_every = 1 # every step


    def reset(self):

        observation = self.env.reset()
        self.index = 0

        resolution = self.market_env.dataset_metadata.resolution

        logger.info(
            'Episode details: '
            f'symbols = {self.market_env.n_symbols}, '
            f'resolution = {resolution}, '
            f'steps = {self.market_env.n_steps:,}, '
            f'features = {self.market_env.n_features:,}'
            )
        
        self.render()

        return observation
    

    def step(self, action):

        observation, reward, done, info = self.env.step(action)
        self.index += 1

        if self.index % self.render_every == 0 or done:
            self.render()

        return observation, reward, done, info


    def render(self, mode='human'):

        progress = self.market_metadata_wrapper.progress
        return_ = self.market_metadata_wrapper.return_
        sharpe = get_sharpe_ratio(self.market_metadata_wrapper.history['net_worth'])

        net_worth = self.market_metadata_wrapper.net_worth
        portfolio_value = sum(self.market_metadata_wrapper.positions)
        cash = self.market_metadata_wrapper.cash
        profit = self.market_metadata_wrapper.profit
        longs = self.market_metadata_wrapper.longs
        shorts = self.market_metadata_wrapper.shorts



        metrics = [f'{progress:.0%}', f'{return_:.2%}', f'{sharpe:.4f}',
            f'${profit:,.0f}', f'${net_worth:,.0f}', f'${cash:,.0f}',
            f'${portfolio_value:,.0f}',  f'${longs:,.0f}', f'${shorts:,.0f}']

        if self.index == 0:

            title = ['Progress', 'Return', 'Sharpe ratio',
                'Profit', 'Net worth', 'Cash',
                'Portfolio value', 'Longs', 'Shorts']
            
            print(tabular_print(title, header=True))

        print(tabular_print(metrics))

        return None



class MinTradeSizeActionWrapper(ActionWrapper):
    
    # ensures action wrappers before this would not modify 
    # actions of this wrapper.

    def __init__(self, env: Env, min_action = 1) -> None:

        super().__init__(env)
        self.min_action = min_action

        return None


    def action(self, actions):

        new_action = [
            action if abs(action) >= self.min_action 
            else 0 for action in actions]
    
        return new_action



class ActionClipperWrapper(ActionWrapper):

    # clips actions to expected range of downstream position
    # sizing wrappers.

    def __init__(self, env: Env, low=-1, high = 1) -> None:

        super().__init__(env)
        self.low = low
        self.high = high

    def action(self, actions):

        new_actions = np.clip(
            actions, self.low, self.high).tolist()

        return new_actions



@metadata
class NetWorthRelativeUniformPositionSizing(ActionWrapper):

    # ensures positions taken at each step is a maximum fixed percentage of net worth.
    # maps actions in (-1, 1) to buy/sell/hold using fixed zones for each action type.
    # trade_ratio = 0.02 means max of 2% of net_worth is traded at each step.
    # max_trade is max notional trade value for each asset at each step.
    # action in (-threshold, threshold) is parsed as hold.
    # action outside this range is linearly projected to (0, max_trade)

    def __init__(self, env: Env, trade_ratio = 0.02, hold_threshold = 0.15):

        super().__init__(env)
        self.trade_ratio = trade_ratio
        self.hold_threshold = hold_threshold
        self._max_trade_per_asset = None

        self.action_space = spaces.Box(
            low = -1, high = 1, shape = (self.n_symbols, ))
        
        return None
        

    def _set_max_trade_per_asset(self, trade_ratio: float) -> float:

        # sets value for self.max_trade
        # Recommended initial_cash >= n_stocks/trade_ratio. 
        # Trades bellow $1 is clipped to 1 (API constraint).

        self._max_trade_per_asset = (
            trade_ratio * self.market_metadata_wrapper.net_worth)/self.n_symbols

        return None


    def parse_action(self, action_: float) -> float:

        # action value in (-threshold, +threshold) is parsed as hold
        fraction = (abs(action_) - self.hold_threshold)/(
            1- self.hold_threshold)

        parsed_action =  fraction * self._max_trade_per_asset * np.sign(action_
            ) if fraction > 0 else 0
        
        return parsed_action


    def action(self, action):

        self._set_max_trade_per_asset(self.trade_ratio)

        new_actions = [self.parse_action(
            action) for action in action]
        
        return new_actions



@metadata
class NetWorthRelativeMaximumShortSizing(ActionWrapper):

    # class for sizing max shorts amount relative to net worth.
    # short_ratio = 0 means no shorting. short_ratio = 0.2 means 
    # shorts at maximum can be 20% of net_worth.
    # actions are partially fulfilled to the extent that short_ratio
    # is not violated. Must be applied ``before`` a position sizing wrapper.
    # ensure enclosed wrappers do not increase sell amount to 
    # guarantee adherence to short to net worth ratio.

    def __init__(self, env: Env, short_ratio = 0.2) -> None:

        super().__init__(env)
        self.short_ratio = short_ratio
        self.short_budget = None

        return None
    

    def _set_short_budget(self):

        max_short_size = self.short_ratio * max(self.market_metadata_wrapper.net_worth, 0)
        self.short_budget = max(max_short_size - abs(self.market_metadata_wrapper.shorts), 0)

        return None
    

    def action(self, actions):

        # performs actions without applying the effects and modifies actions
        # that would lead to short sizing limit violation.

        positions = self.market_metadata_wrapper.positions

        self._set_short_budget()

        for asset, action in enumerate(actions):
            
            if action > 0:
                continue

            # if short budget is zero and there is no owned assets ignore action
            if self.short_budget == 0 and positions[asset] <= 0:
                actions[asset] = 0
                continue

            # if sell amount exceeds current asset portfolio value shorting occurs
            sell = abs(action)

            new_short = abs(min(positions[asset] - sell, 0)) if positions[asset] >=0 else sell


            # modify sell amount if short excess over budget exists.          
            excess = max(new_short - self.short_budget, 0)
            sell -= excess
            new_short -= excess
            
            self.short_budget -= new_short
            actions[asset] = -sell

           
        return actions



@metadata
class FixedMarginActionWrapper(ActionWrapper):

    # Class for sizing maximum margin amount relative to net worth
    # Margin trading allows buying more than available cash using leverage. Positive cash is required
    # thus margin trading can only happen one asset at a time since orders are submitted asset by asset.
    # Initial_margin = 1 means no margin, namely entire purchase should me made with available cash only.
    # leverage = 1/inital_margin. initial_margin = 0.1 means only 10% of purchase value needs to be present
    # in account as cash thus maximum of 1/0.1 = 10 times cash, worth of assets can be purchased.
    # overall margin will have nominal effect on portfolio performance unless large purchase
    # of a single asset is involved.

    def __init__(self, env: Env, initial_margin= 1) -> None:

        super().__init__(env)
        self.initial_margin = initial_margin

        return None


    def action(self, actions):

        # performs actions without applying the effects and proactively modifies actions
        # that would lead to short sizing limit violation.
        cash = self.market_metadata_wrapper.cash

        for asset, action in enumerate(actions):
            
            # margin requires available cash. No cash means no margin.
            # sell actions are ignored due to having no effect on margin.
            if action <= 0:
                continue

            leverage = 1/self.initial_margin
            
            buy = min(action, leverage * cash)
            cash -= buy
            actions[asset] = buy

        return actions



@metadata
class IntegerAssetQuantityActionWrapper(ActionWrapper):

    # modifies actions to amount to integer number of shares
    # would not be valid in trade market env due to price
    # slippage. Ensure other action wrappers
    # applied before this would not modify the actions in a way that
    # asset quantities are not integer anymore.

    def __init__(self, env: Env, integer = True) -> None:

        super().__init__(env)
        self.integer = integer
        self.asset_prices = None

        return None


    def action(self, actions):

        if self.integer:
        
            asset_prices = self.market_metadata_wrapper.asset_prices

            for asset , action in enumerate(actions):
                asset_price = asset_prices[asset]
                action = (action // asset_price) * asset_price
                actions[asset] = action
        
        return actions



@metadata
class PositionsFeatureEngineeringWrapper(ObservationWrapper):

    # augments observations such that instead of asset quantities held
    # USD value of assets (positions) is used. Useful to reflect value of 
    # investements in each asset.

    def __init__(self, env: Env) -> None:

        super().__init__(env)
        self.n_symbols = self.market_metadata_wrapper.n_symbols

        self.observation_space = spaces.Dict({

            'cash': spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),

            'positions': spaces.Box(
                low=-np.inf, high=np.inf, shape=(
                    self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
                low=0, high=np.inf, shape=(
                    self.n_symbols,), dtype=np.int32),

            'features': spaces.Box(
                low=-np.inf, high=np.inf, shape=(
                    self.n_features,), dtype=np.float32)})
        
        return None
        

    def observation(self, observation):

        asset_prices= self.market_metadata_wrapper.asset_prices
        asset_quantities = observation.pop('asset_quantities')

        observation['positions'] = asset_prices * asset_quantities

        return observation
    


@metadata
class WealthAgnosticFeatureEngineeringWrapper(ObservationWrapper):

    # Augment observations so that net worth sensitive features
    # are now independent of net_worth. Apply immediately after
    # PositionsEngineeringWrapper.

    def __init__(self, env: Env) -> None:

        super().__init__(env)
        self.initial_cash = self.market_metadata_wrapper.initial_cash
        self.n_symbols = self.market_metadata_wrapper.n_symbols

        self.observation_space = spaces.Dict({

            'cash':spaces.Box(
            low=-np.inf, high=np.inf, shape = (1,), dtype=np.float32),

            'positions': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_symbols,), dtype=np.float32),

            'holds': spaces.Box(
            low=0, high=np.inf, shape = (
            self.n_symbols,), dtype=np.int32),
            
            'features': spaces.Box(
            low=-np.inf, high=np.inf, shape = (
            self.n_features,), dtype=np.float32),

            'return': spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),})
        
        return None
    

    def observation(self, observation):

        net_worth = self.market_metadata_wrapper.net_worth

        # these features now relative to net_worth making the range
        # of values they take independent from net_worth.

        observation['positions'] /= net_worth
        observation['cash'] /= net_worth
        observation['return'] = net_worth/self.initial_cash - 1

        return observation



class ObservationBufferWrapper(ObservationWrapper):

    # a temporary buffer of observations for subsequent wrappers
    # that require this form of information.

    def __init__(self, env: Env, buffer_size = 10) -> None:

        super().__init__(env)

        return None



class FlattenDictToNumpyObservationWrapper(ObservationWrapper):

    # flattens dictionary of observations to numpy array.

    def __init__(self, env: Env) -> None:

        super().__init__(env)

        return None


    def observation(self, observation):

        observation = np.concatenate(list(observation.values()), axis=None)

        return observation
    


class RunningIndicatorsObsWrapper(ObservationWrapper):

    # computes running financial indicators such as CCI, MACD
    # etc. Requires an observations buffer containing a window 
    # of consecutive observations.

    def observation(self, observation):
        return None



class NormalizeObservationsWrapper(ObservationWrapper):
    pass



class NormalizeRewardsWrapper(RewardWrapper):

    # normalizes rewards of an episode

    pass



class ExperienceRecorderWrapper(Wrapper):

    # saves experiences in bufferand writes to hdf5 when 
    # buffer reaches a certain size.

    pass



class RewardShaperWrapper(Wrapper):
    
    # highly useful for pretraining an agent with multiple degrees of freedom 
    # in actions. Apply relevant reward shaping wrappers to 

    pass