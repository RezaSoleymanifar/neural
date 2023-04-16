from stable_baselines3.common.callbacks import BaseCallback
from gym import ActionWrapper, Env, spaces, Wrapper, ObservationWrapper, RewardWrapper
import numpy as np
from neural.common.log import logger
from neural.tools.ops import sharpe, tabular_print
from neural.meta.env.base import AbstractMarketEnv, TrainMarketEnv, TradeMarketEnv
from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Type, Callable, Optional
from neural.common.exceptions import IncompatibleWrapperError
from datetime import datetime
from neural.connect.client import AlpacaMetaClient
from neural.core.trade.ops import CustomAlpacaTrader
from gym.wrappers import NormalizeReward


class AbstractConstrainedWrapper(ABC):

    @abstractmethod
    def check_constraint(self, env):

        raise NotImplementedError



def constraint(
    constraint_function: Callable,
    constrained_type=Type[Env],
    constraint_attr: Optional[str] = None):

    if not issubclass(constrained_type, Env):

        raise TypeError(
            f"{constrained_type} must be a subclass of {Env}")

    def constraint_decorator(wrapper_class: Type[Wrapper]):

        if not issubclass(wrapper_class, Wrapper):
            raise TypeError(
                f"{wrapper_class} must be a subclass of {Wrapper}")

        class ConstrainedWrapper(
                wrapper_class, AbstractConstrainedWrapper):

            def __init__(self, env: Env, *args, **kwargs) -> None:
                # constraint check starts at env passed to constructor
                if constraint_attr is not None and isinstance(constraint_attr, str):
                    setattr(self, constraint_attr, self.check_constraint(env))

                else:
                    self.check_constraint(env)

                super().__init__(env, *args, **kwargs)

            def check_constraint(self, env):
                return constraint_function(env, constrained_type)

        return ConstrainedWrapper

    return constraint_decorator



def unwrapped_is(wrapper_in_constructor, constrained_type):

    unwrapped_env = wrapper_in_constructor.unwrapped

    if not isinstance(unwrapped_env, constrained_type):

        raise IncompatibleWrapperError(
            f'{wrapper_in_constructor} requires {unwrapped_env} '
            f'to be of type {constrained_type}.'
            )

    return unwrapped_env


def unwrapped_is_not(wrapper_in_constructor, constrained_type):

    unwrapped_env = wrapper_in_constructor.unwrapped

    if  isinstance(unwrapped_env, constrained_type):

        raise IncompatibleWrapperError(
            f'{wrapper_in_constructor} is incompatible with {unwrapped_env} '
            f'of type {constrained_type}.'
        )

    return unwrapped_env

def wraps(wrapper_in_constructor, constrained_type):

    if  hasattr(wrapper_in_constructor, 'env'):

        raise IncompatibleWrapperError(
            f'The enclosing wrapper requires {wrapper_in_constructor} '
            f' be of type {constrained_type}.')

    return wrapper_in_constructor

def before(wrapper_at_constructor, constrained_type):

    def before_helper(wrapper, wrapper_at_constructor):

        if hasattr(wrapper, 'env'):
            if isinstance(wrapper.env, constrained_type):

                raise IncompatibleWrapperError(
                    f'{wrapper.env} of type {constrained_type} '
                    f'is applied before enclosing constrained wrapper {wrapper_at_constructor}.'
                )
            else:
                return before_helper(wrapper_at_constructor.env, wrapper_at_constructor)

    return before_helper(wrapper_at_constructor, wrapper_at_constructor)




def requires(wrapper_at_constructor, constrained_type):

    def requires_helper(wrapper, wrapper_at_constructor):

        if not hasattr(wrapper, 'env'):

            raise IncompatibleWrapperError(
                f'{wrapper_at_constructor} requires wrapper of type {constrained_type} '
                f'to exist in the underlying wrappers.')

        elif isinstance(wrapper.env, constrained_type):
            return wrapper.env

        else:
            return requires_helper(wrapper.env, wrapper_at_constructor)

    return requires_helper(wrapper_at_constructor, wrapper_at_constructor)



class AbstractMarketEnvMetadataWrapper(Wrapper, ABC):
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
        self.sharpe = None
        self.progress = None


    @abstractmethod
    def _update_metadata(self, *args, **kwargs):
        raise NotImplementedError
        
    @abstractmethod
    def _cache_hist(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):

        observation = self.env.reset()
        self._update_metadata()

        return observation

    def step(self, action):

        observation, reward, done, info = self.env.step(action)
        self._update_metadata()

        return observation, reward, done, info



@constraint(unwrapped_is, TrainMarketEnv, 'market_env')
class TrainMarketEnvMetadataWrapper(AbstractMarketEnvMetadataWrapper):
    # wraps a market env to track env metadata
    # upstream wrappers can utilize this metadata
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def _update_metadata(self):

        self.asset_quantities = self.market_env.asset_quantities
        self.asset_prices = self.market_env.asset_prices
        self.positions = self.asset_quantities * self.asset_prices
        self.net_worth = self.market_env.net_worth
        self.initial_cash = self.market_env.initial_cash

        short_mask = self.asset_quantities < 0
        long_mask = self.asset_quantities > 0

        # total value of portfolio
        self.portfolio_value = self.asset_quantities @ self.asset_prices


        self.longs = self.asset_quantities[long_mask] @ self.asset_prices[long_mask]
        self.shorts = self.asset_quantities[short_mask] @ self.asset_prices[short_mask]
        
        # sharpe ratio filters volatility to reflect investor skill
        
        self.profit = self.net_worth - self.initial_cash
        self.return_ = (self.net_worth - self.initial_cash)/self.initial_cash

        self._cache_hist()

        self.sharpe = sharpe(self.history['net_worth'])
        self.progress = self.market_env.index/self.market_env.n_steps


    def _cache_hist(self):

        self.history['assets'].append(self.net_worth)

        return None


@constraint(unwrapped_is, TradeMarketEnv, 'market_env')
class AlpacaTradeEnvMetadataWrapper(AbstractMarketEnvMetadataWrapper):

    def __init__(self, env: Env) -> None:
        super().__init__(env)

        self.trader = self.market_env.trader

        if not isinstance(self.trader, CustomAlpacaTrader):

            raise IncompatibleWrapperError(
                f'Market env {self.market_env} must have trader of type {CustomAlpacaTrader}.')
        


    def _update_metadata(self):

        self.history = defaultdict(list)
        self.asset_quantities = self.trader.asset_quantities
        self.positions = self.trader.positions
        self.net_worth = self.trader.net_worth
        self.initial_cash = self.market_env.initial_cash
        self.cash = self.trader.cash

        self.portfolio_value = self.client.portfolio_value
        self.longs = self.client.account.long_market_value
        self.shorts = self.client.account.short_market_value
        
        self.profit = self.net_worth - self.initial_cash
        self.return_ = (self.net_worth - self.initial_cash)/self.initial_cash

        self._cache_hist()

        self.sharpe_ = sharpe(self.history['net_worth'])

        now = datetime.now()
        self.progress = now.strftime("%Y-%m-%d %H:%M")

        return None


    def _cache_hist(self):

        self.history['net_worth'].append(self.net_wroth)

        return None
    

@constraint(requires, TrainMarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ConsoleTearsheetRenderWrapper:

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
        logger.info(
            f'n_symbols: {self.market_env.n_symbols}, '
            f'n_features: {self.market_env.n_features}')

        return observation
    
    def step(self, action):

        observation, reward, done, info = self.env.step(action)
        self.index += 1

        if self.index % self.render_every == 0 or done:
            self.render(done)

        return observation, reward, done, info

    def render(self, done):
       
        progress = self.market_metadata_env.progress
        return_ = self.market_metadata_env.return_
        sharpe = self.market_metadata_env.sharpe
        net_worth = self.market_metadata_env.net_worth
        positions = self.market_metadata_env.positions
        cash = self.market_metadata_env.cash
        profit = self.market_metadata_env.profit
        longs = self.market_metadata_env.longs
        shorts = self.market_metadata_env.shorts

        metrics = [f'{progress:.0%}', f'{return_:.2%}', f'{sharpe:.4f}',
                    f'${net_worth:,.2f}', f'${positions:,.2f}', f'${cash:,.2f}',
                    f'${profit:,.2f}', f'${longs:,.2f}', f'${shorts:,.2f}']

        if self.index == self.render_every:
            title = ['Progress', 'Return', 'Sharpe ratio',
                'Assets', 'Positions', 'Cash']
            print(tabular_print(title, header=True))

        # add performance metrics to tear sheet
        print(tabular_print(metrics))



@constraint(before, ActionWrapper)
class MinTradeSizeActionWrapper(ActionWrapper):
    
    # ensures no action wrapper exists after this wrapper
    # to maintain min trade size.
    def __init__(self, env: Env, min_action = 1) -> None:
        super().__init__(env)
        self.min_action = min_action

    def action(self, action):

        new_action = [action_ if abs(action_) >= self.min_action
            else 0 for action_ in action]
    
        return new_action


@constraint(requires, TrainMarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RelativeShortSizingActionWrapper(ActionWrapper):

    def __init__(self, env: Env, short_raio = 0.2) -> None:
        super().__init__(env)
        self.short_ratio = short_raio
        self.short_budget = None
    

    def _set_short_budget(self):

        max_short_size = self.short_ratio * self.market_metadata_env.net_worth
        self.short_budget = max(max_short_size - self.market_metadata_env.shorts, 0)

        return None
    

    def action(self, action):
        # performs actions without applying the effects and modifies actions
        # that would lead to short sizing limit violation.

        positions = self.market_metadata_env.positions

        self._set_short_budget()

        for asset, action_ in enumerate(action):

            if action_ > 0:
                continue
            
            if self.short_budget == 0:
                action[asset] = 0
                continue

            sell = min(abs(action_), self.short_budget)
            short = abs(min(0, positions[asset] - sell))
            self.short_budget -= short
            action[asset] = -sell

        return action


@constraint(requires, TrainMarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RelativeMarginSizingActionWrapper(ActionWrapper):

    def __init__(self, env: Env, initial_margin= 1) -> None:
        super().__init__(env)

        self.initial_margin = initial_margin


    def action(self, action):
        # performs actions without applying the effects and modifies actions
        # that would lead to short sizing limit violation.


        for asset, action_ in enumerate(action):

            if action_ < 0:
                continue

            if self.cash <= 0:
                action[asset] = 0
                continue

            leverage = 1/self.initial_margin
            buy = min(action_, leverage * self.cash)
            action[asset] = buy

        return action
    

@constraint(requires, AbstractMarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ContinuousRelativePositionSizingActionWrapper(ActionWrapper):
    # ensures positions taken at each step is a maximum fixed percentage of net worth
    # maps actions in (-1, 1) to buy/sell/hold following position sizing strategy
    # trade_ratio = 0.02 means max of 2% of net_worth is traded at each step
    # max_trade and min_trade are max/min (USD) for each asset at each step
    # action in (-threshold, threshold) is parsed as hold
    # action outside this range is linearly projected to (min_trade, max_trade)
    def __init__(self, env: Env, trade_ratio = 0.02, hold_threshold = 0.15):

        super().__init__(env)
        self.trade_ratio = trade_ratio
        self.hold_threshold = hold_threshold
        self._max_trade_per_asset = None

        self.action_space = spaces.Box(
            low = -1, high = 1, shape = (self.n_symbols, ))
        

    
    def _set_max_trade_per_asset(self, trade_ratio: float) -> float:
        # sets value for self.max_trade
        # Recommended initial_cash >= n_stocks/trade_ratio. 
        # Trades bellow $1 is clipped to 1 (API constraint).
        self._max_trade_per_asset = (trade_ratio * self.market_metadata_env.net_worth)/self.n_symbols

        return None



    def parse_action(self, action: float) -> float:

        # action value in (-threshold, +threshold) is parsed as hold
        fraction = (abs(action) - self.hold_threshold)/(
            1- self.hold_threshold)

        parsed_action =  fraction * self._max_trade_per_asset * np.sign(action
            ) if fraction > 0 else 0
        
        return parsed_action



    def action(self, action):

        self._set_max_trade_per_asset(self.trade_ratio)

        new_actions = [self.parse_action(
            action) for action in action]
        
        return new_actions


@constraint(unwrapped_is_not, TradeMarketEnv)
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class IntegerAssetQuantityActionWrapper(ActionWrapper):
    # enforces actions to map to integer quantity of share
    # would not be valid in trade market env due to price
    # slippage before execution of orders.
    def __init__(self, env: Env) -> None:

        super().__init__(env)
        self.asset_prices = None

    def action(self, action):

        asset_prices = self.market_metadata_env.asset_prices

        for asset , action_ in action:
            asset_price = asset_prices[asset]
            action_ = (action_ // asset_price) * asset_price
            action[asset] = action_
        
        return action


@constraint(before, ObservationWrapper)
@constraint(requires, AbstractMarketEnvMetadataWrapper, 'market_metadata_env')
class PositionsFeatureEngineeringWrapper(ObservationWrapper):
    # augments observations such that instead of number shares held
    # USD value of assets (positions) is used.

    def __init__(self, env: Env) -> None:

        super().__init__(env)
        self.n_symbols = self.market_metadata_env.n_symbols

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
        
    def observation(self, observation):

        asset_prices= self.market_metadata_env.asset_prices
        asset_quantities = observation.pop('asset_quantities')

        observation['positions'] = asset_prices * asset_quantities
        return observation
    


@constraint(wraps, PositionsFeatureEngineeringWrapper)
@constraint(requires, AbstractMarketEnvMetadataWrapper, 'market_metadata_env')
class WealthAgnosticFeatureEngineeringWrapper(ObservationWrapper):
    # Augment observations so that 
    def __init__(self, env: Env) -> None:

        super().__init__(env)
        self.initial_cash = self.market_metadata_env.initial_cash
        self.n_symbols = self.market_metadata_env.n_symbols

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
    

    def observation(self, observation):

        net_worth = self.market_metadata_env.net_worth

        observation['positions'] /= net_worth
        observation['cash'] /= net_worth
        observation['return'] = net_worth/self.initial_cash - 1

        return observation




@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ObservationBufferWrapper(ObservationWrapper):

    # augments observations by stacking them
    # usefull to encode memory in env observation
    pass



class RunningMeanSTD:
    pass


@constraint(wraps, ObservationBufferWrapper)
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RunningIndicatorsObsWrapper(ObservationWrapper):

    # computes running indicators
    def observation(self, observation):
        return None



class NormalizeObservationsWrapper(ObservationWrapper):
    pass



@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class NormalizeRewardsWrapper(RewardWrapper, BaseCallback):
    # normalizes rewards of an episode
    pass



@constraint(unwrapped_is, AbstractMarketEnv)
class ExperienceRecorderWrapper(Wrapper):
    # saves experiences in buffer writes to hdf5 when 
    # buffer reaches a certain size.
    pass
