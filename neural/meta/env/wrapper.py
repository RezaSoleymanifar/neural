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
from neural.core.trade.ops import AlpacaTrader

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



def first_of(wrapper_at_constructor, constrained_type):

    def first_of_helper(wrapper, wrapper_at_constructor):

        if hasattr(wrapper, 'env'):
            if isinstance(wrapper.env, constrained_type):

                raise IncompatibleWrapperError(
                    f'{wrapper.env} of type {constrained_type} '
                    f'is applied before enclosing constrained wrapper {wrapper_at_constructor}.'
                )
            else:
                return first_of_helper(wrapper_at_constructor.env, wrapper_at_constructor)

    return first_of_helper(wrapper_at_constructor, wrapper_at_constructor)




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


        
@constraint(unwrapped_is, None, 'market_env')
class TrainMarketEnvMetadataWrapper(Wrapper):
    # wraps a market env to track env metadata
    # upstream wrappers can utilize this metadata
    def __init__(self) -> None:
        super().__init__()

        self.history = defaultdict(list)
        self.initial_cash = self.market_env.initial_cash
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

        self.history['assets'].append(self.net_wroth)

        return None
    

    def reset(self):

        observation = self.env.reset()
        self._update_metadata()

        return observation
    

    def step(self, actions):

        observation, reward, done, info = self.env.step(actions)
        self._update_metadata()

        return observation, reward, done, info


@constraint(requires, TradeMarketEnv, 'market_env')
class AlpacaTradeEnvMetadataWrapper:

    def __init__(self) -> None:
        super().__init__()

        self.trader = self.market_env.trader

        if not isinstance(self.trader, AlpacaTrader):

            raise IncompatibleWrapperError(
                f'Market env {self.market_env} must have trader of type {AlpacaTrader}.')
        
        self.history = defaultdict(list)
        self.initial_cash = self.market_env.initial_cash
        self.cash = None
        self.asset_quantities = None
        self.positions = None
        self.net_worth = None
        self.portfolio_value = None
        self.longs = None
        self.shorts = None
        self.profit = None
        self.return_ = None
        self.sharpe = None
        self.progress = None

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


    def reset(self):

        observation = self.env.reset()
        self._update_metadata()
        
        return observation
    
    def step(self, actions):

        observation, reward, done, info = self.env.step(actions)
        self._update_metadata()

        return observation, reward, done, info



@constraint(first_of, ActionWrapper)
class MinTradeSizeActionWrapper(ActionWrapper):
    
    def __init__(self, env: Env, min_action = 1) -> None:
        super().__init__(env)
        self.min_action = min_action

    def action(self, actions):

        new_actions = [action if abs(action) >= self.min_action
            else 0 for action in actions]
    
        return new_actions


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
    

    def action(self, actions):
        # performs actions without applying the effects and modifies actions
        # that would lead to short sizing limit violation.

        positions = self.market_metadata_env.positions

        self._set_short_budget()

        for asset, action in enumerate(actions):

            if action > 0:
                continue
            
            if self.short_budget == 0:
                actions[asset] = 0
                continue

            sell = min(abs(action), self.short_budget)
            short = abs(min(0, positions[asset] - sell))
            self.short_budget -= short
            actions[asset] = -sell

        return actions


@constraint(requires, TrainMarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RelativeMarginSizingActionWrapper(ActionWrapper):

    def __init__(self, env: Env, initial_margin=1.00) -> None:
        super().__init__(env)

        self.initial_margin = initial_margin


    def action(self, actions):
        # performs actions without applying the effects and modifies actions
        # that would lead to short sizing limit violation.


        for asset, action in enumerate(actions):

            if action < 0:
                continue

            if self.cash <= 0:
                actions[asset] = 0
                continue

            leverage = 1/self.initial_margin
            buy = min(action, leverage * self.cash)
            actions[asset] = buy

        return actions
    

@constraint(requires, TrainMarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ContinuousRelativePositionSizingActionWrapper(ActionWrapper):
    # ensures positions taken at each step is a maximum fixed percentage of net worth
    # maps actions in (-1, 1) to buy/sell/hold following position sizing strategy
    # trade_ratio = 0.02 means max of 2% of net_worth is traded at each step
    # max_trade and min_trade are max/min (USD) for each asset at each step
    # action in (-threshold, threshold) is parsed as hold
    # action outside this range is linearly projected to (min_trade, max_trade)
    def __init__(self, env: Env, trade_ratio = 0.02, threshold = 0.15):

        super().__init__(env)
        self.trade_ratio = trade_ratio
        self.threshold = threshold
        self.max_trade = None

        self.action_space = spaces.Box(
            low = -1, high = 1, shape = (self.n_symbols, ))
        

    
    def _set_max_trade(self, trade_ratio: float) -> float:
        
        # sets value for self.max_trade
        # Recommended initial_cash >= n_stocks/trade_ratio. 
        # Trades bellow $1 is clipped to 1 (API constraint).
        max_trade = (trade_ratio * self.market_metadata_env.net_worth)/self.n_symbols

        return max_trade



    def parse_action(self, action: float) -> float:

        # action value in (-threshold, +threshold) is parsed as hold
        fraction = (abs(action) - self.threshold)/(
            1- self.threshold)

        parsed_action =  fraction * self.max_trade * np.sign(action
            ) if fraction > 0 else 0
        
        return parsed_action


    def action(self, actions):

        self.max_trade = self._set_max_trade(self.trade_ratio)

        new_actions = [
            self.parse_action(action) for action in actions]
        
        return new_actions


@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class IntegerAssetQuantityActionWrapper(ActionWrapper):
    #enforces actions to map to integer quantity of share
    pass



@constraint(requires, TrainMarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ConsoleTearsheetRenderWrapper:
    
    def __init__(
        self, env: Env,
        verbosity: int = 20
        ) -> None:

        super().__init__(env)
        self.verbosity = verbosity
        self.render_every = self.base_env.n_steps//self.verbosity


    def reset(self):

        state = self.env.reset()
        self._cache_market_env_hist()

        logger.info(
            f'Steps: {self.base_env.n_steps}, '
            f'symbols: {self.base_env.n_symbols}, '
            f'features: {self.base_env.n_features}'
        )

        return state
    

    def step(self, actions):

        state, reward, done, info = self.env.step(actions)
        self._cache_market_env_hist()

        if (self.base_env.index != 0 and
            self.base_env.index % self.render_every == 0
            ) or done:

            self.render()

        return state, reward, done, info
    

    def render(self, done: bool = False) -> None:

        # print header at first render
        if self.base_env.index == self.render_every:

            # print results in a tear sheet format
            print(tabular_print(
                ['Progress', 'Return', 'Sharpe ratio',
                'Net worth', 'Positions', 'Cash', 'Profit',
                'Longs', 'Shorts'], header=True))

        asset_quantities = self.base_env.asset_quantities
        asset_prices = self.base_env.asset_prices
        net_worth = self.base_env.net_worth
        initial_cash = self.base_env.initial_cash

        short_mask = asset_quantities < 0
        long_mask = asset_quantities > 0

        # total value of positions in portfolio
        positions = asset_quantities @ asset_prices

        shorts = asset_quantities[short_mask] @ asset_prices[short_mask]
        longs = asset_quantities[long_mask] @ asset_prices[long_mask]
        # sharpe ratio filters volatility to reflect investor skill
        
        profit = net_worth - initial_cash
        return_ = (net_worth - initial_cash)/initial_cash

        sharpe_ = sharpe(self.history['assets'])
        progress_ = self.base_env.index/self.base_env.n_steps

        metrics = [f'{progress_:.0%}', f'{return_:.2%}', f'{sharpe_:.4f}',
            f'${self.base_env.net_worth:,.2f}', f'${positions:,.2f}', f'${self.base_env.cash:,.2f}',
            f'${profit:,.2f}', f'${longs:,.2f}', f'${shorts:,.2f}']
        
        # add performance metrics to tear sheet
        print(tabular_print(metrics))

        if done:
            logger.info('Episode terminated.')
            logger.info(*metrics)
        return None


@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class PositionValuesObservationWrapper(ObservationWrapper):
    # augments observations such that instead of number shares held
    # USD value of assets (positions) is used.
    pass

@constraint(requires, PositionValuesObservationWrapper)
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class NetWorthAgnosticObsWrapper(ObservationWrapper):
    # scales state with respect to assets to make agent initial assets value.
    pass



@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RunningIndicatorsObsWrapper(ObservationWrapper):
    # computes running indicators
    pass


@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ObservationStackerObsWrapper(ObservationWrapper):
    # augments observations by stacking them
    # usefull to encode memory in env observation
    pass


@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class NormalizeRewardsWrapper(RewardWrapper, BaseCallback):
    # normalizes rewards of an episode
    pass


@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class DiscountRewardsWrapper(RewardWrapper, BaseCallback):
    # discoutns rewards of an episode
    pass

@constraint(unwrapped_is, AbstractMarketEnv)
class ExperienceRecorderWrapper(Wrapper):
    pass
