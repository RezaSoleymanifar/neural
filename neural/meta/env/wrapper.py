from stable_baselines3.common.callbacks import BaseCallback
from gym import ActionWrapper, Env, spaces, Wrapper, ObservationWrapper, RewardWrapper
import numpy as np
from neural.common.log import logger
from neural.tools.ops import sharpe, tabular_print
from neural.meta.env.base import AbstractMarketEnv
from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Type, Callable, Optional
from neural.common.exceptions import IncompatibleWrapperError



class AbstractConstrainedWrapper(ABC):

    @abstractmethod
    def _check_constraint(self):

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

                if constraint_attr is not None and isinstance(constraint_attr, str):
                    setattr(self, constraint_attr, self._check_constraint(env))

                else:
                    self._check_constraint(env)

                super().__init__(env, *args, **kwargs)

            def _check_constraint(self, env):
                return constraint_function(env, constrained_type)

        return ConstrainedWrapper

    return constraint_decorator



def unwrapped_is(wrapped_env, constrained_type):

    unwrapped_env = wrapped_env.unwrapped

    if not isinstance(unwrapped_env, constrained_type):

        raise IncompatibleWrapperError(
            f'{wrapped_env} requires {unwrapped_env} to be of type {constrained_type}.'
            )

    return unwrapped_env



def first_of(wrapped_env, constrained_type):
    
    if hasattr(wrapped_env, 'env'):
        if isinstance(wrapped_env.env, constrained_type):

            raise IncompatibleWrapperError(
                f'{wrapped_env.env} of type {constrained_type} is applied before enclosing constrained wrapper.'
            )
        else:
            first_of(wrapped_env.env)
    


def requires(wrapped_env, constrained_type):

    def requires_helper(wrapped_env, initial_arg):

        if not hasattr(wrapped_env, 'env'):

            raise IncompatibleWrapperError(
                f'{initial_arg} requires wrapper of type {constrained_type} to exist in the underlying wrappers.')

        elif isinstance(wrapped_env.env, constrained_type):
            return wrapped_env.env

        else:
            return requires_helper(wrapped_env.env, initial_arg)

    return requires_helper(wrapped_env, wrapped_env)


def last_of()

    def last_of_helper(outer_most_wrapper, initial_arg):

        if not isinstance(outer_most_wrapper, ConstraintCheckerWrapper):
            return None
        
        else:

            if isinstance(outer_most_wrapper, constrained_type):

                raise IncompatibleWrapperError(
                    f'{outer_most_wrapper} wrapper of type {constrained_type} is applied after FILL LATER')
            else:
                return last_of_helper(outer)


    

class ConstraintCheckerWrapper(Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def _check_constraint(self, env):
        if hasattr(env, 'env'):
            if hasattr(env.env, '')
    
        
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class MarketEnvMetadataWrapper:
    # wraps a market env to track env metadata
    # downstream wrappers can utilize this metadata
    def __init__(self) -> None:
        super().__init__()

    def _update_summary(self):

        asset_quantities = self.market_env.asset_quantities
        asset_prices = self.market_env.asset_prices
        net_worth = self.market_env.net_worth
        initial_cash = self.market_env.initial_cash

        short_mask = asset_quantities < 0
        long_mask = asset_quantities > 0

        # total value of positions in portfolio
        self.positions = asset_quantities @ asset_prices

        self.shorts = asset_quantities[short_mask] @ asset_prices[short_mask]
        self.longs = asset_quantities[long_mask] @ asset_prices[long_mask]
        # sharpe ratio filters volatility to reflect investor skill
        
        self.profit = net_worth - initial_cash
        self.return_ = (net_worth - initial_cash)/initial_cash

        sharpe_ = sharpe(self.history['assets'])
        progress_ = self.base_env.index/self.base_env.n_steps



class EnvWarmupWrapperActionWrapper(ActionWrapper):
    # runs env with random actions.
    # warms up running parameters of observation scaling wrappers.
    pass



@constraint(first_of, ActionWrapper)
class MinTradeSizeActionWrapper(ActionWrapper):
    
    def __init__(self, env: Env) -> None:
        super().__init__(env)
    
    def action(actions):

        new_actions = [action if abs(action) >= 1 
            else 0 for action in actions]
    
        return new_actions


@constraint(requires, MarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RelativeShortSizingActionWrapper(ActionWrapper):

    def __init__(self, env: Env, short_raio = 0.1) -> None:
        super().__init__(env)
        self.short_ratio = short_raio

    
    def _set_max_short_size(self):

        self.short_size = self.short_ratio * self.market_env.net_worth

        return None
    
    def action(self, actions):

        self._set_max_short_size()

        # iterates over actions
        for asset, action in enumerate(actions):

            if action > 0 and self.cash > 0:  # buy

                    buy = min(self.cash, action)
                    quantity = buy/self.asset_prices[asset]

                    self.asset_quantities[asset] += quantity
                    self.cash -= buy

            elif action < 0 and self.asset_quantities[asset] > 0:  # sell

                sell = min(
                    self.asset_quantities[asset] * self.asset_prices[asset], abs(action))
                quantity = sell/self.asset_prices[asset]

                self.asset_quantities[asset] -= quantity
                self.cash += sell
                self.holds[asset] = 0

        return actions


@constraint(requires, MarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ContinuousRelativeMarginSizingActionWrapper(ActionWrapper):
    pass


@constraint(requires, MarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RelativePositionSizingActionWrapper(ActionWrapper):
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
        max_trade = (trade_ratio * self.net_worth)/self.n_symbols

        if max_trade < self.min_trade:
            raise ValueError(
                f'max_trade: {max_trade} < min_trade: {self.min_trade}.'
            )

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


@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class RelativeActionScalerActionWrapper(ActionWrapper):
    # linearly scales magnitute of actions to make it 
    # proportional to a starting cash differnet than training.
    pass


@constraint(requires, MarketEnvMetadataWrapper, 'market_metadata_env')
@constraint(unwrapped_is, AbstractMarketEnv, 'market_env')
class ConsoleTearsheetRenderWrapper:
    
    def __init__(
        self, env: Env,
        verbosity: int = 20
        ) -> None:

        super().__init__(env)
        self.verbosity = verbosity
        self.history = defaultdict(list)
        self.render_every = self.base_env.n_steps//self.verbosity


    def _cache_market_env_hist(self):

        self.history['assets'].append(self.base_env.net_worth)


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
