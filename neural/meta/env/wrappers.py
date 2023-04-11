from stable_baselines3.common.callbacks import BaseCallback
from gym import ActionWrapper, Env, spaces, Wrapper, ObservationWrapper, RewardWrapper
import numpy as np
from neural.common.log import logger
from neural.tools.ops import sharpe, tabular_print
from neural.meta.env.base import AbstractMarketEnv
from collections import defaultdict
from abc import abstractmethod, ABC



class MarketEnvWrapper(Wrapper):

    # adds funcitonality to type check and return underlyig market env
    def __init__(self, env: Env) -> None:
        self.base_env = env.unwrapped
        super().__init__(env)
        

        if not isinstance(
            self.base_env, AbstractMarketEnv):

            raise ValueError(
                f'Base env {self.base_env} is not instance of {AbstractMarketEnv}.'
                )


def market_wrapper(wrapper_class):

    class MarketEnvWrapper(wrapper_class):
        pass

    return MarketEnvWrapper



class AbstractConflictingWrapper(ABC):

    @abstractmethod
    def _check_conflict(self):

        raise NotImplementedError


class TerminalActionWrapper(Wrapper):
    # actions after this wrapper cannot be altered
    # used to enforce hard market constraints

    def __init__(self, env: Env) -> None:

        self._check_conflict(env)
        super().__init__(env)


    def _check_conflict(self, env: Env):

        if isinstance(env, ActionWrapper):

            raise ValueError(
            f'Wrapping {ActionWrapper} with {TerminalActionWrapper} not allowed. Environment {env} is a {ActionWrapper}.'
            )
        
        # Recursively check for conflicting action wrappers in the enclosed environments
        return self._check_conflict(env.env) if hasattr(env, 'env') else None


def terminal_action(class_):

    class TerminalActionWrapper(class_):
        pass

    return TerminalActionWrapper



@market_wrapper
class MarketEnvMetadataWrapper:
    # wraps a market env to track env metadata
    # downstream wrappers can utilize this metadata
    def __init__(self) -> None:
        super().__init__()


    def _update_summary():
        pass


class MetadataDependentWrapper:

    def __init__(self, env):

        self.market_metadata_env = self._check_conflict(env)
        super().__init__(env)

    def _check_conflict(self, env):

        if isinstance(env, MarketEnvMetadataWrapper):
            return env
        
        elif hasattr(env, 'env'):
            return self._check_conflict(env)
        
        else:
            raise TypeError(
                f'{MetadataDependentWrapper} cannot wrap underlying {env} with no {MarketEnvMetadataWrapper} around it.'
            )


def metadata_dependent(wrapper_class):

    class MetadataDependentWrapper(wrapper_class):
        pass

    return MetadataDependentWrapper



@terminal_action
class MinTradeSizeActionWrapper(ActionWrapper):
    
    def action(actions):

        new_actions = [action if abs(action) >= 1 
            else 0 for action in actions]
    
        return new_actions

@metadata_dependent
@market_wrapper
class RelativeShortSizingActionWrapper(ActionWrapper):

    def __init__(self, env: Env, short_raio = 0.1) -> None:
        super().__init__(env)
        self.short_ratio = short_raio

    
    def _set_max_short_size(self):
        self.short_size = self.short_ratio * self.base_env.net_worth
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


@metadata_dependent
@market_wrapper
class RelativeMarginSizingActionWrapper(ActionWrapper):
    pass


class EnvWarmupWrapperActionWrapper(ActionWrapper):
    # runs env with random actions.
    # warms up running parameters of observation scaling wrappers.
    pass


@metadata_dependent
@market_wrapper
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


class IntegerAssetQuantityActionWrapper(ActionWrapper):
    #enforces actions to map to integer quantity of share
    pass


@metadata_dependent
@market_wrapper
class RelativeActionScalerActionWrapper(ActionWrapper):
    # linearly scales magnitute of actions to make it 
    # proportional to a starting cash differnet than training.
    pass


@metadata_dependent
@market_wrapper
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
    

@market_wrapper
class NetWorthAgnosticObsWrapper(ObservationWrapper):
    # scales state with respect to assets to make agent initial assets value.
    pass


@market_wrapper
class RunningIndicatorsObsWrapper(ObservationWrapper):
    # computes running indicators
    pass

@market_wrapper
class ObservationStackerObsWrapper(ObservationWrapper):
    # augments observations by stacking them
    # usefull to encode memory in env observation
    pass

@market_wrapper
class NormalizeRewardsWrapper(RewardWrapper, BaseCallback):
    # normalizes rewards of an episode
    pass

market_wrapper
class DiscountRewardsWrapper(RewardWrapper, BaseCallback):
    # discoutns rewards of an episode
    pass
