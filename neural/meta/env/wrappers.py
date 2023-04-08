from stable_baselines3.common.callbacks import BaseCallback
from gym import ActionWrapper, Env, spaces
import numpy as np


class BHSFractionalActionWrapper(ActionWrapper):

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


class ActionMagnitudeScaler(ActionWrapper):
    # scales magnitute of actions of env

    pass

class EnvWarmupWrapper(ActionWrapper):
    # runs env with random actions.
    # warms up running parameters of observation scaling wrappers.
    pass


class RunningIndicatorsWrapper(Env):
    # computes running indicators
    pass

class ObservationStackerWrapper(Env):
    # stacks successive observations of env to augment env state
    # usefull to encode memory in env state
    pass

class NormalizeRewardsWrapper(BaseCallback):
    # normalizes rewards of an episode
    pass

class DiscountRewardsWrapper(BaseCallback):
    # discoutns rewards of an episode
    pass








# class CustomRewardWrapper(BaseCallback):

#     def __init__(self, env, discount_factor=0.99):
#         super(CustomRewardWrapper, self).__init__()
#         self.env = env
#         self.discount_factor = discount_factor
#         self.rewards_buffer = []

#     def _on_step(self) -> bool:
#         reward = self.locals["rewards"][0]
#         self.rewards_buffer.append(reward)
#         return True

#     def _on_episode_end(self) -> bool:
#         rewards = np.array(self.rewards_buffer)
#         self.rewards_buffer = []
#         discounted_rewards = []
#         cum_reward = 0
#         for reward in rewards[::-1]:
#             cum_reward = reward + self.discount_factor * cum_reward
#             discounted_rewards.insert(0, cum_reward)
#         mean = np.mean(discounted_rewards)
#         std = np.std(discounted_rewards)
#         normalized_rewards = (discounted_rewards - mean) / (std + 1e-9)
#         for i in range(len(normalized_rewards)):
#             self.env.memory[i]["reward"] = normalized_rewards[i]
#         return True
