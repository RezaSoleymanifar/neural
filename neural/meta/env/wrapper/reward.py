from abc import ABC

import numpy as np
from gym import (Env, Wrapper, RewardWrapper)

from neural.tools.misc import RunningMeanStandardDeviation



class NormalizeReward(RewardWrapper):

    """
    This wrapper will normalize immediate rewards. 
    Usage:
        env = NormalizeReward(env, epsilon=1e-8, clip_threshold=10)
    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    Methods:
        reward(reward: float) -> float
            Normalize the reward.
    """

    def __init__(
        self,
        env: Env,
        epsilon: float = 1e-8,
        clip_threshold: float = np.inf,
    ) -> None:
        """
        This wrapper normalizes immediate rewards so that rewards have mean 0 and standard deviation 1.

        Args:
            env (Env): The environment to apply the wrapper.
            epsilon (float, optional): A small constant to avoid divide-by-zero errors when normalizing data. Defaults to 1e-8.
            clip_threshold (float, optional): A value to clip normalized data to, to prevent outliers 
            from dominating the statistics. Defaults to np.inf.

    Example
    -------
    >>> from neural.meta.env.base import TrainMarketEnv
    >>> from neural.meta.env.wrapper.reward import NormalizeReward
    >>> env = TrainMarketEnv(...)
    >>> env = NormalizeReward(env)
    """

        super().__init__(env)

        self.reward_rms = RunningMeanStandardDeviation()
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

    def reward(self, reward: float) -> float:
        
        """Normalize the reward.

        Args:
            reward (float): The immediate reward to normalize.

        Returns:
            float: The normalized reward.
        """

        self.reward_rms.update(reward)
        normalized_reward = self.reward_rms.normalize(
            reward, self.epsilon, self.clip_threshold)

        return normalized_reward


class AbstractRewardShaperWrapper(Wrapper, ABC):

    # highly useful for pretraining an agent with some degrees of freedom
    # in actions. Apply relevant reward shaping wrappers to define and restrict unwanted
    # actions. Start with a pipe of wrappers that enforce the desired behaviour and later
    # remove the influencing wrappers to allow the agent to learn the desired behaviour.
    # if desired behavior is a starting point, at a final step remove the reward shaping wrapper
    # then the agent will learn to improve on it.

    pass


class PenalizeShortRatioRewardWrapper(RewardWrapper):
    pass


class PenalizeCashRatioRewardWrapper(RewardWrapper):
    pass
