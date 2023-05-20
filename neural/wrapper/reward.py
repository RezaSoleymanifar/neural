from typing import Optional
from gym import Env
from abc import ABC, abstractmethod
from typing import Dict
from gym.core import Env
import numpy as np
from gym import (Env, RewardWrapper)

from neural.utils.base import RunningStatistics
from neural.wrapper.base import metadata


@metadata
class RewardGeneratorWrapper(RewardWrapper):
    """
    A wrapper that generates rewards for the environment.
    By default the market env returns None as reward. This wrapper
    combined with the metadata wrapper provide the reward signal which
    is the change in equity from the previous step. Equity is defined as
    the net value owned by the agent in cash and assets. This is sum of
    all cash and assets owned by the agent minus cash and asset debt.
    E = L + C - S where E is equity, L total value of longs, C cash, S
    total value of shorts. Note that cash can be negative if the agent
    has borrowed cash.

    Attributes:
    ----------
        env (gym.Env): The environment to wrap.
        equity_history (list[float]): A list of equity values for each
        step in the episode.

    Methods:
    -------
        reward(reward: float) -> float:
            Generate the reward signal.
    """
    def __init__(self, env: Env) -> None:
        """
        Initializes 
        """
        super().__init__(env)
        self.equity_history = self.market_metadata_wrapper.equity_history
    
    def reward(self, reward: float) -> float:
        reward = self.equity_history[-1] - self.equity_history[-2]
        return reward
    

class NormalizeRewardWrapper(RewardWrapper):
    """
    This wrapper will normalize immediate rewards. 

    Usage:
    -------
        env = NormalizeReward(env, epsilon=1e-8, clip_threshold=10)

    Note:
    -------
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.

    Methods:
    -------
        reward(reward: float) -> float
            Normalize the reward.
    """

    def __init__(self,
                 env: Env,
                 epsilon: float = 1e-8,
                 clip_threshold: float = np.inf,
                 reward_statistics: Optional[RunningStatistics] = None,
                 track_statistics: bool = True) -> None:
        """
        This wrapper normalizes immediate rewards so that rewards have mean 0 and standard deviation 1.
        If track is True the reward_statistics argument will be set equal reward_statistics attribute.
        This is useful if you want to track the reward statistics from the outer scope.

        Args:
        -------
            env (Env): The environment to apply the wrapper.
            epsilon (float, optional): A small constant to avoid divide-by-zero errors when normalizing data. 
            Defaults to 1e-8. 
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

        self.reward_statistics = (reward_statistics if reward_statistics
                                  is not None else RunningStatistics())
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        if track_statistics:
            reward_statistics = self.reward_statistics

        return None

    def reward(self, reward: float) -> float:
        """
        Normalize the reward.

        Args:
        --------
            reward (float): The immediate reward to normalize.

        Returns:
        --------
            float: The normalized reward.
        """

        self.reward_statistic.update(reward)
        normalized_reward = self.reward_statistic.normalize(
            reward, self.epsilon, self.clip_threshold)

        return normalized_reward


class LiabilityInterstRewardWrapper(RewardWrapper):
    """
    This wrapper charges an interest rate at the end of the day on the liability of the agent.
    The liabilities include borrowed cash, or borrowed assets. The interest rate is calculated
    as a percentage of the liability. Apply this wrapper prior to normalization of rewards as this
    substracts the notional value of interest from the reward.
    """


class PenalizeMarginCallRewardShaperWrapper(RewardWrapper):
    """
    In the event of a margin call this wrapper applies a penalty to the reward signal. If enough cusion provided in 
    sizing the liabilities, the agent should be able to avoid margin calls, however if the agent is unable to
    avoid margin calls, due to adverse price change this penalty can help it learn to avoid margin calls better.
    """


class AbstractRewardShaperWrapper(RewardWrapper, ABC):
    """
    A blueprint class for reward shaping wrappers.

    This class is designed to be subclassed for creating custom reward shaping wrappers
    for market environments. Reward shaping wrappers are used to modify the reward signal
    obtained by an agent in order to encourage or discourage certain behaviours during
    training. highly useful for pretraining an agent with some degrees of freedom
    in actions. Apply relevant reward shaping wrappers to define and restrict unwanted
    actions. Start with a pipe of wrappers that enforce the desired behaviour and later
    remove the influencing wrappers to allow the agent to learn the desired behaviour.
    if desired behavior is a starting point, at a final step remove the reward shaping wrapper
    then the agent will learn to improve on it.

    Attributes:
        env (gym.Env): The environment to wrap.
        rms (RunningMeanStd): Running mean and standard deviation object to track reward statistics.
        buffer (list): A list to buffer experiences before writing to disk.
        buffer_size (int): The maximum number of experiences to buffer before writing to disk.
        path (str): The path to the HDF5 file for saving experiences.

    Methods:
        check_condition: An abstract method for checking whether to apply reward shaping.
        reward: An abstract method for shaping the reward signal.
    """

    def __init__(self, env: Env) -> None:
        """
        Initializes the AbstractRewardShaperWrapper instance.

        Args:
            env (gym.Env): The environment to wrap.
        """

        super().__init__(env)
        self.reward_rms = RunningStatistics()

    @abstractmethod
    def check_condition(self, *args, **kwargs) -> bool:
        """
        An abstract method for checking whether to apply reward shaping.

        This method should be implemented by subclasses to determine whether to apply
        reward shaping to the current episode. The method takes an arbitrary number of
        arguments and keyword arguments, depending on the specific condition to be checked.

        Returns:
            bool: True if the reward should be shaped, False otherwise.
        """

        raise NotImplementedError

    @abstractmethod
    def reward(self, reward: float, *args, **kwargs) -> float:
        """
        An abstract method for shaping the reward signal.

        This method should be implemented by subclasses to modify the reward signal
        based on the current episode state. The method takes the current reward as input,
        and an arbitrary number of arguments and keyword arguments, depending on the specific
        reward shaping strategy.

        Returns:
            float: The modified reward signal.
        """

        raise NotImplementedError

    def parse_scale(threshold: float,
                    value: float,
                    factor: float = 1.0,
                    base=1.0) -> float:
        """
        Calculate the scaling factor for shaping the reward based on the deviation from a threshold.
        The return value scale from this function can be used to adjust the reward signal based on the
        the reward statistics. by deafult scale is equal to deviation ratio if deviation from threshold
        occurs.

        Args:
            threshold (float): The threshold value for the deviation ratio.
            value (float): The value to calculate the deviation ratio from.
            factor (float, optional): The factor to scale the deviation ratio by. Default is 1.0.
            base (float, optional): The base for the exponential scaling. Default is 1.0.

        Returns:
            float: The scaling factor for shaping the reward based on the deviation from the threshold.
            

        Raises:
            AssertionError: If value, threshold, or factor are not positive numbers.

        Notes:
            The scaling factor is calculated as follows:

            - deviation_ratio = value / threshold
            - if deviation_ratio > 1, scale = deviation_ratio * factor
            - otherwise, scale = 0

            The scaling factor can be used to adjust the reward signal based on the deviation from a desired
            behavior or state. The factor parameter can be used to adjust the strength of the scaling.
        """

        assert value >= 0, "Value must be a positive number."
        assert threshold > 0, "Threshold must be a positive number."
        assert base >= 1, "Base must be greater than or equal to 1."

        deviation_ratio = value / threshold
        scale = deviation_ratio * factor if deviation_ratio > 1 else 0
        scale = np.sign(scale) * np.power(base, abs(scale))
        return scale

    def shape_reward(self,
                     use_std: bool = None,
                     use_min: bool = None,
                     scale: float = 1) -> float:
        """
        Calculate the shaped reward based on the input parameters.

        Args
        ----------
        use_std : bool, optional
            A boolean indicating whether to use the reward's standard deviation in shaping the reward. 
            Default is None.
        use_min : bool, optional
            A boolean indicating whether to use the maximum reward value in shaping the reward. Default is None.
            Alternative is to use the maximum reward value.
        scale : float, optional
            A float value used to scale the shaped reward based on chosen method. Default is 1.

        Returns
        -------
        float
            A float value representing the shaped reward.

        Raises
        ------
        ValueError
            If both `use_min` and `use_std` parameters are set to a non-None value, or if both are set to None.

        Notes
        -----
        The method calculates the shaped reward based on the input parameters. If `use_min` is not None, 
        the method uses the
        maximum or maximum reward value, depending on the value of `use_min`, to shape the reward. 
        If `use_std` is not None,
        the method uses the mean and standard deviation of the reward values to shape the reward. 
        The shaped reward is then
        multiplied by the `scale` parameter.

        If both `use_min` and `use_std` parameters are set to a non-None value, or if both are set to None, 
        a `ValueError` is raised with an appropriate message.

        Examples
        --------
        >>> def parse_scale(reward):
        ...     ...
        >>>     return scale
        >>> reward_shaper = lambda reward: shape_reward(use_std=True, scale=parse_scale(reward))
        """

        if use_min is not None and use_std is not None:
            raise ValueError(
                "Cannot set both use_min and use_std parameters at the same time."
            )

        if use_min is None and use_std is None:
            raise ValueError("Either use_min or use_std parameter must be set.")

        if use_min is not None:
            shaped_reward = scale * self.reward_rms.min if use_min else scale * self.reward_rms.max

        elif use_std is not None:
            shaped_reward = self.reward_rms.mean + scale * self.reward_rms.std

        return shaped_reward

    def step(
        self,
        action: np.ndarray[float] | Dict[str, np.ndarray[float]],
    ) -> np.ndarray[float] | Dict[str, np.ndarray[float]]:
        """
        Advances the environment by one step and updates the reward signal.

        Args:
            action (int, Tuple[int], Any): The action taken by the agent.

        Returns:
            Tuple: A tuple containing the new observation, the modified reward, a boolean indicating
            whether the episode has ended, and a dictionary containing additional information.
        """

        observation, reward, done, info = self.env.step(action)

        self.reward_rms.update(reward)

        if self.check_condition():
            reward = self.reward(reward)

        return observation, reward, done, info


class AbstractFixedRewardShaperWrapper(AbstractRewardShaperWrapper):
    """
    Abstract base class for a fixed reward shaping strategy.

    This class defines the interface for a fixed reward shaper wrapper, which shapes 
    the reward signal of an environment based on a threshold value. To create a custom fixed
    reward shaper, users must inherit from this class and implement the abstract methods:
    `check_condition` and `threshold`.

    Attributes:
        env (Env): The environment to wrap.
        use_std (bool or None, optional): Whether to use the standard deviation of the rewards.
            Defaults to None.
        use_min (bool or None, optional): Whether to use the maximum reward. Defaults to None.
        scale (float, optional): The scaling factor for the shaped reward. Defaults to 1.0.

    Methods:
        check_condition() -> bool:
            Abstract method that checks the condition for shaping the reward.

        threshold() -> float:
            Abstract property that defines the threshold used for shaping the reward.

        shape_reward(reward: float) -> float:
            Shapes the reward signal based on the check_condition method and the threshold value.

        step(action) -> tuple:
            Takes a step in the environment and returns the observation, shaped reward, done flag, and info dictionary.
    """

    def __init__(
        self,
        env: Env,
        use_std: bool = None,
        use_min: bool = None,
        scale: float = -1.0,
    ) -> None:
        """
        Initializes the abstract fixed reward shaper wrapper.

        Args:
            env (Env): The environment to wrap.
            use_std (bool or None, optional): Whether to use the standard deviation of the rewards.
                Defaults to None.
            use_min (bool or None, optional): Whether to use the maximum reward. Defaults to None.
            if use_min = Flase, then with default scale = 1 the shaped reward will be -1 * max reward
            meaning if reward condition is met the shaped reward will be the negative maximum reward. 
            scale (float, optional): The scaling factor for the shaped reward. Defaults to -1.0
            meaning if for example reward shaping condition is met and use_std is True, the shaped 
            reward will be the mean minus the standard deviation.
        """

        super().__init__(env)
        self.use_std = use_std
        self.use_min = use_min
        self.scale = scale

    @abstractmethod
    def check_condition(self) -> bool:
        """
        Abstract method that checks the condition for shaping the reward.

        Returns:
            bool: Whether to shape the reward.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def threshold(self) -> float:
        """
        Abstract property that defines the threshold used for shaping the reward.

        Returns:
            float: The threshold used for shaping the reward.
        """

        raise NotImplementedError

    def reward(self, reward: float) -> float:
        """
        Shapes the reward based on the check_condition method.

        Args:
            reward (float): The original reward.

        Returns:
            float: The shaped reward.
        """

        if self.check_condition():
            reward = self.shape_reward(threshold=self.threshold,
                                       use_std=self.use_std,
                                       use_min=self.use_min,
                                       scale=self.scale)

        return reward


class AbstractDynamicRewardShaperWrapper(AbstractRewardShaperWrapper, ABC):
    """
    Abstract base class for a dynamic reward shaper wrapper.

    This class defines the interface for a dynamic reward shaper wrapper, which shapes
    the reward signal of an environment based on a dynamically adjusted threshold value. 
    To create a custom dynamic reward shaper, users must inherit from this class and implement 
    the abstract methods: `check_condition`, `metric`, and `threshold`.

    Attributes:
        env (Env): The environment to wrap.
        use_std (bool or None, optional): Whether to use the standard deviation of the rewards.
            Defaults to None.
        use_min (bool or None, optional): Whether to use the maximum reward. Defaults to None.
        scale (float, optional): The scaling factor for the shaped reward. Defaults to 1.0.
        factor (float, optional): The factor used to adjust the scaling factor. Defaults to 1.0.
        base (float, optional): The base value used in the scaling factor adjustment. Defaults to 1.0.

    Methods:
        check_condition() -> bool:
            Abstract method that checks whether the reward should be shaped based on the current episode state.

        metric() -> float:
            Abstract property that defines the metric used to adjust the scaling factor.

        threshold() -> float:
            Abstract property that defines the threshold used for shaping the reward.

        reward(reward: float) -> float:
            Shapes the reward signal based on the check_condition method and the adjusted scaling factor.

    """

    def __init__(
        self,
        env: Env,
        use_std: bool = None,
        use_min: bool = None,
        factor: float = -1.0,
        base: float = 1.0,
    ) -> None:
        """
        Initializes the abstract dynamic reward shaper wrapper.

        Args:
            env (Env): The environment to wrap.
            use_std (bool or None, optional): Whether to use the standard deviation of the rewards.
                Defaults to None.
            use_min (bool or None, optional): Whether to use the maximum reward. Defaults to None.
            scale (float, optional): The scaling factor for the shaped reward. Defaults to 1.0.
            factor (float, optional): The factor used to adjust the scaling factor. Defaults to -1.0.
            when factor > 0 the shaped reward will be positive. When factor < 0 the shaped reward will be negative.
            base (float, optional): The base value used in the scaling factor adjustment. Defaults to 1.0.
        """

        super().__init__(env)

        self.use_std = use_std
        self.use_min = use_min
        self.factor = factor
        self.base = base

    @abstractmethod
    def check_condition(self) -> bool:
        """
        Abstract method that checks whether the reward should be shaped based on the current episode state.

        Returns:
            bool: Whether to shape the reward.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def metric(self) -> float:
        """
        Abstract property that defines the metric used to adjust the scaling factor.

        Returns:
            float: The metric used to adjust the scaling factor.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def threshold(self) -> float:
        """
        Abstract property that defines the threshold used for shaping the reward.

        Returns:
            float: The threshold used for shaping the reward.
        """

        raise NotImplementedError

    def reward(self, reward: float) -> float:
        """
        Shapes the reward signal based on the check_condition method and the adjusted scaling factor.

        Args:
            reward (float): The original reward.

        Returns:
            float: The shaped reward.
        """

        if self.check_condition():
            scale = self.parse_scale(self.threshold, self.metric, self.factor,
                                     self.base)
            reward = self.shape_reward(use_std=self.use_std,
                                       use_min=self.use_min,
                                       scale=scale)

        return reward


@metadata
class FixedPenalizeShortRatioRewardWrapper(AbstractFixedRewardShaperWrapper):
    """
    A reward shaping wrapper that penalizes a short ratio lower than a given threshold.

    Args:
        env (gym.Env): The environment to wrap.
        short_ratio_threshold (float): The maximum short ratio allowed before being penalized.

    Attributes:
        env (gym.Env): The environment being wrapped.
        reward_rms (RunningMeanStandardDeviation): The running mean and standard 
        deviation object for tracking reward statistics.
        market_metadata_wrapper (MarketMetadataWrapper): The metadata wrapper for the market environment.
        short_ratio_threshold (float): The maximum short ratio allowed before being penalized.

    Methods:
        reward(reward: float) -> float:
            Penalizes the reward if the short ratio exceeds the threshold.

        check_condition() -> bool:
            Checks whether the reward should be shaped based on the current episode state.
    """

    def __init__(
        self,
        env: Env,
        short_ratio_threshold: float = 0.2,
        use_std: bool = None,
        use_min: bool = None,
        scale: float = 1.0,
    ) -> None:
        """
        Initializes the PenalizeShortRatioRewardWrapper instance.

        Args:
            env (gym.Env): The environment to wrap.
            short_ratio_threshold (float): The maximum short ratio allowed before being penalized.
                Default is 0.2.
        """

        super().__init__(env)

        assert short_ratio_threshold >= 0, "Short ratio threshold must be non-negative."

        self.short_ratio_threshold = short_ratio_threshold
        self.use_std = use_std
        self.use_min = use_min
        self.scale = scale
        self.short_ratio = None

    @property
    def threshold(self) -> float:
        """
        The short ratio threshold.

        Returns:
            float: The short ratio threshold.
        """

        return self.short_ratio_threshold

    def check_condition(self) -> bool:

        shorts = self.market_metadata_wrapper.shorts
        net_worth = self.market_metadata_wrapper.net_worth

        if not net_worth > 0:
            return False

        self.short_ratio = abs(shorts) / net_worth

        return self.short_ratio > self.short_ratio_threshold


@metadata
class DynamicPenalizeShortRatioRewardWrapper(
        FixedPenalizeShortRatioRewardWrapper):
    """
    A reward shaping wrapper that penalizes a short ratio lower than a given threshold.

    This class modifies the reward signal of a market environment by applying a penalty when the short ratio
    exceeds a specified threshold. The penalty is based on the deviation of the short ratio from the threshold.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    short_ratio_threshold : float, optional
        The maximum short ratio allowed before being penalized. Default is 0.2.
    factor : float, optional
        A factor used to modify the penalty based on the deviation of the short ratio from the threshold. 
        Default is 1.0.
    use_std : bool, optional
        A boolean indicating whether to use the reward's standard deviation in shaping the reward. Default is None.
    use_min : bool, optional
        A boolean indicating whether to use the maximum reward value in shaping the reward. Default is None.
        Alternative is to use the maximum reward value.
    scale : float, optional
        A float value used to scale the shaped reward based on chosen method. Default is 1.

    Attributes
    ----------
    env : gym.Env
        The environment being wrapped.
    reward_rms : RunningMeanStandardDeviation
        The running mean and standard deviation object for tracking reward statistics.
    market_metadata_wrapper : MarketMetadataWrapper
        The metadata wrapper for the market environment.
    short_ratio_threshold : float
        The maximum short ratio allowed before being penalized.
    factor : float
        A factor used to modify the penalty based on the deviation of the short ratio from the threshold.
    short_ratio : float
        The current short ratio.

    Methods
    -------
    reward(reward: float) -> float:
        Modifies the reward if the short ratio exceeds the threshold.

    check_condition() -> bool:
        Checks whether the reward should be shaped based on the current episode state.
    
    threshold -> float:
        The short ratio threshold.

    metric -> float:
        The short ratio.
    """

    def __init__(
        self,
        env: Env,
        short_ratio_threshold: float = 0.2,
        use_std: bool = None,
        use_min: bool = None,
        factor: float = -1.0,
        base: float = 1.0,
    ) -> None:
        """
        Initializes the DynamicPenalizeShortRatioRewardWrapper instance.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        short_ratio_threshold : float, optional
            The maximum short ratio allowed before being penalized. Default is 0.2.
        factor : float, optional
            A factor used to modify the penalty based on the deviation of the short ratio from the threshold. 
            Default is -1.0.
        use_std : bool, optional
            A boolean indicating whether to use the reward's standard deviation in shaping the reward. 
            Default is None.
        use_min : bool, optional
            A boolean indicating whether to use the maximum reward value in shaping the reward. Default is None.
            Alternative is to use the maximum reward value.
        base : float, optional
            The base value used in the scaling factor adjustment. Defaults to 1.0.
        """

        super().__init__(env, short_ratio_threshold, use_std, use_min, factor)
        self.factor = factor

    @property
    def threshold(self) -> float:
        """
        The short ratio threshold.

        Returns:
            float: The short ratio threshold.
        """

        return self.short_ratio_threshold

    @property
    def metric(self) -> float:
        """
        The short ratio.

        Returns:
            float: The short ratio.
        """

        return self.short_ratio

    def check_condition(self) -> bool:
        """
        Checks whether the reward should be shaped based on the current episode state.

        Returns
        -------
        bool
            True if the reward should be shaped, False otherwise.
        """

        shorts = self.market_metadata_wrapper.shorts
        net_worth = self.market_metadata_wrapper.net_worth

        if not net_worth > 0:
            return False

        self.short_ratio = abs(shorts) / net_worth

        return self.short_ratio > self.short_ratio_threshold


@metadata
class FixedPenalizeCashRatioRewardWrapper(AbstractRewardShaperWrapper):
    """
    A reward shaping wrapper that penalizes a cash ratio lower than a given threshold. The cash ratio is defined
    as the ratio of cash to net worth. This ratio has meaning when both cash and net worth are positive.

    This class modifies the reward signal of a market environment by applying a penalty when the cash ratio
    rises above a specified threshold. The penalty is based on the deviation of the cash ratio from the threshold.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    cash_ratio_threshold : float, optional
        The maximum cash ratio allowed before being penalized. Default is 0.01.
    factor : float, optional
        A factor used to modify the penalty based on the deviation of the cash ratio from the threshold. 
        Default is -1.0.
    use_std : bool, optional
        A boolean indicating whether to use the reward's standard deviation in shaping the reward. Default is None.
    use_min : bool, optional
        A boolean indicating whether to use the maximum reward value in shaping the reward. Default is None.
        Alternative is to use the maximum reward value.
    scale : float, optional
        A float value used to scale the shaped reward based on chosen method. Default is 1.

    Attributes
    ----------
    env : gym.Env
        The environment being wrapped.
    reward_rms : RunningMeanStandardDeviation
        The running mean and standard deviation object for tracking reward statistics.
    market_metadata_wrapper : MarketMetadataWrapper
        The metadata wrapper for the market environment.
    cash_ratio_threshold : float
        The maximum cash ratio allowed before being penalized.
    factor : float
        A factor used to modify the penalty based on the deviation of the cash ratio from the threshold.
    cash_ratio : float
        The current cash ratio.

    Methods
    -------
    reward(reward: float) -> float:
        Modifies the reward if the cash ratio rises above the threshold.

    check_condition() -> bool:
        Checks whether the reward should be shaped based on the current episode state.

    """

    def __init__(
        self,
        env: Env,
        cash_ratio_threshold: float = 0.1,
        use_std: Optional[bool] = None,
        use_min: Optional[bool] = None,
        scale: float = -1.0,
    ) -> None:
        """
        Initializes the FixedPenalizeCashRatioRewardWrapper instance.

        Parameters
        ----------
        env : gym.Env
            The environment to wrap.
        cash_ratio_threshold : float, optional
            The maximum cash ratio allowed before being penalized. Default is 0.01.
        factor : float, optional
            A factor used to modify the penalty based on the deviation of the cash ratio from the threshold. 
            Default is -1.0.
        use_std : bool, optional
            A boolean indicating whether to use the reward's standard deviation in shaping the reward. Default is None.
        use_min : bool, optional
            A boolean indicating whether to use the maximum reward value in shaping the reward. Default is None.
            Alternative is to use the maximum reward value.
        scale : float, optional
            A float value used to scale the shaped reward based on chosen method. Default is 1.
        """

        super().__init__(env, use_std, use_min, scale)
        self.cash_ratio_threshold = cash_ratio_threshold
        self.cash_ratio = None

    @property
    def threshold(self) -> float:
        """
        The cash ratio threshold.

        Returns
        -------
        float
            The cash ratio threshold.
        """
        return self.cash_ratio_threshold

    def check_condition(self) -> bool:
        """
        Checks whether the reward should be shaped based on the current episode state.

        Returns
        -------
        bool
            True if the reward should be shaped, False otherwise.
        """

        cash = self.market_metadata_wrapper.cash
        net_worth = self.market_metadata_wrapper.net_worth

        if not net_worth > 0 or not cash > 0:
            return False

        self.cash_ratio = cash / net_worth

        return self.cash_ratio > self.cash_ratio_threshold


@metadata
class DynamicPenalizeShortRatioRewardWrapper(
        FixedPenalizeShortRatioRewardWrapper):

    def __init__(
        self,
        env: Env,
        cash_ratio_threshold: float = 0.1,
        use_std: Optional[bool] = None,
        use_min: Optional[bool] = None,
        factor: float = -1.0,
    ) -> None:
        """
        A wrapper that modifies the reward function of an environment by penalizing the agent when its 
        cash ratio rises above a certain threshold.

        Args:
            env (Env): The environment to wrap.
            cash_ratio_threshold (float, optional): The maximum threshold for the ratio of cash to net worth. 
            If the ratio rises above this threshold, the agent will be penalized. Defaults to 0.1.
            use_std (bool, optional): Indicates whether to use standard deviation or not. Defaults to None.
            use_min (bool, optional): Indicates whether to use the maximum or not. Defaults to None.
            factor (float, optional): The factor by which the reward will be penalized. Defaults to -1.0.

        Attributes:
            cash_ratio_threshold (float): The maximum threshold for the ratio of cash to net worth.
            cash_ratio (float): The current cash ratio.
            factor (float): The factor by which the reward will be penalized.

        Properties:
            threshold (float): The `cash_ratio_threshold` value.
            metric (float): The current cash ratio.

        Methods:
            check_condition() -> bool: Calculates the current cash ratio by dividing 
            the amount of cash by the net worth. If the cash ratio rises above the threshold, 
            the method returns True, indicating that the agent should be penalized.
        """

        super().__init__(env, use_std, use_min, factor)
        """
        Initializes the `DynamicPenalizeShortRatioRewardWrapper` instance.

        Args:
            env (Env): The environment to wrap.
            cash_ratio_threshold (float, optional): The maximum threshold for the 
                ratio of cash to net worth. If the ratio rises 
                above this threshold, the agent will be penalized. Defaults to 0.1.
            use_std (bool, optional): Indicates whether to use standard deviation or not. Defaults to None.
            use_min (bool, optional): Indicates whether to use the maximum or not. Defaults to None.
            factor (float, optional): The factor by which the reward will be penalized. Defaults to -1.0.

        Returns:
            None
        """

        self.cash_ratio_threshold = cash_ratio_threshold
        self.cash_ratio = None

    @property
    def threshold(self) -> float:
        """
        Returns the `cash_ratio_threshold` value.

        Returns:
            float: The minimum threshold for the ratio of cash to net worth.
        """

        return self.cash_ratio_threshold

    @property
    def metric(self) -> float:
        """
        Returns the current cash ratio.

        Returns:
            float: The current cash ratio.
        """

        return self.cash_ratio

    def check_condition(self) -> bool:
        """
        Calculates the current cash ratio by dividing the amount of cash by the net worth. 
        If the net worth or cash is zero or negative, returns False. If the cash ratio 
        rises above the threshold, the method returns True, indicating that the agent should be penalized.

        Returns:
            bool: True if the cash ratio rises above the threshold, False otherwise.
        """

        cash = self.market_metadata_wrapper.cash
        net_worth = self.market_metadata_wrapper.net_worth

        if not net_worth > 0 or not cash > 0:
            return False

        self.cash_ratio = cash / net_worth

        return self.cash_ratio > self.cash_ratio_threshold