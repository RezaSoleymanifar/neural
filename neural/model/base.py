"""
This module contains the base class for all models.
"""

import gym
from torch import nn


class AbstractModel:
    """
    This is the base class for all models.
    """

    def __init__(self):
        """
        Initialize the model.
        """
        pass

    def __call__(self, observation):
        """
        Given an observation, return an array of actions.

        args:
        ----------
        observation (numpy.ndarray): 
            The observation from the environment.

        Returns:
        ----------
        numpy.ndarray: 
            An array of actions.
        """
        raise NotImplementedError

    def save(self, file_path):
        """
        Save the model to a file.

        Parameters:
        ----------
        file_path (str): 
            Path to save the model.
        """
        raise NotImplementedError

    def train(self, env: gym.Env, *args, **kwargs):
        """
        Train the model.

        Parameters:
        ----------
        env (gym.Env): 
            The environment to train the model on.
        """
        raise NotImplementedError


class StableBaselinesModel(AbstractModel):

    def __init__(self):
        super().__init__()
        self.base_model = None

    def __call__(self, observation):
        if self.base_model is None:
            raise RuntimeError("Model is not trained yet.")
        return self.base_model(observation)

    def train(self, env, *args, **kwargs):
        if self.base_model is None:
            self.base_model = self.build_model(env)
            self.base_model.learn(*args, **kwargs)

    def build_model(self, env: gym.Env, feature_extractor: nn.Module, policy: nn.Module):
        raise NotImplementedError
