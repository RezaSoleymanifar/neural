"""
This module contains the base class for all models.
"""

import gym


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
    def train(self, env: Env):
        return super().train(env)