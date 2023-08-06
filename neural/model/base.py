"""
This module contains the base class for all models.
"""
from copy import copy
import dill
import os

import gym
from torch import nn
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG


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
    """
    This is the base class for all models that use stable-baselines.
    Options for the algorithm are:
        - 'ppo':
            Proximal Policy Optimization (PPO)
        - 'a2c':
            Advantage Actor Critic (A2C)
        - 'dqn':
            Deep Q-Network (DQN)
        - 'sac':
            Soft Actor Critic (SAC)
        - 'td3':    
            Twin Delayed Deep Deterministic Policy Gradient (TD3)
        - 'ddpg':
            Deep Deterministic Policy Gradient (DDPG)
    """
    ALGORITHMS = {
        'ppo': PPO,
        'a2c': A2C,
        'dqn': DQN,
        'sac': SAC,
        'td3': TD3,
        'ddpg': DDPG,
    }

    def __init__(
        self, 
        algorithm: str,
        feature_extractor: nn.Module,
        policy: nn.Module):

        super().__init__()
        self.algorithm = self._get_algorithm(algorithm)
        self.feature_extractor = feature_extractor
        self.policy = policy
        self.base_model = None

    def __call__(self, observation):
        if self.base_model is None:
            raise RuntimeError("Model is not trained yet.")
        return self.base_model(observation)

    def _get_algorithm(self, algorithm_name: str) -> OnPolicyAlgorithm:
        algorithm_class = self.ALGORITHMS.get(algorithm_name.lower())
        if algorithm_class is None:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}. "
                             f"Supported options: {self.ALGORITHMS.keys()}")
        return algorithm_class

    def train(self, env, *args, **kwargs):
        if self.base_model is None:
            self.base_model = self._build_model(env)
        else:
            self.base_model.env = env

        self.base_model.learn(*args, **kwargs)
        return None

    def _build_model(self, env: gym.Env):
        model = self.algorithm(policy=self.policy, env=env)
        return model

    def save(self, dir: str | os.PathLike):
        os.makedirs(dir, exist_ok=True)
        self.base_model.save(os.path.join(dir, 'base_model'))
        with open(os.path.join(dir, 'model'),
            'wb') as model_file:
            model_copy = copy(self)
            del model_copy.base_model
            dill.dump(model_copy, model_file)
        return None
    
    def load(self, dir: str | os.PathLike):
        """
        Load the model from a directory.

        Parameters:
        ----------
        dir (str): 
            The directory to load the model from.
        """
        with open(os.path.join(dir, 'model'), 'rb') as model_file:
            model = dill.load(model_file)
            model.base_model = model.algorithm.load(os.path.join(dir, 'base_model'))

        for attr_name, attr_value in vars(model).items():
            if hasattr(self, attr_name):
                setattr(self, attr_name, attr_value)