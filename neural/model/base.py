"""
This module contains the base class for all models.
"""
from copy import copy
import os

import dill

import gym
import torch
from torch import nn
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG

from neural.env.base import TrainMarketEnv

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

    def load(self, file_path):
        """
        This 
        """

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

    MODEL_SAVE_FILE_NAME = 'model'
    BASE_MODEL_SAVE_FILE_NAME = 'stable_baselines3_model'

    def __init__(self, algorithm: str, policy: str | nn.Module = 'MlpPolicy'):

        super().__init__()
        self.algorithm = self._get_algorithm(algorithm)
        self.policy = policy
        self.base_model = None

    def __call__(self, observation):
        if self.base_model is None:
            raise RuntimeError("Model is not trained yet.")
        with torch.no_grad(), torch.set_grad_enabled(False):
            return self.base_model(observation)

    def _get_algorithm(self, algorithm_name: str) -> OnPolicyAlgorithm:
        algorithm_class = self.ALGORITHMS.get(algorithm_name.lower())
        if algorithm_class is None:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}. "
                             f"Supported options: {self.ALGORITHMS.keys()}")
        return algorithm_class

    def save(self, dir: str | os.PathLike):
        os.makedirs(dir, exist_ok=True)
        self.base_model.save(os.path.join(dir, self.BASE_MODEL_SAVE_FILE_NAME))
        with open(os.path.join(dir, self.MODEL_SAVE_FILE_NAME), 'wb') as model_file:
            model_copy = copy(self)
            del model_copy.base_model
            dill.dump(model_copy, model_file)
        return None

    @classmethod
    def load(cls, dir: str | os.PathLike):
        """
        Load the model from a directory. File structure should be:
        dir
        └── model
        └── base_model.zip

        Args:
        ----------
            dir (str):
                The directory to load the model from.
        """
        with open(os.path.join(dir, cls.MODEL_SAVE_FILE_NAME), 'rb') as model_file:
            model = dill.load(model_file)
            model.base_model = model.algorithm.load(
                os.path.join(dir, cls.BASE_MODEL_SAVE_FILE_NAME))

        return model

    def _build_base_model(self, env: TrainMarketEnv):
        model = self.algorithm(policy=self.policy, env=env)
        return model

    def _set_base_model_env(self, env: TrainMarketEnv) -> None:
        self.base_model.save("temp_model")
        self.base_model = self.base_model.load("temp_model", env)
        os.remove("temp_model")
        return None

    def train(self, env, *args, **kwargs):
        if self.base_model is None:
            self.base_model = self._build_base_model(env)
        else:
            self._set_base_model_env(env)

        self.base_model.learn(*args, **kwargs)
        return None