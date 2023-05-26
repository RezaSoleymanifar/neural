"""
base.py
"""
import os
import inspect
import copy

from abc import ABC, abstractmethod
from torch import nn
from typing import Optional, Tuple
import numpy as np

from neural.env.base import TrainMarketEnv
from neural.meta.agent import Agent
from neural.data.base import StaticDataFeeder
from neural.utils.io import from_hdf5

from gym.vector import AsyncVectorEnv, SyncVectorEnv


class AbstractTrainer(ABC):
    """
    This is an abstract class for training agents. It is designed to
    proivde common functionalities for training agents. The features
    provided by this class are:
        - Train/test split
        - Training on multiple environments
        - Random initializaiton of environments
        - Splitting environments into exclusive temporal groups
    
    Args:
        agent (Agent): 
            Agent to be trained.
        file_path (os.PathLike): 
            Path to the HDF5 file.
        dataset_name (str):
            Name of the dataset in the HDF5 file. If None, all datasets
            are joined together.
        n_chunks (int):
            Number of chunks to split the dataset into, per environment
            for loading data. Used for memory management. If n_chunks =
            1 then entire dataset is loaded into memory.
        train_ratio (float):
            Ratio of the dataset to be used for training. Must be in (0,
            1].
        n_envs (int):
            Number of environments to train on. If more than one then
            multiple environments are used for training. Ensure n_envs
            does not exceed CPU core count.
        async_envs (bool):
            If True, environments are run asynchronously, i.e. multiple
            environments are run in parallel on CPU cores. If False,
            environments are run synchronously, i.e. one at a time.
        exclusive_envs (bool):
            If True, environments are split into exclusive temporal
            groups, i.e. if time horizon is from 0 to 100, and n_envs =
            5 then for each interval [0, 20), [20, 40), [40, 60), [60,
            80), [80, 100) a new environment is created. If False, then
            n_envs copies of the same environment are created, with
            entire time horizon.
    
    Attributes:
    ----------
        agent (Agent):
            Agent to be trained.
        file_path (os.PathLike):
            Path to the HDF5 file.
        dataset_name (str):
            Name of the dataset in the HDF5 file. If None, all datasets
            are joined together.
        n_chunks (int):
            Number of chunks to split the dataset into, per environment
            for loading data. Used for memory management. If n_chunks =
            1 then entire dataset is loaded into memory.    
        train_ratio (float):
            Ratio of the dataset to be used for training. Must be in (0,
            1].
        n_envs (int):
            Number of environments to train on. If more than one then
            multiple environments are used for training. Ensure n_envs
            does not exceed CPU core count.
        async_envs (bool):
            If True, environments are run asynchronously, i.e. multiple
            environments are run in parallel on CPU cores. If False,
            environments are run synchronously, i.e. one at a time.
        exclusive_envs (bool):
            If True, environments are split into exclusive temporal
            groups, i.e. if time horizon is from 0 to 100, and n_envs = 
            5 then for each interval [0, 20), [20, 40), [40, 60), [60,
            80), [80, 100) a new environment is created. If False, then
            n_envs copies of the same environment are created, with
            entire time horizon.
        train_market_env (TrainMarketEnv):
            Training environment.
        test_market_env (TrainMarketEnv):
            Testing environment.
        train_data_feeder (StaticDataFeeder):
            Data feeder for training environment.
        test_data_feeder (StaticDataFeeder):
            Data feeder for testing environment.
        env_pipes (list):
            List of pipes for saved for parallel training. Can be reused
            to continue training in parallel.

    Methods:
    -------
        _initialize_data_feeders():
            Initializes data feeders for training and testing
            environments.
        _get_piped_envs():
            Returns a list of piped environments for parallel training.
            if n_envs = 1 then a single environment is returned. If 
            n_envs > 1 then a single parallel environment is returned.
            Parallel environments are like single environments, except
            that they return a list of observations, actions, rewards,
            info pairs, and take a list of actions as input.
            If called from 'train' method, then the environments are
            created using train_data_feeder. If called from 'test'
            method, then the environments are created using
            test_data_feeder.
        train():
            Uses an RL trainer to train the agent. Implementation is 
            left to the child class.
        test():
            Uses an RL trainer to test the agent. Implementation is

    """
    def __init__(
        self,
        agent: Agent,
        file_path: os.PathLike,
        dataset_name: Optional[str] = None,
        n_chunks: int = 1,
        train_ratio: float = 1,
        n_envs: int = 1,
        async_envs: bool = True,
        exclusive_envs: True = False,
        initial_cash_range: Optional[Tuple[float, float]] = None,
        initial_assets_range: Optional[Tuple[float, float]] = None,
    ) -> None:

        self.agent = agent
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.n_chunks = n_chunks
        self.train_ratio = train_ratio
        self.n_envs = n_envs
        self.async_envs = async_envs
        self.exclusive_envs = exclusive_envs
        self.initial_cash_range = initial_cash_range
        self.initial_assets_range = initial_assets_range

        self.train_market_env = None
        self.test_market_env = None
        self.train_data_feeder = None
        self.test_data_feeder = None

        self.env_pipes = None

        if not 0 < train_ratio <= 1:
            raise ValueError("train_ratio must be in (0, 1]")

        if not n_envs >= 1:
            raise ValueError("n_envs must be >= 1")

        self._initialize_data_feeders()

        return None

    def _initialize_data_feeders(self) -> None:

        dataset_metadata, datasets = from_hdf5(self.file_path,
                                               self.dataset_name)
        data_feeders = StaticDataFeeder(
            dataset_metadata=dataset_metadata,
            datasets=datasets,
            n_chunks=self.n_chunks).split(n=self.train_ratio)

        if len(data_feeders) == 2:

            self.train_data_feeder, self.test_data_feeder = data_feeders

        else:
            self.train_data_feeder = data_feeders.pop()
            self.test_data_feeder = None

        return None

    def _get_piped_env(self) -> TrainMarketEnv:
        """
        Deep copies of agent pipe is create when n_envs > 1. This is to
        avoid complications arised during parallel training and possibly
        modifying the same pipe object at the same time. Pipes created
        in parallel training will be saved for future reference so that
        when performing more paralell training/testing state of the
        parallel pipes are preserved.
        
        The common practice is to train on multiple environments and
        perform a final test on a single environement.
        """

        caller_name = inspect.stack()[1].function

        if caller_name == 'train':
            data_feeder = self.train_data_feeder

        elif caller_name == 'test':
            data_feeder = self.test_data_feeder

        else:
            raise ValueError("Caller must be either train or test")

        n_assets = self.agent.dataset_metadata.n_assets
        initial_cash = lambda: np.random.uniform(
            *self.initial_cash_range
        ) if self.initial_cash_range is not None else lambda: None
        initial_assets = lambda: np.random.uniform(
            *self.initial_assets_range, size=len(n_assets, )
        ) if self.initial_assets_range is not None else lambda: None

        if self.n_envs == 1:
            train_market_env = TrainMarketEnv(data_feeder=data_feeder,
                                        trainer=self,
                                        initial_cash=initial_cash(),
                                        initial_assets=initial_assets())
            piped_market_env = self.agent.pipe.pipe(train_market_env)

            return piped_market_env

        if self.exclusive_envs:
            data_feeders = data_feeder.split(n=self.n_envs)
        else:
            data_feeders = [data_feeder] * self.n_envs

        envs = [
            TrainMarketEnv(data_feeder=data_feeder,
                           initial_cash=initial_cash(),
                           initial_assets=initial_assets())
            for data_feeder in data_feeders
        ]
        self.env_pipes = [
            copy.deepcopy(self.agent.pipe) for _ in range(self.n_envs)
        ] if self.env_pipes is None else self.env_pipes
        env_callables = [
            lambda pipe=pipe, env=env: pipe.pipe(env)
            for pipe, env in zip(self.env_pipes, envs)
        ]

        if self.async_envs:
            piped_market_env = AsyncVectorEnv(env_callables)
        else:
            piped_market_env = SyncVectorEnv(env_callables)

        return piped_market_env

    @abstractmethod
    def train(self, *args, **kwargs) -> nn.Module:

        raise NotImplementedError

    @abstractmethod
    def test(self, *args, **kwargs) -> None:

        raise NotImplementedError
