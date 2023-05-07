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

    def __init__(
        self, 
        agent: Agent,
        file_path: os.PathLike,
        dataset_name,
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

        dataset_metadata, datasets  = from_hdf5(self.file_path, self.dataset_name)
        data_feeders =  StaticDataFeeder(
            dataset_metadata = dataset_metadata, datasets = datasets, n_chunks=self.n_chunks).split(n = self.train_ratio)
        
        if len(data_feeders) == 2:

            self.train_data_feeder, self.test_data_feeder = data_feeders

        else:
            self.train_data_feeder = data_feeders.pop()
            self.test_data_feeder = None
    

    def _get_piped_env(self, data_feeder: StaticDataFeeder) -> TrainMarketEnv:

        """
        Deep copies of agent pipe is create when n_envs > 1. This is to avoid to complications arised when doing parallel training
        using and possibly modifying the same pipe object. Pipes created in parallel training will be saved for future reference
        so that when performing more parallell training and/or testing state of the parallel pipes are preserved.
        
        The common practice is to train on multiple environments and perform a final test on a single environement. 
        If tracking is enabled (which is by default), this will also prime the pipe to track and save the statistics of the final test environement. 
        If pipe is wealth insensitive i.e. range of features across different envs with intial_cash and initial_assets envs is the same, 
        then this is not necessary. In this case it is possible to set the sate of agent pipe euqal to state of any of the parallel pipes, as
        an alternative to performing a final test on a single environment, to save the trained state of the pipe.

        It is entirely possible
        to test on multiple environments, however in order to  
        """

        caller_name = inspect.stack()[1].function

        if caller_name == 'train':
            data_feeder = self.train_data_feeder

        elif caller_name == 'test':
            data_feeder = self.test_data_feeder

        else:
            raise ValueError("Caller must be either train or test")
        
        
        initial_cash = lambda: np.random.uniform(
            *self.initial_cash_range) if self.initial_cash_range is not None else lambda: None
        initial_assets = lambda: np.random.uniform(
            *self.initial_assets_range, size=len(n_symbols,)) if self.initial_assets_range is not None else lambda: None

        if self.n_envs == 1:

            market_env = TrainMarketEnv(
                data_feeder = data_feeder, trainer = self, initial_cash=initial_cash(), initial_assets=initial_assets())
            piped_market_env = self.agent.pipe.pipe(market_env)

            return piped_market_env

        if self.exclusive_envs:
            data_feeders = data_feeder.split(n = self.n_envs)
        else:
            data_feeders = [data_feeder] * self.n_envs

        n_symbols = len(self.agent.dataset_metadata.assets)

        envs = [TrainMarketEnv(
            data_feeder=data_feeder, initial_cash=initial_cash(), initial_assets=initial_assets()) for data_feeder in data_feeders]
        self.env_pipes = [copy.deepcopy(self.agent.pipe) for _ in range(self.n_envs)] if self.env_pipes is None else self.env_pipes
        env_callables = [lambda: pipe.pipe(env) for pipe, env in zip(self.env_pipes, envs)]


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