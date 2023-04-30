import os

from abc import ABC, abstractmethod
from torch import nn

from neural.env.base import TrainMarketEnv
from neural.meta.agent import Agent
from neural.data.base import StaticDataFeeder
from neural.utils.io import from_hdf5



class AbstractTrainer(ABC):

    def __init__(
        self, 
        agent: Agent,
        file_path: os.PathLike,
        dataset_name,
        n_chunks: int = 1,
        ) -> None:
    
        self.agent = agent
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.n_chunks = n_chunks

        self.train_market_env = None
        self.static_data_feeder = None


    def _get_train_market_env(self) -> TrainMarketEnv:
        
        dataset_metadata, datasets  = from_hdf5(self.file_path, self.dataset_name)
        self.static_data_feeder =  StaticDataFeeder(
            dataset_metadata = dataset_metadata, datasets = datasets, n_chunks=self.n_chunks)
        train_market_env = TrainMarketEnv(trainer = self)
        piped_market_env = self.agent.pipe.pipe(train_market_env)

        return piped_market_env

    @abstractmethod
    def train(self, *args, **kwargs) -> nn.Module:

        raise NotImplementedError
    
    @abstractmethod
    def test(self, *args, **kwargs) -> None:

        raise NotImplementedError