from typing import Type

from torch import nn

from neural.data.base import DatasetMetadata
from neural.env.base import AbstractMarketEnv
from neural.meta.pipe import AbstractPipe
from neural.train.base import AbstractTrainer



class Agent:

    def __init__(
            self,
            model: nn.Module,
            pipe: AbstractPipe,
            dataset_metadata: DatasetMetadata,
            ):

        self.model = model
        self.pipe = pipe
        self.dataset_metadata = dataset_metadata

        return None


    def train(self, 
        trainer: Type[AbstractTrainer]
        ) -> None:
    
        trained_model = trainer.train(self.agent)
        self.model = trained_model


    def trade(self, market_env: AbstractMarketEnv):

        """
        Starts the trading process by creating a trading environment and executing
        actions from the model.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        market_env = self.pipe.pipe(market_env)
        observation = market_env.reset()

        while True:
            action = self.model(observation)
            observation, reward, done, info  = market_env.step(action)
            if done:
                market_env.reset()