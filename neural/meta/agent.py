from typing import Type

from torch import nn

from neural.data.base import DatasetMetadata
from neural.meta.pipe import AbstractPipe
from neural.train.base import AbstractTrainer
from neural.trade.base import AbstractTrader


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


    def train(self, trainer: AbstractTrainer) -> None:
    
        trained_model = trainer.train(self.agent)
        self.model = trained_model


    def test(self, trainer: AbstractTrainer) -> None:

        trainer.test(self.agent)
                     

    def trade(self, trader: AbstractTrader) -> None:

        trader.trade(self.agent)
