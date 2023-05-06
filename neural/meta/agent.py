from dataclasses import dataclass
from torch import nn

from neural.data.base import DatasetMetadata
from neural.meta.pipe import AbstractPipe


@dataclass
class Agent:

    """
    A reinforcement learning agent. This is a self-contained entity that can be used with
    any other training or trading entity. It contains a neural network model, a pipe object that
    transforms the outputs of market envvironment to the inputs of the model, and metadata
    about the dataset used to train the agent. This later is used to map the training data
    to a trading stream that matches the dataset used to train the agent. Trainer and Trader
    objects will use this model to update the parameters of the model and to make trading
    decisions, respectively.

    Attributes:
        model (nn.Module): The neural network model used by the agent.
        pipe (AbstractPipe): The data pipe used to transform input data.
        dataset_metadata (DatasetMetadata): Metadata about the dataset used by the agent.
    """

    model: nn.Module
    pipe: AbstractPipe
    dataset_metadata: DatasetMetadata