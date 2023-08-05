"""
aget.py

Description:
------------
This module defines the Agent class.

License:
--------
    MIT License. See LICENSE.md file.

Author(s):
-------
    Reza Soleymanifar, Email: Reza@Soleymanifar.com
    
Classes:
--------
    Agent:
        A reinforcement learning agent. This class is a self-contained
        entity that can be used with any other training or trading
        object. It bundles together a Pytorch neural network model, a
        Pipe object that transforms the outputs of market environment to
        the inputs of the model through a stack of wrappers, and
        metadata about the dataset used to train the agent. This later
        is used to map the training dataset to a matching trading stream
        to reconstruct the data that agent was trained on. Trainer
        object interacts with this model to update the parameters
        of the model (training) and Trader object interacts with this
        model to make trading decisions (inference).

Examples:
---------
    >>> from neural.meta.agent import Agent
    >>> from neural.meta.pipe import MarginAccountPipe
    >>> from neural.meta.model import ActorCriticModel
    >>> pipe = MarginAccountPipe(...)
    >>> model = ActorCriticModel(...)
    >>> agent = Agent(model, pipe)
"""
from dataclasses import dataclass
from typing import Optional

from neural.data.base import DatasetMetadata
from neural.meta.pipe import AbstractPipe
from neural.model.base import AbstractModel


@dataclass
class Agent:
    """
    A reinforcement learning agent. This is a self-contained entity that
    can be used with any other training or trading object. It bundles
    together a neural network model, a pipe object that
    transforms the outputs of market envvironment to the inputs of the
    model, and metadata about the dataset used to train the agent. This
    later is used to map the training data to a trading stream that
    matches the dataset used to train the agent. Trainer and Trader
    objects interact with this model to update the parameters of the
    model and to make trading decisions, respectively.

    Args:
    ----------
        model (AbstractModel): 
            The PyTorch neural network model used by the agent.
        pipe (AbstractPipe): 
            The data pipe used to transform input data.
        dataset_metadata (Optional[DatasetMetadata]): 
            Metadata about the dataset used by the agent.
    """
    model: AbstractModel
    pipe: AbstractPipe
    dataset_metadata: Optional[DatasetMetadata] = None