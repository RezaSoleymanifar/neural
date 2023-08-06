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
    >>> model = StableBaselinesModel(...)
    >>> agent = Agent(model, pipe)
"""
from dataclasses import dataclass
import dill
from typing import Optional
import os

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

    def save(self, dir: str | os.PathLike):
        """
        A function to save an agent to a tarfile. The agent is saved in
        a directory with the following structure:
        dir
        ├── dataset_metadata
        ├── pipe
        └── model
            └── base_model.zip
            └── model

        Args:
        -------
            file_path (str | os.PathLike):
                The path to the tarfile to save the agent to.
        """
        os.makedirs(dir, exist_ok=True)

        with open(os.path.join(dir, 'pipe'), 'wb') as pipe_file:
            dill.dump(self.pipe, pipe_file)

        with open(os.path.join(dir, 'dataset_metadata'),
                'wb') as dataset_metadata_file:
            dill.dump(self.dataset_metadata, dataset_metadata_file)

        model_dir = os.path.join(dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(model_dir)


    def load_agent(
        dir: str | os.PathLike,
    ) -> Tuple[nn.Module, AbstractPipe, DatasetMetadata]:
        """
        Loads an agent from a tarfile. The agent is saved as a tarfile with
        the following structure:
        agent.tar
        ├── dataset_metadata
        ├── pipe
        └── model.tar
            └── base_model.zip
            └── model

        Args:
        -------
            file_path (str | os.PathLike):
                The path to the tarfile to load the agent from.
        Returns:
        --------
            model (nn.Module):
                The model of the agent.
            pipe (AbstractPipe):
                The pipe of the agent.
            dataset_metadata (DatasetMetadata):
                The metadata of the dataset used to train the agent.
        """
        with open(os.path.join(dir, 'pipe'), 'rb') as pipe_file:
            pipe = dill.load(pipe_file)

        with open(os.path.join(dir, 'dataset_metadata'), 'rb') as dataset_metadata_file:
            dataset_metadata = dill.load(dataset_metadata_file)

        model_dir = os.path.join(dir, 'model')
        StableBaselinesModel.load(model_dir)

        return Agent(model=model, pipe=pipe, dataset_metadata=dataset_metadata)
