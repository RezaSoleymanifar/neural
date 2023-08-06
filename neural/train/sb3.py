"""
sb3.py

"""
import os
from typing import Optional, Tuple, Union

from torch import nn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from neural.meta.agent import Agent
from neural.env.base import TrainMarketEnv
from neural.train.base import AbstractTrainer


class StableBaselinesTrainer(AbstractTrainer):
    """
    A trainer for Stable Baselines 3 algorithms. Provides a unified
    interface for training and testing Stable Baselines 3 algorithms.

    Args:
    ----
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
        *args:
            Additional arguments.
        **kwargs:
            Additional keyword arguments.

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
        initial_cash_range (Optional[Tuple[float, float]]):
            Range of initial cash values. If None, then initial cash is
            not randomized.
        initial_asset_quantities_range (Optional[Tuple[float, float]]):
            Range of initial asset quantities. If None, then initial
            asset quantities are not randomized. entire time horizon.
        _train_market_env (TrainMarketEnv):
            Training environment.
        _test_market_env (TrainMarketEnv):
            Testing environment.
        _async_env_pipes (list[AbstractPipe]):
            List of pipes saved after parallel training. If training is
            resumed, then the saved pipes are used to restore the state
            of the parallel environments. Useful for multi-stage
            training with different configurations.

    Properties:
    ----------
        model (nn.Module):
            Returns the agent's model.
        pipe (AbstractPipe):
            Returns the agent's pipe. If n_envs > 1 then instead of
            using the agent's pipe, a deep copy of the agent's pipe is
            used to avoid simultaneous modification of the same pipe by
            parallel environments.
        dataset_metadata (DatasetMetadata):
            Returns the dataset metadata of the agent. If dataset
            metadata is not set, then it is set to the metadata of the
            dataset in the file path.
        async_env_pipes (list[AbstractPipe]):
            Returns a list of pipes saved after parallel training. If
            training is resumed, then the saved pipes are used to
            restore the state of the parallel environments. Useful for
            multi-stage training with different configurations.

    Methods:
    -------
        _get_train_test_data_feeders() -> Tuple[StaticDataFeeder,
        StaticDataFeeder]:
            Splits the dataset time horizon into training and testing
            intervals, and creates data feeders for training and testing
            environments. If train ratio is 0.8 then the first 80% of
            the dataset is is used for training and the last 20% is used
            for testing. If train ratio is 1 then the entire dataset is
            used for training and no testing is performed.
        _get_market_env() -> TrainMarketEnv | AsyncVectorEnv |
        SyncVectorEnv:
            If n_envs = 1 or caller is test then a single environment is
            returned and agent's pipe is used to pipe the environment.
            when caller is train and n_envs > 1, deep copies of agent
            pipe is created. This is to avoid complications arised
            during parallel training and possibly modifying the same
            pipe object at the same time. Pipes created in parallel
            training will be saved for future reference so that when
            performing more paralell training state of the parallel
            pipes are preserved.
        _run_episode(env: TrainMarketEnv, random_actions: bool = False)
        -> None:
            Runs a single episode on the given environment. If random
            actions are used then the agent's model is not used to
            generate actions.
        test(n_episodes: int = 1, n_warmup: int = 0) -> None:
            This method is used to test the agent's performance on the
            testing dataset. if n_warmup > 0 then n_warmup episodes are
            run with random actions before testing.
        train(algorithm: OnPolicyAlgorithm, steps: int = 1_000_000,
        **kwargs) -> None:
            Trains the agent using the given algorithm for the given
            number of steps.

    Notes:
    -----
    Note that if n_envs > 1 then a deep copy of pipe is created for each
    environment. Thus agent's pipe attribute is not used. In this case
    perform a final train/test on a single environment with target
    initial conditions. This way agent's pipe is used and its
    observation normalizer stats will be tuned to live account initial
    cash/assets, prior to deoployment for trading. Training on multiple
    environments with random initial conditions can potentially help the
    model generalize better.
    """
    def __init__(self,
                 agent: Agent,
                 file_path: os.PathLike,
                 dataset_name: str,
                 n_chunks: int = 1,
                 train_ratio: float = 1,
                 n_envs: int = 1,
                 async_envs: bool = True,
                 exclusive_envs: True = False,
                 initial_cash_range: Optional[Tuple[float, float]] = None,
                 initial_assets_range: Optional[Tuple[float, float]] = None
                 ) -> None:

        super().__init__(agent=agent,
                         file_path=file_path,
                         dataset_name=dataset_name,
                         n_chunks=n_chunks,
                         train_ratio=train_ratio,
                         n_async_envs=n_envs,
                         async_envs=async_envs,
                         exclusive_async_envs=exclusive_envs,
                         initial_cash_range=initial_cash_range,
                         initial_asset_quantities_range=initial_assets_range
                         )

        return None
    
    def get_async_env(self, env_callables) -> Union[DummyVecEnv, SubprocVecEnv]:
        """
        Returns a vectorized environment for parallel training.
        """
        if self.async_envs:
            market_env = SubprocVecEnv(env_callables)
        else:
            market_env = DummyVecEnv(env_callables)
        return market_env

    def _set_env(self, env: TrainMarketEnv) -> None:
        self.model.save("temp_model")
        self.model = self.model.load("temp_model", env)
        os.remove("temp_model")
        return None
    
    def train(self,
              n_warmup_episodes: int = 1,
              steps: int = 1_000_000,
              progress_bar: bool = True,
              **kwargs) -> nn.Module:
        """
        This method is used to train the agent using the given
        algorithm.

        Args:
        ----
            algorithm (AbstractModel):
                Algorithm to be used for training.
            steps (int):
                Number of steps to train the agent for.
            progress_bar (bool):
                If True, a progress bar is shown during training.
            **kwargs:
                Additional keyword arguments.
            
        """
        market_env = self._get_market_env()
        for episode in range(n_warmup_episodes):
            self.run_episode(market_env, random_actions=True)
        
        self.model.train(
            market_env,
            total_timesteps=steps,
            progress_bar=progress_bar)

        return None
