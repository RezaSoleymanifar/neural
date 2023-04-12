from neural.core.data.enums import DatasetMetadata
from neural.core.data.ops import AlpacaMetaClient, AsyncDataFeeder
from neural.meta.env.base import TradeMarketEnv, TrainMarketEnv
from neural.meta.env.pipe import AbstractPipe
from alpaca.trading.enums import AccountStatus
from torch import nn
import os


class Trader:

    def __init__(self, client: AlpacaMetaClient, model : nn.Module, pipe: AbstractPipe, dataset_metadata: DatasetMetadata):

        self.dataset_metadata = dataset_metadata
        self.data_feeder = AsyncDataFeeder(self.dataset_metadata)
        self._trade_market_env = TradeMarketEnv(self.data_feeder)
        self._
        self.client = client
        self.model = None

        try:
            client.account.status == AccountStatus.ACTIVE
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_object: nn.Module):
        
        if not isinstance(model_object, nn.Module):
            raise TypeError(
                f'model {model_object} needs to be instance of {nn.Module}.'
            )
        
        self._model = model_object


    @property
    def train_market_env(self):
        return self._train_market_env
    
    @property
    def trade_market_env(self):
        return self._trade_market_env

    @trade_market_env.setter
    def trade_market_env(self, trade_market_env_object: TradeMarketEnv):

        if not isinstance(trade_market_env_object, TradeMarketEnv):

            raise TypeError(
                f'Trading environment {trade_market_env_object} needs to be instance of {TradeMarketEnv}.'
            )

        self._trade_market_env = trade_market_env_object

    def set_model(self, model: torch.nn.Module) -> None:
        self.model = model

    def trade(self, resolution: str, cash: float, commission: float, max_episode_steps: int) -> None:
        # Create the trade environment
        trade_env = TradeMarketEnv(
            cash=cash,
            commission=commission,
            max_episode_steps=max_episode_steps,
            resolution=resolution,
            metadata=self.dataset_metadata,
            data_feeder=self.data_feeder,
            client=self.client,
            model=self.model
        )

        # Run the trade environment
        obs = trade_env.reset()
        while True:
            action = self.model(obs)
            obs, reward, done, info = trade_env.step(action)
            if done:
                break

    def warmup_pipe(self, env: TrainMarketEnv, n_episodes: Optional[int] = None):
        # warms up pipe wrappers for application of new env
        # initial investement dpendent wrappers like
        # observation normalizers will get readjusted
        # with this method.
        piped_train_env = self.pipe(self.env)
        piped_train_env.reset()

        n_episodes = n_episodes if n_episodes is not None else 1

        for _ in range(n_episodes):
            while True:

                action = piped_train_env.action_space.sample()
                *_, done, _ = piped_train_env.step(action)

                if done:
                    break

    def trade(self):

        if self.trade_market_env is None:

            raise ValueError(
                f'A trading environement of type {TradeMarketEnv} must be provided for trading.'
            )

        if self.model is None:

            raise ValueError(
                f'A model object of type {nn.Module} must be provided for trading.'
            )

        piped_env = self.pipe(self.trade_market_env)
        observation = self.reset()

        while True:

            action = self.model(observation)
            observation, reward, done, info = piped_env.step(action)
