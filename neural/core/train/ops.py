from abc import ABC, abstractmethod
from neural.meta.env.base import AbstractMarketEnv
from typing import Optional


class AbstractTrainer(ABC):

    @abstractmethod
    def train(self, *args, **kwargs):

        raise NotImplementedError


class AbstractRewardShaper(ABC):
    pass
