from neural.core.data.ops import AlpacaMetaClient
from torch import nn

class PPOTrainPipe:
    pass

class TradePipe:
    # transforms a training pipe into a trading pipe
    def __init__(
        self,
        client: AlpacaMetaClient,
        ppo_train_pipe: PPOTrainPipe,
        model: nn.module) -> None:
        pass
    pass