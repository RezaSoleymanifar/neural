from typing import List, Type, Dict, Tuple
from gym import Wrapper

from neural.meta.env.wrappers import (
    RunningIndicatorsObsWrapper, 
    RelativePositionSizingActionWrapper, 
    ConsoleTearsheetRenderWrapper)

from neural.meta.env.base import TrainMarketEnv
from dataclasses import dataclass


@dataclass
class Pipe:
    wrapper_classes: List[
        Tuple[Type[Wrapper], Dict]]
    # Other attributes for the pipe

    def Make_pipe(
        wrappers: List[Type[Wrapper]]):

        def pipe(env_instance):
            for wrapper_class in wrappers:
                env_instance = wrapper_class(env_instance)
            return env_instance

        return pipe

class PPOPipe(Pipe):
    def __init__(self):
        wrapper_classes = [
            (RunningIndicatorsObsWrapper, {'arg1': 10, 'arg2': 'foo'}),
            (RelativePositionSizingActionWrapper, {'arg1': 20, 'arg2': 'bar'}),
            (ConsoleTearsheetRenderWrapper, {'arg1': 30, 'arg2': 'baz'})
        ]
        super().__init__(wrapper_classes=wrapper_classes)
