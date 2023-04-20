from log import logger
import unittest
import sys
import os

from base import BaseConnectionTest
from neural.meta.env.base import TrainMarketEnv
from neural.meta.env.pipe import NetWorthRelativeShortMarginPipe
from neural.core.data.ops import StaticDataFeeder, DatasetIO

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'neural'))


class TestData(BaseConnectionTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):

        dataset_metadata, datasets = DatasetIO.load_from_hdf5(
            file_path='../storage/test.hdf5', target_dataset_name='init')
        self.data_feeder = StaticDataFeeder(
            dataset_metadata=dataset_metadata, datasets=datasets, n_chunks=200)
        
    def test_train_env(self):


        market_env = TrainMarketEnv(market_data_feeder=self.data_feeder)

        market_pipe = NetWorthRelativeShortMarginPipe(
            trade_ratio=0.1, verbosity=10, initial_margin=0.9, short_ratio=0.1)
        piped_market_env = market_pipe.pipe(market_env)

        observation = piped_market_env.reset()

        while True:

            actions = piped_market_env.action_space.sample()
            observation, reward, done, info = piped_market_env.step(actions)

            if done:
                break
        
        logger.info('Train environment test: SUCCESSFUL')


if __name__ == '__main__':
    unittest.main()
