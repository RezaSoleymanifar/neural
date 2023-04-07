import unittest
import sys
import os
from log import logger

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from neural.connect.client import AlpacaMetaClient

class BaseConnectionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.client = AlpacaMetaClient()
        cls.client.setup_clients_and_account()
        logger.info('Base connection test: SUCCESSFUL')


if __name__ == '__main__':
    unittest.main()
