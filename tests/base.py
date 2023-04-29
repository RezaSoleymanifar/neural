import unittest
import sys
import os
from log import logger

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from neural.client.alpaca import AlpacaClient

class BaseConnectionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.client = AlpacaClient()
        cls.client.connect()
        logger.info('Base connection test: SUCCESSFUL')


if __name__ == '__main__':
    unittest.main()
