import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from alpacarl.connect import AlpacaMetaClient


class BaseConnectionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.client = AlpacaMetaClient()
        cls.client.setup_clients_and_account()



if __name__ == '__main__':
    unittest.main()
