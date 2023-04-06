from log import logger
import unittest
import sys
import os
from base import BaseConnectionTest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

class TestData(BaseConnectionTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        logger.info('Client setup for data testing successful.')
    

if __name__ == '__main__':
    unittest.main()
