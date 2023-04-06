import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import unittest
from base import BaseConnectionTest
from log import logger


class TestClient(BaseConnectionTest):
        
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        logger.info('Client setup test: SUCESSFUL.')
        
    
    def test_functions(self):
        
        print(
        f"""
        assets: {self.client.assets}
        positions: {self.client.positions}
        asset classes: {self.client.asset_classes}
        exchange: {self.client.exchanges}
        """)
        logger.info(
            'Client functions test: SUCCESSFUL.')
    
    def set_credentials(client):

        client.set_credentials(
            secret= 'False', key= 'True')
        
        logger.info(
            'Credentials setup test: SUCCESSFUL.')

    def tearDown(self):
        logger.info('CLIENT TEST DONE')


if __name__ == '__main__':
    unittest.main()
