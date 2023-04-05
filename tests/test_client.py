import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import unittest
from alpacarl.connect import AlpacaMetaClient
from log import logger


class TestClient(unittest.TestCase):
        
    @classmethod
    def setUpClass(cls):

        cls.client = AlpacaMetaClient()
        cls.client.setup_clients_and_account()

        logger.info('connection test passed.')
        
    
    def test_client_functionality(self):
        
        print(
        f"""
        assets: {self.client.assets}
        positions: {self.client.positions}
        asset classes: {self.client.asset_classes}
        exchange: {self.client.exchanges}
        """)
        logger.info(
            'Client functions test successful.')
    
    def set_credentials(client):

        client.set_credentials(
            secret= 'False', key= 'True')
        logger.info(
            'Credentials setting test successful.')


class TestData(TestClient):

    @classmethod

    def setUpClass(cls):
        super().setUpClass()
        logger.info('Client setup for for data testing successful.')
    
    def func():
        pass

if __name__ == '__main__':
    unittest.main()
