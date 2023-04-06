from log import logger
import unittest
import sys
import os
from base import BaseConnectionTest
from alpaca.trading.enums import AssetClass
import logging


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from alpacarl.core.data.ops import DatasetDownloader
from alpacarl.common.constants import DOW_JONES_SYMBOLS
from alpaca.trading.enums import AssetStatus


class TestData(BaseConnectionTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
    

    def test_symbol_validation(self):

        self.dataset_downloader = DatasetDownloader(client=self.client)
        self.assets = self.client.assets



        # empty sequence
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = []
            self.dataset_downloader._validate_symbols(SYMBOLS)

        logger.info(cm.exception)
        self.assertEqual(
            str(cm.exception),
            'symbols argument cannot be an empty sequence.')



        # duplicate values
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = ['AAPL', 'AAPL']
            self.dataset_downloader._validate_symbols(SYMBOLS)

        logger.info(cm.exception)
        self.assertEqual(
            str(cm.exception),
            f'Symbols {list(set(SYMBOLS))} have duplicate values.')



        # valid name
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = ['???????']
            self.dataset_downloader._validate_symbols(SYMBOLS)

        logger.info(cm.exception)
        self.assertEqual(
            str(cm.exception),
            f"Symbol {SYMBOLS[0]} is not a known symbol.")



        try:
            asset = self.assets[self.assets['tradable'] == False].iloc[0]

        except(IndexError):
            logger.info('No non-tradable asset exist to perform test.')

        else:

            with self.assertRaises(ValueError) as cm:
                SYMBOLS = [asset['symbol']]
                self.dataset_downloader._validate_symbols(SYMBOLS)

            logger.info(cm.exception)
            self.assertEqual(
                str(cm.exception),
                f'Symbol {SYMBOLS.pop()} is not a tradable symbol.')



        try:
            asset = self.assets[(
                self.assets['status'] == AssetStatus.INACTIVE)
                & (self.assets['tradable'] == True)].iloc[0]
            
        except(IndexError):
            logger.info(
                'No non-tradable, inactive asset exists to perform inactive asset test.')

        else:

            with self.assertRaises(ValueError) as cm:
                SYMBOLS = [asset['symbol']]
                self.dataset_downloader._validate_symbols(SYMBOLS)

            logger.info(cm.exception)
            self.assertEqual(
                str(cm.exception),
                f'Symbol {SYMBOLS.pop()} is not an active symbol.')

        self.dataset_downloader._validate_symbols(DOW_JONES_SYMBOLS)



        # different asset classes
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = ['AAPL', 'BTC/USD']
            self.dataset_downloader._validate_symbols(SYMBOLS)

        logger.info(cm.exception)
        self.assertEqual(str(cm.exception), 
            'Symbols are not of the same asset class.')
        
        SYMBOLS = ['AAPL', 'GOOGL']
        asset_class = self.dataset_downloader._validate_symbols(SYMBOLS)
        self.assertEqual(asset_class, AssetClass.US_EQUITY)
        logger.info(asset_class)

        SYMBOLS = ['BTC/USD', 'ETH/USD']
        asset_class = self.dataset_downloader._validate_symbols(SYMBOLS)
        self.assertEqual(asset_class, AssetClass.CRYPTO)
        logger.info(asset_class)

        logger.info('Symbol validation test: SUCCESSFUL.')

if __name__ == '__main__':
    unittest.main()
