from log import logger
from datetime import datetime
import unittest
import sys
import os
from base import BaseConnectionTest
from alpaca.trading.enums import AssetClass, AssetStatus


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from neural.core.data.ops import AlpacaDataFetcher
from neural.core.data.enums import DatasetType
from neural.common.constants import DOW_JONES_SYMBOLS
from neural.tools.ops import to_timeframe


class TestData(BaseConnectionTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()


    def setUp(self):
        self.data_fetcher = AlpacaDataFetcher(client=self.client)


    def test_validate_symbols(self):
        
        self.assets = self.client.assets

        # empty sequence
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = []
            self.data_fetcher._validate_symbols(SYMBOLS)


        logger.info(cm.exception)
        self.assertEqual(
            str(cm.exception),
            'symbols argument cannot be an empty sequence.')


        # duplicate values
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = ['AAPL', 'AAPL']
            self.data_fetcher._validate_symbols(SYMBOLS)

        logger.info(cm.exception)
        self.assertEqual(
            str(cm.exception),
            f'Symbols {list(set(SYMBOLS))} have duplicate values.')


        # valid name
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = ['NOT_A_SYMBOL']
            self.data_fetcher._validate_symbols(SYMBOLS)


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
                self.data_fetcher._validate_symbols(SYMBOLS)

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
                self.data_fetcher._validate_symbols(SYMBOLS)

            logger.info(cm.exception)
            self.assertEqual(
                str(cm.exception),
                f'Symbol {SYMBOLS.pop()} is not an active symbol.')

        self.data_fetcher._validate_symbols(DOW_JONES_SYMBOLS)



        # different asset classes
        with self.assertRaises(ValueError) as cm:
            SYMBOLS = ['AAPL', 'BTC/USD']
            self.data_fetcher._validate_symbols(SYMBOLS)

        logger.info(cm.exception)
        self.assertEqual(str(cm.exception), 
            'Symbols are not of the same asset class.')
        
        SYMBOLS = ['AAPL', 'GOOGL']
        asset_class = self.data_fetcher._validate_symbols(SYMBOLS)
        self.assertEqual(asset_class, AssetClass.US_EQUITY)
        logger.info(asset_class)

        SYMBOLS = ['BTC/USD', 'ETH/USD']
        asset_class = self.data_fetcher._validate_symbols(SYMBOLS)
        self.assertEqual(asset_class, AssetClass.CRYPTO)
        logger.info(asset_class)


        self.data_fetcher._validate_symbols(DOW_JONES_SYMBOLS)
        
        logger.info('Symbol validation test: SUCCESSFUL.')


    def test_dataset_download(self):
                
        resolution = '1Min'
        start_date = '02/03/2023'
        end_date = '03/03/2023'

        symbols = ['WMT', 'GOOGL', 'AAPL', 'AXP']

        raw_dataset = self.data_fetcher.download_features_to_hdf5(
            file_path='../storage/test.hdf5',
            target_dataset_name='init',
            dataset_type=DatasetType.BAR,
            symbols=symbols,
            resolution=resolution,
            start_date=start_date,
            end_date=end_date
        )

        with self.assertRaises(ValueError) as cm:
            start_date = '02/10/2023'
            end_date = '04/03/2023'

            raw_dataset = self.data_fetcher.download_features_to_hdf5(
                file_path='../storage/test.hdf5',
                target_dataset_name='init',
                dataset_type=DatasetType.BAR,
                symbols=symbols,
                resolution=resolution,
                start_date=start_date,
                end_date=end_date
            )

        logger.info(cm.exception)
        self.assertEqual(
            str(cm.exception),
            'Current end time: 2023-03-03 21:00:00+00:00, and appended dataset '
            'start time 2023-02-10 14:30:00+00:00 overlap.')

        logger.info('dataset download test: SUCCESSFUL.')

if __name__ == '__main__':
    unittest.main()
