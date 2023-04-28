from datetime import datetime
from typing import List
import os

from alpaca.trading.enums import AssetClass
import pandas as pd
import numpy as np

from alpaca.trading.enums import AssetClass

from neural.client.alpaca import AlpacaClient
from neural.common import logger
from neural.common.constants import ACCEPTED_DOWNLOAD_RESOLUTIONS
from neural.data.enums import DatasetType, DatasetMetadata
from neural.tools.base import (progress_bar, to_timeframe, 
    create_column_schema, validate_path)
from neural.tools.base import Calendar
    

class AlpacaDataDownloader():

    """
    A class to download and process financial data using the Alpaca API.

    The AlpacaDataFetcher class handles validation of symbols and resolutions,
    data downloading, and data processing tasks. It works in conjunction with
    the AlpacaClient class to fetch the required data from the Alpaca API
    and process it for further use.
    """

    def __init__(
        self,
        client: AlpacaClient
        ) -> None:

        """
        Initializes the AlpacaDataFetcher class.

        Args:
            client (AlpacaClient): An instance of the AlpacaClient class.

        Returns:
            None
        """

        self.client = client

        return None


    def _validate_resolution(self, resolution):
        
        """
        Validates the resolution of the dataset.

        Parameters:
        resolution (str): The resolution of the dataset.

        Returns:
        None.

        Raises:
        ValueError: If the resolution is not one of the accepted resolutions.
        """

        if resolution not in ACCEPTED_DOWNLOAD_RESOLUTIONS:
            raise ValueError(f'Accepted resolutions: {ACCEPTED_DOWNLOAD_RESOLUTIONS}.')

        return



    def download_raw_dataset(
        self,
        dataset_type: DatasetType,
        symbols: List[str],
        asset_class: AssetClass,
        resolution: str,
        start: datetime,
        end: datetime,
        ) -> None:
        """
        Downloads raw dataset from the Alpaca API.

        Args:
            dataset_type (DatasetType): The type of dataset to download (bar, quote, or trade).
            symbols (List[str]): A list of symbols to download.
            asset_class (AssetClass): The asset class to download.
            resolution (str): The resolution of the dataset to download (e.g., "1Min").
            start (datetime): The start date and time of the dataset to download.
            end (datetime): The end date and time of the dataset to download.

        Returns:
            pd.DataFrame: The downloaded dataset as a pandas DataFrame.
        """

        resolution = to_timeframe(resolution)

        data_fetcher = AlpacaDataDownloader(self.client)

        downloader, request = data_fetcher.get_downloader_and_request(
            dataset_type=dataset_type,
            asset_class=asset_class)

        data = downloader(request(
            symbol_or_symbols=symbols,
            timeframe=resolution,
            start=start,
            end=end))

        try:
            data_df = data.df

        except KeyError:
            raise KeyError(f'No data in requested range {start}-{end}')

        return data.df

    def download_features_to_hdf5(
        self,
        file_path: str | os.PathLike,
        target_dataset_name: str,
        dataset_type: DatasetType,
        symbols: List[str],
        resolution: str,
        start_date: str | datetime,
        end_date: str | datetime
    ) -> DatasetMetadata:
        """
        Downloads financial features data for the given symbols and saves it in an HDF5 file format.
        
        Args:
            file_path (str | os.PathLike): The file path of the HDF5 file to save the data.
            target_dataset_name (str): The name of the dataset to create in the HDF5 file.
            dataset_type (DatasetType): The type of dataset to download. Either 'BAR', 'TRADE', or 'QUOTE'.
            symbols (List[str]): The list of symbol names to download features data for.
            resolution (str): The frequency at which to sample the data. One of '1Min', '5Min', '15Min', or '30Min'.
            start_date (str | datetime): The start date to download data for, inclusive. If a string, it should be in
                the format 'YYYY-MM-DD'.
            end_date (str | datetime): The end date to download data for, inclusive. If a string, it should be in
                the format 'YYYY-MM-DD'.
                
        Returns:
            metadata (DatasetMetadata): The metadata of the saved dataset.
        """

        validate_path(file_path=file_path)

        asset_class = self._validate_symbols(symbols)
        self._validate_resolution(resolution=resolution)

        calendar = Calendar(asset_class=asset_class)
        schedule = calendar.schedule(
            start_date=start_date, end_date=end_date)

        if len(schedule) == 0:
            raise ValueError(
                'No market hours in date range provided.')

        logger.info(
            f"Downloading dataset for {len(symbols)} symbols | resolution: {resolution} |"
            f" {len(schedule)} market days from {start_date} to {end_date}")

        # shows dataset download progress bar
        progress_bar_ = progress_bar(len(schedule))

        # fetches and saves data on a daily basis
        for market_open, market_close in schedule.values:

            raw_dataset = self.download_raw_dataset(
                dataset_type=dataset_type,
                symbols=symbols,
                asset_class=asset_class,
                resolution=resolution,
                start=market_open,
                end=market_close)

            # check for missing symbols
            dataset_symbols = raw_dataset.index.get_level_values(
                'symbol').unique().tolist()
            missing_symbols = set(dataset_symbols) ^ set(symbols)

            if missing_symbols:
                raise ValueError(
                    f'No data for symbols {missing_symbols} in {market_open}, {market_close} time range.')

            # reordering rows to symbols. API does not maintain symbol order.
            raw_dataset = raw_dataset.reindex(
                index=pd.MultiIndex.from_product([
                    symbols, raw_dataset.index.levels[1]]))

            # resets multilevel symbol index
            raw_dataset = raw_dataset.reset_index(level=0, names='symbol')

            processed_groups = list()
            # raw data is processed symbol by symbol
            for symbol, group in raw_dataset.groupby('symbol'):

                processed_group = DataProcessor.reindex_and_forward_fill(
                    data=group, open=market_open,
                    close=market_close, resolution=resolution)

                processed_groups.append(processed_group)

            features_df = pd.concat(processed_groups, axis=1)
            features_df = features_df.select_dtypes(include=np.number)

            column_schema = create_column_schema(data=features_df)

            features_np = features_df.to_numpy(dtype=np.float32)
            n_rows, n_columns = features_np.shape

            metadata = DatasetMetadata(
                dataset_type=[dataset_type],
                column_schema=column_schema,
                asset_class=asset_class,
                symbols=symbols,
                start=market_open,
                end=market_close,
                resolution=resolution,
                n_rows=n_rows,
                n_columns=n_columns,
            )

            DatasetIO.write_to_hdf5(
                file_path=file_path,
                data_to_write=features_np,
                metadata=metadata,
                target_dataset_name=target_dataset_name)

            progress_bar_.set_description(
                f"Density: {DataProcessor.running_dataset_density:.0%}")

            progress_bar_.update(1)

        progress_bar_.close()

        return None


class AlpacaDataStreamer:
    pass