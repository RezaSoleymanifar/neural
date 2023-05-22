from datetime import datetime
from typing import List, Optional
import os

import pandas as pd
import numpy as np

from neural.client.alpaca import AlpacaDataClient
from neural.common import logger
from neural.common.constants import ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS
from neural.data.base import DatasetMetadata, AlpacaDataSource, CalendarType
from neural.data.enums import AssetType, AlpacaDataSource
from neural.utils.time import Calendar
from neural.utils.io import IOHandler
from neural.utils.base import (
    progress_bar, validate_path, Calendar, RunningStatistics)
from neural.utils.misc import to_timeframe



class AlpacaDataDownloader():

    """
    A class to download and process financial data using the Alpaca API.

    The AlpacaDataFetcher class handles validation of symbols and
    resolutions, data downloading, and data processing tasks. It works
    in conjunction with the AlpacaClient class to fetch the required
    data from the Alpaca API and process it for further use.
    """

    def __init__(self, data_client: AlpacaDataClient) -> None:

        """
        Initializes the AlpacaDataFetcher class.

        Args:
            client (AlpacaClient): An instance of the AlpacaClient
            class.

        Returns:
            None
        """

        self.data_client = data_client

        return None


    def _validate_resolution(self, resolution):
        
        """
        Validates the resolution of the dataset. Resolutions not
        accepted can potentially lead to incoherencies in the end to end
        process, due to irregular aggregation output from the Alpaca
        API. For example resoluion = 43Min can shift the start and end
        times of the trading day in a way that is unpredictable and
        inconsistent with other resoluions.

        Parameters: resolution (str): The resolution of the dataset.

        Returns: None.

        Raises: ValueError: If the resolution is not one of the accepted
        resolutions.
        """

        if resolution not in ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS:
            raise ValueError(
                f'Accepted resolutions: {ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS}.')

        return


    def download_dataset(
        self,
        dataset_type: AlpacaDataSource.DatasetType,
        symbols: List[str],
        resolution: str,
        start: datetime,
        end: datetime,
        ) -> None:

        """
        Downloads raw dataset from the Alpaca API. This is a dataframe
        downloaded from the API. Typically daily data is downloaded in
        chunks using this method and saved to disk day by day. This is
        because the Alpaca API has a limit on the number of rows that
        can be downloaded at once and also to save progress.

        Args:
            dataset_type (DatasetType): The type of dataset to download
            (bar, quote, or trade). symbols (List[str]): A list of
            symbols to download. Note that API does not preserve the
            order of the symbols in the output dataframe. asset_class
            (AssetClass): The asset class to download. resolution (str):
            The resolution of the dataset to download (e.g., "1Min",
            "15Min"). start (datetime): The start date and time of the
            dataset to download. end (datetime): The end date and time
            of the dataset to download.

        Returns:
            pd.DataFrame: The downloaded dataset as a pandas DataFrame.
        """

        if not symbols:
            raise ValueError(
                'symbols argument cannot be an empty sequence.')

        duplicate_symbols = [
            symbol for symbol in set(symbols) if symbols.count(symbol) > 1]
        if duplicate_symbols:
            raise ValueError(f'Duplicate symbols found: {duplicate_symbols}.')

        resolution = to_timeframe(resolution)

        downloader, request = self.data_client.get_downloader_and_request(
            dataset_type=dataset_type,
            asset_type=AssetType)

        data = downloader(request(
            symbol_or_symbols=symbols,
            timeframe=resolution,
            start=start,
            end=end))

        try:
            dataset_dataframe = data.df

        except KeyError:
            raise KeyError(f'No data in requested range {start}-{end}')

        return dataset_dataframe


    def download_to_hdf5(
        self,
        file_path: str | os.PathLike,
        dataset_name: str,
        dataset_type: AlpacaDataSource.DatasetType,
        symbols: List[str],
        start_date: str | datetime,
        end_date: str | datetime,
        resolution: Optional[str] = None,
        ) -> DatasetMetadata:

        """
        Downloads financial features data for the given symbols and
        saves it in an HDF5 file format.
        
        Args:
            file_path (str | os.PathLike): The file path of the HDF5
            file to save the data. dataset_name (str): The name of the
            dataset to create in the HDF5 file. dataset_type
            (DatasetType): The type of dataset to download. Either
            'BAR', 'TRADE', or 'QUOTE'. symbols (List[str]): The list of
            symbol names to download features data for. resolution
            (str): The frequency at which to sample the data. One of
            '1Min', '5Min', '15Min', or '30Min'. start_date (str |
            datetime): The start date to download data for, inclusive.
            If a string, it should be in
                the format 'YYYY-MM-DD'.
            end_date (str | datetime): The end date to download data
            for, inclusive. If a string, it should be in
                the format 'YYYY-MM-DD'.
                
        Returns:
            metadata (DatasetMetadata): The metadata of the saved
            dataset.
        """

        validate_path(file_path=file_path)
        self._validate_resolution(resolution=resolution)
        assets = self.data_client.symbols_to_assets(symbols)

        asset_types = set(asset.asset_type for asset in assets)
        marginability_types = set(asset.marginable for asset in assets)

        if len(asset_types) != 1:
            raise ValueError(f'Non-homogenous asset types: {asset_types}.')

        if len(marginability_types) != 1:
            raise ValueError(
                f'Non-homogenous marginability types: {marginability_types}.')
        
        asset_type = asset_types.pop()
        calendar_type_map = {AssetType.STOCK: CalendarType.NEW_YORK_STOCK_EXCHANGE,
                        AssetType.CRYPTOCURRENCY: CalendarType.TWENTY_FOUR_SEVEN}
        calendar = calendar_type_map[asset_type]
        schedule = calendar.schedule(
            start_date=start_date, end_date=end_date)
        
        if len(schedule) == 0:
            raise ValueError(
                f'No market hours in date range {start_date}-{end_date}.')
        
        days = len(schedule)
        n_assets = len(assets)
        logger.info('Downloading dataset...'
                    f'\n\t start = {start_date}'
                    f'\n\t end = {end_date}'
                    f'\n\t days = {days}'
                    f'\n\t resolution = {resolution}'
                    f'\n\t n_assets = {n_assets}')

        progress_bar_ = progress_bar(len(schedule))

        for start, end in schedule.values:

            dataset = self.download_dataset(
                dataset_type=dataset_type,
                symbols=symbols,
                resolution=resolution,
                start=start,
                end=end)

            dataset_symbols = dataset.index.get_level_values(
                'symbol').unique().tolist()
            missing_symbols = set(dataset_symbols) ^ set(symbols)

            if missing_symbols:
                raise ValueError(
                    f'No data for symbols {missing_symbols} in '
                    f'{start}, {end} time range.')

            dataset = dataset.reindex(
                index=pd.MultiIndex.from_product([
                    symbols, dataset.index.levels[1]]))
            dataset = dataset.reset_index(level=0, names='symbol')

            data_processor = AlpacaDataProcessor()
            symbol_groups = list()
            for symbol, group in dataset.groupby('symbol'):

                processed_group = data_processor.reindex_and_forward_fill(
                    data=group, open=start,
                    close=end, resolution=resolution)

                symbol_groups.append(processed_group)

            features_df = pd.concat(symbol_groups, axis=1)
            features_df = features_df.select_dtypes(include=np.number)


            feature_schema = DatasetMetadata.create_feature_schema(
                data=features_df)
            data_schema = Dict[dataset_type, tuple(assets)]

            features_np = features_df.to_numpy(dtype=np.float32)
            n_rows, n_columns = features_np.shape

            metadata = DatasetMetadata(
                dataset_type=[dataset_type],
                column_schema=feature_schema,
                asset_class=asset_class,
                assets=symbols,
                start=start,
                end=end,
                resolution=resolution,
                n_rows=n_rows,
                n_columns=n_columns,
            )

            IOHandler.write_to_hdf5(
                file_path=file_path,
                data_to_write=features_np,
                metadata=metadata,
                dataset_name=dataset_name)

            progress_bar_.set_description(
                f"Density: {AlpacaDataProcessor.running_dataset_density:.0%}")

            progress_bar_.update(1)

        progress_bar_.close()

        return None


class AlpacaDataStreamer:
    def __init__(self, data_client: AlpacaDataClient) -> None:
        pass


class AlpacaDataProcessor:

    def __init__(self):

        self.processing_statistics = RunningStatistics()

    def reindex_and_forward_fill(
            self,
            data: pd.DataFrame,
            open: datetime,
            close: datetime,
            resolution: str):
        """
        Reindexes and forward-fills missing rows in the given DataFrame
        in the [open, close) range based on the given resolution.
        Returns the processed DataFrame.

        :param data: The DataFrame to be processed.
        :type data: pd.DataFrame
        :param open: The open time of the market data interval to
            process.
        :type open: datetime
        :param close: The close time of the market data interval to
            process.
        :type close: datetime
        :param resolution: The frequency of the time intervals in the
            processed data.
        :type resolution: str
        :return: The processed DataFrame.
        :rtype: pd.DataFrame
        """

        # resamples and forward fills missing rows in [open, close)
        # range, i.e. time index = open means open <= time < close.
        index = pd.date_range(
            start=open, end=close, freq=resolution, inclusive='left')

        # creates rows for missing intervals
        processed = data.reindex(index)

        non_nan_count = processed.notna().sum().sum()
        total_count = processed.size
        density = non_nan_count/total_count
        self.processing_statistics.update(density)

        if processed.isna().any().any():
            processed = processed.ffill()

        if processed.isna().any().any():
            processed = processed.bfill()

        return processed
