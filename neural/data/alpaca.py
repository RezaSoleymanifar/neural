from datetime import datetime
from typing import List, Optional
import os

import pandas as pd
import numpy as np

from neural.client.alpaca import AlpacaDataClient
from neural.common import logger
from neural.common.constants import (
    ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS, GLOBAL_DATA_TYPE)
from neural.data.base import (
    DatasetMetadata, AlpacaDataSource, CalendarType, AlpacaAsset)
from neural.data.enums import AssetType, AlpacaDataSource
from neural.utils.io import to_hdf5
from neural.utils.base import (
    progress_bar, validate_path, RunningStatistics)
from neural.utils.misc import to_timeframe



class AlpacaDataDownloader():

    """
    A class to download and process financial data using the Alpaca API.
    This includes bar, quote, and trade data. The data is downloaded and
    processed from the Alpaca API on a daily basis and saved to disk in
    an HDF5. The download later can be resume for consecutive market 
    days. The data is downloaded in chunks to avoid exceeding the
    memory limit and saving progress.

    Attributes:
    ----------
        data_client (AlpacaDataClient): An instance of the
        AlpacaDataClient class. This is responsible for communicating
        with the Alpaca API and providing the basic facility to download
        data.

    Methods:
    -------
        __init__:
            Initializes the AlpacaDataFetcher class.
        _validate_resolution:
            Validates the resolution of the dataset. Resolutions not
            accepted can potentially lead to incoherencies in the end to
            end process, due to irregular aggregation output from the
            Alpaca API. For example resoluion = 43Min can shift the
            start and end times of the trading day in a way that is
            unpredictable and inconsistent with other resoluions.
        download_dataset: 
            Downloads raw dataset from the Alpaca API. This is a
            dataframe downloaded from the API. Typically daily data is
            downloaded in chunks using this method and saved to disk day
            by day. This is to avoid memory issues and to save progress.
        download_to_hdf5: 
            Downloads financial features data for the
            given symbols and saves it in an HDF5 file format. Ensures
            the data is downloaded in a consistent way and that the data
            is saved in a consistent format, supported by the library.
    """

    def __init__(self, data_client: AlpacaDataClient) -> None:

        """
        Initializes the AlpacaDataFetcher class.

        Args:
        ----------
            data_client (AlpacaDataClient): An instance of the
            AlpacaDataClient class. This is responsible for
            communicating with the Alpaca API and providing the basic
            facility to download data.
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

        Args: 
        ----------
        resolution (str): 
            The resolution of the dataset.

        Raises:
        ----------
        ValueError:
            If the resolution is not accepted.
        """

        if resolution not in ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS:
            raise ValueError(
                f'Accepted resolutions: {ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS}.')

        return None


    def download_dataset(
        self,
        dataset_type: AlpacaDataSource.DatasetType,
        assets: List[AlpacaAsset],
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
        ----------
            dataset_type (DatasetType):
                The type of dataset to download. Either 'BAR', 'TRADE',
                or 'QUOTE'.
            symbols (List[str]):
                The list of symbol names to download features data for.
            resolution (str):
                The frequency at which to sample the data. One of
                '1Min', '5Min', '15Min', or '30Min'.
            start (datetime):
                The start date to download data for, inclusive.
            end (datetime):
                The end date to download data for, inclusive.

        Returns:
        ----------
            dataset_dataframe (pd.DataFrame):
                The raw dataset downloaded from the Alpaca API.
            
        Raises:
        ----------
            ValueError:
                If the symbols argument is empty or if there are
                duplicate symbols in the symbols argument.
            KeyError:
                If there is no data in the requested range.
        """

        if not assets:
            raise ValueError(
                'symbols argument cannot be an empty sequence.')

        duplicate_symbols = [
            symbol for symbol in set(assets) if assets.count(symbol) > 1]
        if duplicate_symbols:
            raise ValueError(f'Duplicate symbols found: {duplicate_symbols}.')

        asset_types = set(asset.asset_type for asset in assets)
        if len(asset_types) != 1:
            raise ValueError(f'Non-homogenous asset types: {asset_types}.')
        
        resolution = to_timeframe(resolution)

        downloader, request = self.data_client.get_downloader_and_request(
            dataset_type=dataset_type,
            asset_type=AssetType)

        data = downloader(request(
            symbol_or_symbols=assets,
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

        marginability_types = set(asset.marginable for asset in assets)

        if len(marginability_types) != 1:
            raise ValueError(
                f'Non-homogenous marginability types: {marginability_types}.')
        
        asset_type = assets[0].asset_type
        calendar_type_map = {AssetType.STOCK: CalendarType.NEW_YORK_STOCK_EXCHANGE,
                        AssetType.CRYPTOCURRENCY: CalendarType.TWENTY_FOUR_SEVEN}
        calendar_type = calendar_type_map[asset_type]
        schedule = calendar_type.schedule(
            start_date=start_date, end_date=end_date)
        
        if len(schedule) == 0:
            raise ValueError(
                f'No market hours in date range {start_date}-{end_date}.')
        
        days = len(schedule)
        n_assets = len(assets)
        logger.info('Downloading dataset:'
                    f'\n\t start = {start_date}'
                    f'\n\t end = {end_date}'
                    f'\n\t days = {days}'
                    f'\n\t resolution = {resolution}'
                    f'\n\t n_assets = {n_assets}')

        progress_bar_ = progress_bar(len(schedule))

        for start, end in schedule.values:

            dataset = self.download_dataset(
                dataset_type=dataset_type,
                assets=symbols,
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
            data_schema = {dataset_type: tuple(assets)}

            features_np = features_df.to_numpy(dtype=GLOBAL_DATA_TYPE)

            metadata = DatasetMetadata(
                data_schema=data_schema,
                feature_schema=feature_schema,
                resolution=resolution,
                calendar_type = calendar_type,
                start = start,
                end = end,
            )

            to_hdf5(
                file_path=file_path,
                data_to_write=features_np,
                metadata=metadata,
                dataset_name=dataset_name)

            processing_statistics = data_processor.processing_statistics
            low = processing_statistics.minimum
            high = processing_statistics.maximum
            mean = processing_statistics.mean

            progress_bar_.set_description(
                f"low:{low:.0%}/high:{high:.0%}/mean:{mean:.0%}")

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
