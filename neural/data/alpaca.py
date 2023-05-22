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
        asset_type: AssetType,
        symbols: List[AlpacaAsset],
        resolution: str,
        start: datetime,
        end: datetime,
        ) -> None:

        """
        Downloads raw dataset from the Alpaca API. This is a dataframe
        downloaded from the API. Typically daily data is downloaded in
        chunks using this method and saved to disk day by day. This is
        because the Alpaca API has a limit on the number of rows that
        can be downloaded at once and also to save progress. Note that
        in Alpaca API the download clients are asset type specific. This
        means that the data for stocks and cryptocurrencies are
        downloaded separately. A mix of stocks and cryptocurrencies
        cannot be written to the same HDF5 file, since they have
        different calendar types and trading hours. The dataset requires
        underlying data to have identical trading hours and calendar
        types. This also can restricts the assets that can be traded 
        concurrently even for stocks, since some stocks have different
        trading hours and calendar types due to being listed on
        different exchanges. In Alpaca API however all stock tickers are
        traded in New York Stock Exchange, so this is not an issue. 

        Args:
        ----------
            dataset_type (DatasetType):
                The type of dataset to download. Either 'BAR', 'TRADE',
                or 'QUOTE'.
            asset_type (AssetType):
                The type of asset to download data for. Either 'STOCK'
                or 'CRYPTOCURRENCY'.
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
            dataset (pd.DataFrame):
                The raw dataset downloaded from the Alpaca API.
        """

        resolution = to_timeframe(resolution)

        downloader, request = self.data_client.get_downloader_and_request(
            dataset_type=dataset_type,
            asset_type=asset_type)

        data = downloader(request(
            symbol_or_symbols=symbols,
            timeframe=resolution,
            start=start,
            end=end))

        try:
            dataset = data.df

        except KeyError:
            raise KeyError(f'No data in requested range {start}-{end}')

        return dataset


    def download_to_hdf5(
        self,
        file_path: str | os.PathLike,
        dataset_name: str,
        dataset_type: AlpacaDataSource.DatasetType,
        symbols: List[str],
        start_date: str | datetime,
        end_date: str | datetime,
        resolution: Optional[str],
        ) -> None:

        """
        Downloads financial features data for the given symbols and
        saves it in an HDF5 file format. A mix of stocks and
        cryptocurrencies cannot be written to the same HDF5 file, since
        they have different calendar types and trading hours. The
        dataset requires underlying data to have identical trading hours
        and calendar types. This also can restricts the assets that can
        be traded concurrently even for stocks, since some stocks have
        different trading hours and calendar types due to being listed
        on different exchanges. In Alpaca API however all stock tickers
        are traded in New York Stock Exchange, so this is not an issue. 

        A mix of marginable and non-marginable assets cannot be written
        to the same HDF5 file, since they have different marginability
        types. Handling marginable and non-marginable assets is not
        supported in the library.

        Args:
        ----------
            file_path (str | os.PathLike):
                The file path of the HDF5 file to write to.
            dataset_name (str):
                The name of the dataset to write to in the HDF5 file.
            dataset_type (DatasetType):
                The type of dataset to download. Either 'BAR', 'TRADE',
                or 'QUOTE'.
            symbols (List[str]):
                The list of symbol names to download features data for.
                Example: 'AAPL', 'MSFT', 'GOOG', 'BTC/USD', 'ETH/USD'.
            start_date (str | datetime):
                The start date to download data for, inclusive. example:
                '2020-01-01', or datetime(2020, 1, 1), or '05/01/2020'.
                This should be a format accepted by pandas to_datetime
                function. More info here:
                https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
            end_date (str | datetime):
                The end date to download data for, inclusive. example:
                '2020-01-01', or datetime(2020, 1, 1), or '05/01/2020'.
                This should be a format accepted by pandas to_datetime
            resolution (str):
                The frequency at which to sample the data. One of
                '1Min', '5Min', '15Min', or '30Min'.
        
        Raises:
        ----------
            ValueError:
                If the resolution is not accepted.
            ValueError:
                If the symbols argument is an empty sequence.
            ValueError:
                If the symbols argument contains duplicate symbols.
            ValueError:
                If the asset types of the symbols are not homogenous.
            ValueError:
                If the marginability types of the symbols are not
                homogenous.
            ValueError:
                If there are no market hours in the given date range.
            ValueError:
                If there is no data for some symbols in the given date
                range.
        """

        validate_path(file_path=file_path)
        self._validate_resolution(resolution=resolution)

        if not symbols:
            raise ValueError(
                'symbols argument cannot be an empty sequence.')
        duplicate_symbols = [
            symbol for symbol in set(symbols) if symbols.count(symbol) > 1]
        if duplicate_symbols:
            raise ValueError(f'Duplicate symbols found: {duplicate_symbols}.')
        
        assets = self.data_client.symbols_to_assets(symbols)
        asset_types = set(asset.asset_type for asset in assets)
        marginability_types = set(asset.marginable for asset in assets)

        if len(asset_types) != 1:
            raise ValueError(f'Non-homogenous asset types: {asset_types}.')
        if len(marginability_types) != 1:
            raise ValueError(
                f'Non-homogenous marginability types: {marginability_types}.')
        
        calendar_type_map = {AssetType.STOCK: CalendarType.NEW_YORK_STOCK_EXCHANGE,
                        AssetType.CRYPTOCURRENCY: CalendarType.TWENTY_FOUR_SEVEN}
        asset_type = asset_types.pop()
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
                asset_type=asset_type,
                symbols=symbols,
                resolution=resolution,
                start=start,
                end=end)

            symbols_in_dataset = dataset.index.get_level_values(
                'symbol').unique().tolist()
            missing_symbols = set(symbols_in_dataset) ^ set(symbols)

            if missing_symbols:
                raise ValueError(
                    f'No data for symbols {missing_symbols} in '
                    f'{start}, {end} time range.')

            dataset = dataset.reindex(
                index=pd.MultiIndex.from_product([
                    symbols, dataset.index.levels[1]]))
            dataset = dataset.reset_index(level=0, names='symbol')

            data_processor = AlpacaDataProcessor()
            processing_statistics = data_processor.processing_statistics
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
            features_array = features_df.to_numpy(dtype=GLOBAL_DATA_TYPE)

            dataset_metadata = DatasetMetadata(
                data_schema=data_schema,
                feature_schema=feature_schema,
                resolution=resolution,
                calendar_type = calendar_type,
                start = start,
                end = end,
            )

            to_hdf5(
                file_path=file_path,
                numpy_array=features_array,
                dataset_metadata=dataset_metadata,
                dataset_name=dataset_name)

            progress_bar_.set_description(
                f'low:{processing_statistics.minimum:.0%}/'
                f'high:{processing_statistics.maximum:.0%}/'
                f'mean:{processing_statistics.mean:.0%}')

            progress_bar_.update(1)

        progress_bar_.close()

        return None


class AlpacaDataStreamer:
    """
    This class is responsible for streaming data from the Alpaca API.
    Usually a dataset is downloaded from the Alpaca API and saved to
    disk in an HDF5 file format. This class is responsible for
    streaming the live data that matches the downloaded dataset. This
    class is also responsible for processing the data in a consistent
    way with the downloaded dataset. This is important, since the
    downloaded dataset is used to train the model and the streamed data
    is used to deploy the model. If the data is not processed in a
    consistent way, the model will not be able to make predictions on
    the streamed data.
    """
    def __init__(self, data_client: AlpacaDataClient) -> None:
        """
        Initializes the AlpacaDataStreamer class. 
        """
        pass


class AlpacaDataProcessor:
    """
    A class to process financial data downloaded from the Alpaca API.
    This includes reindexing and forward filling missing rows in the
    data. This is important, since even with a perfect data collection
    process there can be missing rows in the data, due to trading halt
    events, or other anomalies. In this case forward filling is used to
    indicate features has not changed over time when no trade occurs.

    Example:
    ----------
    for NYSE stocks, the market opens at 9:30 AM and closes at 4:00 PM.
    If resolution = 1Min, then the data is sampled every minute. This
    means that the data is sampled at 9:30 AM, 9:31 AM, 9:32 AM, ...,
    3:59 PM, 4:00 PM. If there is no trade at 9:31 AM, then the data
    will be missing for that minute. In this case forward filling is
    used to indicate that the features has not changed over time when no
    trade occurs. If data at first interval is missing, then forward
    filling won't work, since there is no data to forward fill from. In
    this case backward filling is used to fill the missing value with
    closes non-missing value row. If after forward/backward filling
    there is still missing data, then the entire dataset is empty, 
    namely there is not data for the given time range.
    """
    def __init__(self):

        self.processing_statistics = RunningStatistics()

    def reindex_and_forward_fill(
            self,
            data: pd.DataFrame,
            open: datetime,
            close: datetime,
            resolution: str):
        """
        Reindexes and forward fills missing rows in [open, close)
        range, i.e. time_index = open means any time with open <= time <
        open + resolution will be included in the time_index interval.
        The final interval will have time_index = close - resolution.

        Args:
        ----------
            data (pd.DataFrame):
                The data to reindex and forward fill.
            open (datetime):
                The open time of the time index.
            close (datetime):
                The close time of the time index.
            resolution (str):
                The frequency at which to sample the data. One of
                '1Min', '5Min', '15Min', or '30Min'.

        Returns:
        ----------
            processed (pd.DataFrame):
                The reindexed and forward filled data.
        """

        index = pd.date_range(
            start=open, end=close, freq=resolution, inclusive='left')

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
