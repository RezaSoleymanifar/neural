"""
alpaca.py

Description:

License:
--------
    MIT License. See LICENSE.md file.

Author(s):
-------
    Reza Soleymanifar, Email: Reza@Soleymanifar.com
------------
"""
from dataclasses import dataclass
from datetime import datetime
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from neural.client.alpaca import AlpacaDataClient
from neural.common.constants import (
    ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS, GLOBAL_DATA_TYPE)
from neural.common.log import logger
from neural.data.base import (
    AbstractDataSource, AbstractAsset, CalendarType, DatasetMetadata)
from neural.data.enums import AssetType
from neural.utils.base import (
    progress_bar, validate_path, RunningStatistics)
from neural.utils.io import to_hdf5
from neural.utils.misc import resolution_to_timeframe
from neural.utils.time import Resolution


class AlpacaDataSource(AbstractDataSource):
    """
    Represents Alpaca API as a data source. Provides standardized enums
    for historical and live data from Alpaca API.
    """

    class DatasetType(AbstractDataSource.DatasetType):
        """
        Enumeration class that defines constants for the different types
        of datasets.

        Attributes
        ----------
        BAR (str)
            Represents one bar/candlestick of aggregated trade data over
            a specified interval. Includes following fields:
            - symbol (str):
                the symbol of the asset.
            - timestamp (datetime):
                the closing time stamp of the bar.
            - open (float):
                the opening price of the bar.
            - high (float):
                the highest price of the bar.
            - low (float):
                the lowest price of the bar.
            - close (float):
                the closing price of the bar.
            - volume (int):
                the trade volume of the bar.
            - trade_count (int):
                the number of trades in the bar.
            - vwap (float):
                the volume weighted average price of the bar.
        TRADE : str
            This is a 
        QUOTE : str
            The type of dataset for aggregated quote stream.
        ORDER_BOOK : str
            The type of dataset for aggregated order book data.
        """
        BAR = 'BAR'
        TRADE = 'TRADE'
        QUOTE = 'QUOTE'
        ORDER_BOOK = 'ORDER_BOOK'

    class StreamType(AbstractDataSource.StreamType):
        """
        Enumeration class that defines constants for the different types
        of data streams.

        Attributes
        ----------
        QUOTE : str
            The type of data stream for quotes.
        TRADE : str
            The type of data stream for trades.
        ORDER_BOOK : str
            The type of data stream for order book data.
        """
        BAR = 'BAR'
        TRADE = 'TRADE'
        QUOTE = 'QUOTE'


@dataclass(frozen=True)
class AlpacaAsset(AbstractAsset):
    """
    A dataclass representing a financial asset in Alpaca API:
    https://alpaca.markets/. This can be a stock, or cryptocurrency.
    This class standardizes the representation of assets in Alpaca API.
    Natively encodes the mechanics for opening and closing positions.
    When creating nonmarginable assets, maintenance_margin, shortable,
    and easy_to_borrow attributes are set to None by default.

    Attributes:
    ----------
        symbol: str
            A string representing the symbol or ticker of the asset.
        asset_type: AssetType
            An instance of the `AssetType` enum class representing the
            type of asset.
        fractionable: bool
            A boolean indicating whether the asset can be traded in
            fractional shares. This is useful for trading for example
            cryptocurrencies or stocks that are expensive to buy as a
            whole share.
        marginable: bool
            A boolean indicating whether the asset is a marginable
            asset. Marginable assets can be used as collateral for
            margin trading. Margin trading is a process where the
            brokerage lends money to the trader to buy more assets than
            the trader can afford. More info here:
            https://www.investopedia.com/terms/m/margin.asp.
        shortable: bool
            A boolean indicating whether the asset can be sold short.
            When asset is sold short you sell the asset when you do not
            own it. This way an asset debt is recorded in your account.
            You can then buy the asset at a lower price and return it to
            the brokerage. This form of trading allows making profit in
            a bear market. More info here:
            https://www.investopedia.com/terms/s/shortselling.asp.
        easy_to_borrow: bool
            A boolean indicating whether the asset can be borrowed
            easily. Alpaca API has restrictive rules for hard to borrow
            assets and in general HTB assets cannot be shorted.
        maintenance_margin: float | None
            A float representing the maintenance margin of the asset.
            This means that maintenace_margin * position_value should be
            available in marginable equity at all times. In practice the
            gross maintenance margin for entire portfolio is used to
            measure maintenance margin requirement. If maintenance
            margin requirement is violated the brokerage will issue a
            margin call. More info here:
            https://www.investopedia.com/terms/m/maintenance_margin.asp.
            Alpaca API in reality enforces this at the end of day or
            when it is violated by a greate extent.

    Properties:
    -----------
        shortable: bool | None
            A boolean indicating whether the asset can be sold short
            (i.e., sold before buying to profit from a price decrease).
        easy_to_borrow: bool | None
            A boolean indicating whether the asset can be borrowed
            easily.

    Methods:
    --------
        get_initial_margin(self, short: bool = False) -> float | None
            A float representing the initial margin of the asset.
        get_maintenance_margin(self, price: float, short: bool = False)
            -> float | None A float representing the maintenance margin
            of the asset.
    Notes:
    ------
        The easy_to_borrow, intial_margin, and maintenance_margin
        properties are only valid for assets that are marginable. For
        example, cryptocurrencies are not marginable and therefore do
        not need to set these attributes. For consistency for
        nonmarginable assets default boolean valued attributes are
        returned as False and maintenance margin and initial margin are
        returned as 0 and 1 respectively. nonmarginable assets can only
        be purchased using cash and we assume they cannot be shorted.
        There are rare cases where non-marginable assets can be shorted,
        but this is not supported by this library due to the complexity
        of the process. A mix of marginable and non-marginable assets in
        portoflio is not supporeted either due to the same level of
        irregularities.
    """
    marginable: bool
    maintenance_margin: Optional[float] = None
    shortable: Optional[bool] = None
    easy_to_borrow: Optional[bool] = None

    @property
    def shortable(self) -> bool | None:
        """
        A boolean indicating whether the asset can be sold short (i.e.,
        sold before buying to profit from a price decrease). In Alpaca
        API shorted assets cannot have faractional quantities. Also if
        asset is not marginable it cannot be shorted. There are rare
        cases where non-marginable assets can be shorted, but this is
        not supported by this library due to the complexity of the
        process.

        Returns:
        --------
            bool:
                A boolean indicating whether the asset can be sold
                short.
        """
        return self.shortable if self.marginable else False

    @property
    def easy_to_borrow(self) -> bool | None:
        """
        A boolean indicating whether the asset can be borrowed easily.
        Alpaca API has restrictive rules for hard to borrow assets and
        in general HTB assets cannot be shorted. This library only
        allows easy to borrow assets to be shorted.

        Returns:
        --------
            bool:
                A boolean indicating whether the asset can be borrowed
                easily.
        """
        return self.easy_to_borrow if self.marginable else False

    def get_initial_margin(self, short: bool = False) -> float | None:
        """
        A float representing the initial margin of the asset. 25% margin
        for long positions and 150% margin for short positions is a
        FINRA requirement:
        https://www.finra.org/filing-reporting/regulation-t-filings.
        Alpaca API has a 50% margin requirement for opening positions,
        by default. Initial margin for nonmarginable assets is 1.0
        namely entire value of trade needs to be available in cash.
        Since nonmarginable assets cannot be margined this is an abuse
        of terminalogy to provide a convenient interface for working
        with marginable and onnmarginable assets in a consistent way.

        Args:
        -----
            short (bool):
                A boolean indicating whether the initial margin is for a
                short position. By default False. If asset is
                nonmarginable returns 1.0.
        
        Returns:
        --------
            float:
                A float representing the initial margin of the asset.

        TODO:
        -----
            Investigate why Alpaca API does not follow FINRA 150% margin
            requirement for short positions. This still works since
            Alpaca requires 50% initial margin for short positions.
            Reducing this to 50% can unlock more leverage for short
            positions.
        """
        if not self.marginable:
            return 1
        elif not short:
            return 0.5
        elif short:
            return 1.5

    def get_maintenance_margin(self,
                               price: float,
                               short: bool = False) -> float | None:
        """
        A float representing the maintenance margin of the asset. This
        means that maintenace_margin * position_value should be
        available in marginable equity. Maintenance margin is cumulative
        for all assets and needs to be satisfied at all times. Alpaca
        API in reality enforces this at the end of day or when it is
        violated by a greate amount intraday. We enforce this at all
        times in a conservative manner. Maintenance margin for
        nonmarginable assets is 0. Since nonmarginable assets cannot be
        margined this is an abuse of terminalogy to provide a convenient
        interface for working with marginable and onnmarginable assets.
        
        Default maintenance marigin is the maintenance margin that
        Alpaca API reports by default. maintenance margin attribute is
        the value received from Alpaca API, however it is subject to
        change given price change and position change. Thus taking max
        of default maintenace margin and maintenance margin attribute
        ensures that the most conservative value is used for calculating
        the maintenance margin. Because Alpaca API checks both initial
        and maintenance margin at the end of day, we set the maintenance
        margin to be the maximum of the two. This is not a common
        behavior since typically initial margin is used for opening
        positions and maintenance margin is used for maintaining
        positions. However Alpaca API enforces both at end of day. In
        the end maximum of default margin, initial margin, and
        maintenance margin is used for final calculation of the
        maintenance margin.

        Args:
        -----
            price (float):
                A float representing the price of the asset.
            short (bool):
                A boolean indicating whether the maintenance margin is
                for a short position. By default False. 
        
        Returns:
        --------
            float:
                A float representing the maintenance margin of the
                asset.
        """

        def default_maintenance_margin(price: float,
                                       short: bool = False) -> float:
            """
            Link: https://alpaca.markets/docs/introduction/. The
            maintenance margin is by default calculated based on the
            following table:

            | Pos | Cond      | Margin Req        |
            | --- | --------- | ----------------- |
            | L   | SP < $2.5 | 100% of EOD MV    |
            | L   | SP >= $2.5| 30% of EOD MV     |
            | L   | 2x ETF    | 100% of EOD MV    |
            | L   | 3x ETF    | 100% of EOD MV    |
            | S   | SP < $5.0 | Max $2.50/S or 100% |
            | S   | SP >= $5.0| Max $5.00/S or 30% |

            where SP is the stock price and ETF is the exchange traded
            fund. L and S are long and short positions respectively. EOD
            MV is the end of day market value of the position.

            TODO: Add support for 2x and 3x ETFs. Currently there is no
            way in API to distinguish between ETFs and stocks. This
            means that 2x and 3x ETFs are treated as stocks.

            Args:
            -----
                price (float):
                    A float representing the price of the asset.
                short (bool):
                    A boolean indicating whether the maintenance margin
                    is for a short position. By default False.
            """

            if not self.marginable:
                return 0

            elif not short:
                if price >= 2.50:
                    return 0.3
                else:
                    return 1.0

            elif short:
                if price < 5.00:
                    return max(2.5 / price, 1.0)
                elif price >= 5.00:
                    return max(5.0 / price, 0.3)
                
        required_margin = max(self.maintenance_margin,
                              default_maintenance_margin(price, short),
                              self.get_initial_margin(short))
        return required_margin



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


    def _validate_resolution(self, resolution: Resolution, schedule: pd.DataFrame):
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

        daily_durations = schedule['end'] - schedule['start']
        rows_per_day = (daily_durations /
                        resolution.pandas_timedelta).values
        if not all(value.is_integer() for value in rows_per_day):
            raise ValueError('Incompatible resolution for given date range. '
                             'Daily durations is not divisible by resolution.')
        
        if resolution not in ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS:
            raise ValueError(
                f'Accepted resolutions: {ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS}.')

        return None


    def download_dataset(
        self,
        dataset_type: AlpacaDataSource.DatasetType,
        asset_type: AssetType,
        symbols: List[AlpacaAsset],
        timeframe: str,
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

        timeframe = resolution_to_timeframe(timeframe)

        downloader, request = self.data_client.get_downloader_and_request(
            dataset_type=dataset_type,
            asset_type=asset_type)

        data = downloader(request(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
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
        
        self._validate_resolution(resolution=resolution, schedule = schedule)
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
                timeframe=resolution,
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
                    dataset=group, open=start,
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
    process there can be missing rows in the data, due to trading halts
    or other market anomalies. In this case forward filling is used to
    indicate features has not changed over time when no trade occurs.

    Example:
    ----------
    for NYSE stocks, the market opens at 9:30 AM and closes at 4:00 PM.
    If resolution = 1Min, then the data is sampled every minute. This
    means that the data is sampled at 9:30 AM, 9:31 AM, 9:32 AM, ...,
    3:59 PM. Dataset is then reindexed such that rows corresponding to
    each interval are matched. If for example there is no trade at 9:31
    AM, then the data will be missing for that interval. In this case
    forward filling is used to indicate that the features has not
    changed over time when no trade occurs. If data at first interval is
    missing, then forward filling won't work, since there is no data to
    forward fill from. In this case backward filling is used to fill the
    missing value with closes non-missing value row. If after
    forward/backward filling there is still missing data, then the
    entire dataset is empty, namely there is no data for the given time
    range.
    """
    def __init__(self):

        self.processing_statistics = RunningStatistics()

    def reindex_and_forward_fill(
            self,
            dataset: pd.DataFrame,
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

        processed = dataset.reindex(index)

        non_nan_count = processed.notna().sum().sum()
        total_count = processed.size
        density = non_nan_count/total_count
        self.processing_statistics.update(density)

        if processed.isna().any().any():
            processed = processed.ffill()

        if processed.isna().any().any():
            processed = processed.bfill()

        return processed
