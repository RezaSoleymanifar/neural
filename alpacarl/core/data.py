import os
from datetime import datetime
from enum import Enum
import re
from typing import (List, Optional, Iterable, Type,
    Generator, Any, Dict, Tuple)
from dataclasses import dataclass
from functools import reduce
import io


import pandas_market_calendars as market_calendars
import pandas as pd
from pandas import DataFrame
import numpy as np
import pickle
import tarfile
from more_itertools import peekable
import h5py

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading import TradingClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import AssetExchange, AssetClass

from alpacarl.meta.config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_ENDPOINT,
    ALPACA_API_ENDPOINT_PAPER)
from alpacarl.meta import log
from alpacarl.aux.tools import progress_bar

class DatasetType(Enum):
    BAR = 'BAR'
    QUOTE = 'QUOTE'
    TRADE = 'TRADE'
    ORDER_BOOK = 'ORDER_BOOK'

class ColumnType(Enum):
    ASSET_PRICE = 'ASSET_PRICE'
    OPEN = 'OPEN'
    HIGH = 'HIGH'
    LOW = 'LOW'
    CLOSE = 'CLOSE'


@dataclass
class DatasetMetadata:
    dataset_type: List[DatasetType]
    column_schema = Dict[ColumnType, Tuple[bool]]
    asset_class = AssetClass
    symbols = Tuple(str)
    start: datetime
    end: datetime
    resolution: str
    n_rows: int
    n_columns: int

    def join_column_schema(self, other):
        if set(self.column_schema.keys()) != set(other.column_schema.keys()):
            raise ValueError('Datasets do not have matching column_schema structure.')
        
        merged_schema = dict()

        for key in self.column_schema.keys():
            merged_schema[key] = self.column_schema[key] + other.column_schema[key]
        
        return merged_schema


    def __or__(self, other):
        # For automatic type checking and metadata generation when joining datasets
        # x | y automatically type checks and updates metadata of joined datasets.

        # checking compatibility
        if not isinstance(other, DatasetMetadata):
            raise ValueError('Only Dataset objects can be chained.')
        
        if other.dataset_type in self.dataset_type:
            raise ValueError('Duplicate dataset type is not allowed in horizontal chaining.')

        if self.asset_class != other.asset_class:
            raise ValueError('Datasets must have the same asset classes.')
        
        if self.symbols != other.symbols:
            raise ValueError('Datasets must have the same symbols.')
        
        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        if self.n_rows != other.n_rows:
            raise ValueError('Datasets must have the same number of rows.')

        if self.start != other.start:
            raise ValueError('Datasets do not have the same start dates.')

        if self.end != other.end:
            raise ValueError('Datasets do not have the same end dates.')

        dataset_type = self.dataset_type + other.dataset_type
        n_columns = self.n_columns + other.n_columns
        column_schema = self.join_column_schema(other)

        return DatasetMetadata(
            dataset_type=dataset_type,
            column_schema = column_schema,
            asset_class=self.asset_class,
            symbols=self.symbols,            
            start=self.start,
            end=self.end,
            resolution=self.resolution,
            n_rows=self.n_rows,
            n_columns=n_columns)


    def __and__(self, other):
        # For automatic type checking and metadata generation when appending to datasets
        # x + y automatically type checks and updates metadata of appended datasets.

        # checking compatibility
        if not isinstance(other, DatasetMetadata):
            raise ValueError('Only Dataset objects can be appended.')
        
        if self.dataset_type != other.dataset_type:
            raise ValueError(f'Dataset types {self.dataset_type} and {other.dataset_type} are mismatched.')
        
        if self.column_schema != other.column_schema:
            raise ValueError(f'Datasets must have identical column schema.')
        
        if self.asset_class != other.asset_class:
            raise ValueError('Datasets must have the same asset classes.')
        
        if self.symbols != other.symbols:
            raise ValueError('Datasets must have the same symbols.')

        if other.start <= self.end:
            raise ValueError(f'Cannot perform append. End time: {self.end}, and start time: {other.start} overlap.')       

        if abs(self.end.date() - other.start.date()).days != 1:
            raise ValueError(f'End date {self.end} and start date {other.start}  are not 1 day apart.')
        
        if self.resolution != other.resolution:
            raise ValueError(
                f'Dataset resolutions{self.resolution} and {other.resolution} are mismatched.')
        
        if self.n_columns != other.n_columns:
            raise ValueError('Dataset number of columns mismatch.')


        dataset_type = self.dataset_type + other.dataset_type
        n_columns = self.n_columns + other.n_columns
        column_schema = self.join_column_schema(other)

        return DatasetMetadata(
            dataset_type=dataset_type,
            start=self.start,
            end=self.end,
            symbols=self.symbols,
            resolution=self.resolution,
            n_rows=self.n_rows,
            n_columns=n_columns,
            column_schema = column_schema)
    
class CalendarType(Enum):
    NYSE = 'NYSE'
    ALWAYS_OPEN = '24/7'

class Calendar:

    def __init__(self, calendar_type=CalendarType) -> None:
        self.calendar_type = calendar_type
        self.calendar = None
    
    def get_calendar(self):

        calendar = market_calendars.get_calendar(self.calendar_type.value)

        return calendar

    # get core hours of calendar
    def get_schedule(self, start_date, end_date):

        self.calendar = self.get_calendar()
        schedule = self.calendar.schedule(start_date=start_date, end_date=end_date)

        return schedule
    
    def get_time_zone(self):

        if self.calendar_type == Calendar.ALWAYS_OPEN:
            time_zone =  'UTC'

        elif self.calendar_type == Calendar.NYSE:
            time_zone = 'America/New_York'

        return time_zone
    
class AlpacaMetaClient:
    def __init__(
        self,
        key: str,
        secret: str,
        sandbox=False
        ) -> None:

        # if sandbox = True tries connecting to paper account endpoint
        self.key = key if key else ALPACA_API_KEY
        self.secret = secret if secret else ALPACA_API_SECRET
        self.endpoint = ALPACA_API_ENDPOINT if not sandbox else ALPACA_API_ENDPOINT_PAPER
        self.clients = None
        self.account = None

        self._assets = None
        self._asset_classes = None
        self._exchanges = None
        
        return None

    def setup_clients_and_account(self) -> None:

        # crypto does not need key, and secret but will be faster if provided with credentials
        self.clients['crypto'] = CryptoHistoricalDataClient(
            self.key, self.secret)
        self.clients['stocks'] = StockHistoricalDataClient(
            self.key, self.secret)
        self.clients['trading'] = TradingClient(self.key, self.secret)

        try:
            self.account = self.clients['trading'].get_account()

            if self.account.status.val == "ACTIVE":
                log.logger.info(f'Clients and account setup is successful.')

        except Exception as e:
            log.logger.exception("Account setup failed: {}".format(str(e)))

        return None

    @property
    def assets(self):

        if self._assets:
            return self._assets
        
        assets_ = self.clients['trading'].get_all_assets()

        # keep tradable active assets only.
        assets_ = [
            asset for asset in assets_ if
            asset.status.value == "ACTIVE" and
            asset.tradable]
        
        self._assets = dicts_enum_to_df(assets_)
        return self._assets
    
    @property
    def exchanges(self):
        if self.exchanges:
            return self._exchanges
        exchanges_ = [item.value for item in AssetExchange]
        self._exchanges = exchanges_
        return self._exchanges
    
    @property
    def asset_classes(self):
        if self._asset_classes:
            return self._asset_classes
        asset_classes_ = [item.value for item in AssetClass]
        self._asset_classes = asset_classes_
        return self._asset_classes

    @property
    def positions(self):
        positions_ = self.clients['trading'].get_all_positions()
        self._positions = dicts_enum_to_df(positions_)
        return self._positions

    def set_credentials(self, key: str, secret: str) -> None:

        if not isinstance(key, str) or not isinstance(secret):
            raise ValueError(f'key and secret must of type {str}')
        
        self.key = key
        self.secret = secret

        return None

    def set_endpoint(self, endpoint: str) -> None:

        if not isinstance(endpoint, str):
            raise ValueError(f'endpoint must of type {str}')

        return None

    def validate_symbols(self, symbols: List[str]):

        valid_symbols = self.assets['symbol'].unique()

        # checks if symbols name is valid
        for symbol in symbols:
            if symbol not in valid_symbols:
                raise ValueError(f'Symbol {symbol} is not a supported symbol.')
        
        # checks if symbols have the same asset class
        symbol_classes = self.assets.loc[self.assets['symbol'].isin(symbols),
            'asset_class'].unique()

        if len(symbol_classes) != 1:
            raise ValueError('Symbols are not of the same asset class.')
        
        class_ = symbol_classes.pop()
        
        return class_
    
    def download_and_create_dataset(self,
        dataset_type: DatasetType,
        start_date: str,
        end_date: str,
        resolution: str,
        symbols: str | List[str],
        dir: Optional[str] = None,
        ) -> str | pd.DataFrame:
        
        # converts to expected input formats
        start_date, end_date = to_datetime(start_date), to_datetime(end_date)
        resolution = to_timeframe(resolution)

        # check if symbols are valid names and of the same asset class type
        asset_class = self.validate_symbols(symbols)

        downloader, request = DataDownloader(meta_client = self, asset_class = asset_class)

        if asset_class == 'USE_EQUITY':
            calendar = Calendar(calendar_type= Calendar.NYSE)
        
        elif asset_class == 'CRYPTO':
            calendar = Calendar(calendar_type = Calendar.ALWAYS_OPEN)

        # if dir does not exist create it
        # header flag allows download continuity of large datasets
        if dir is not None:
            if not os.path.exists(dir):
                os.makedirs(dir)
                header = True
            # if file already exists just appends rows
            else:
                if os.path.exists(os.path.join(dir, file_name)):
                    header = False
                else:
                    header = True

        schedule = calendar.get_schedule(start_date=start_date, end_date=end_date)
        time_zone = calendar.get_time_zone()

        # shows dataset download progress bar
        progress_bar_ = progress_bar(total=len(schedule))

        # use memory if dir is not provided
        if dir is None:
            data = list()

        log.logger.info(
            f"Downloading dataset for {len(self.symbols)} symbols | resolution: {resolution} |"
            f" {len(schedule)} working days from {start_date} to {end_date}"
        )

        # cache and download data day by day
        for _, day in schedule:
            
            market_open = day['market_open']
            market_close = day['market_close']

            bars = downloader(
                request(symbol_or_symbols=symbol, timeframe=resolution,
                start=market_open, end=market_close))
            
            bars = bars.tz_convert(time_zone)

            features = pd.concat([_resample_and_ffil(group[1], day, resolution) for group in bars.groupby('symbol')], axis=1)

            features = features.select_dtypes(include=np.number)
            if dir is not None:
                features.to_csv(os.path.join(dir, 'data.csv'),
                                index=True, mode='a', header=header)
                header = False
            else:
                data.append(features)

            progress_bar_.update(1)
        progress_bar_.close()

        return pd.concat(data) if dir is None else None

class DataDownloader():
    def __init__(self, meta_client: AlpacaMetaClient) -> None:

        self.meta_client = meta_client

        return None

    def get_downloader_and_request(self, dataset_type: DatasetType, asset_class = AssetClass):
        
        if dataset_type == DatasetType.BAR:
            # choose relevant client
            if asset_class == AssetClass.US_EQUITY:
                client = self.meta_client.clients['stocks']
                downloader = client.get_stocks_bars
                request = StockBarsRequest
                
            elif asset_class == AssetClass.CRYPTO:
                client = self.meta_client.clients['crypto']
                downloader = client.get_crypto_bars
                request = CryptoBarsRequest
                
            return downloader, request

class DataProcessor:
    def __init__(self) -> None:
        pass 

    def resample_and_ffil(data, open: datetime, close: datetime, interval: str):
        # resamples and forward fills missing rows in [open, close] range

        index = pd.date_range(start=open, end=close, freq=interval)

        # creates rows for missing intervals
        resampled = data.reindex(index, method='ffill')
        if resampled.isna().all().all():
            log.logger.exception(
                'Data does not have entries in NYSE market hours.')
            raise ValueError

        # backward fills if first row is nan
        if resampled.isna().any().any():
            resampled = resampled.bfill()

        # Prefix column names with symbol
        symbol = resampled['symbol'][0]
        resampled.columns = [f'{symbol}_{col}' for col in data.columns]
        return resampled

class DatasetIO:

    def write_to_hdf5(
        path: str | os.PathLike, 
        data_to_write: np.ndarray, 
        metadata: DatasetMetadata, 
        target_dataset_name: str):

        if os.path.exists(path):

            with h5py.File(path, 'w') as hdf5:

                if target_dataset_name not in hdf5:

                    # Create a fixed-size dataset with a predefined data type and dimensions
                    target_dataset = hdf5.create_dataset(
                        name = target_dataset_name, shape=data_to_write.shape, dtype=np.float32, chunks = True)
                    
                    serialized_metadata = pickle.dumps(metadata)
                    target_dataset.attrs['metadata'] = serialized_metadata

                else:

                    target_dataset_metadata, target_dataset = DatasetIO.load_from_hdf5(hdf5, target_dataset_name= target_dataset_name)

                    new_metadata = target_dataset_metadata + metadata
                    n_rows, n_columns = target_dataset_name.shape
                    n_rows_, n_columns_ = data_to_write.shape

                    if n_columns_ != n_columns:
                        raise ValueError('Mismatch in number of columns.')

                    new_n_rows, new_n_columns = n_rows + n_rows_, n_columns

                    target_dataset_name.resize((new_n_rows, new_n_columns))

                    # Append the new data to the dataset and update metadata
                    target_dataset_name[n_rows:new_n_rows, :] = data_to_write
                    target_dataset_name.attrs['metadata'] = new_metadata

        else:

            raise ValueError(f'Path {path} does not exist.')
        
        return None

    def extract_dataset(hdf5: h5py.File, target_dataset_name: str):

        target_dataset = hdf5[target_dataset_name]
        serialized_metadata = target_dataset.attrs['metadata']
        metadata = pickle.loads(serialized_metadata)

        return metadata, target_dataset            

    def load_from_hdf5(
        path: str | os.PathLike, 
        target_dataset_name: Optional[str] = None
        ) -> Tuple[DatasetMetadata, List[h5py.Dataset]]:

        if os.path.exists(path):

            with h5py.File(path, 'r') as hdf5:

                if target_dataset_name is None:
                    dataset_list = list()
                    metadata_list = list()
                    for dataset_name in hdf5:

                        metadata, dataset = DatasetIO.extract_dataset(
                            hdf5 = hdf5, target_dataset_name= dataset_name)
                        dataset_list.append(dataset)
                        metadata_list.append(metadata)
                        metadata = reduce(lambda x, y: x | y, metadata_list)

                    return metadata, dataset_list
                
                else:
                    metadata, dataset =  DatasetIO.extract_dataset(
                        hdf5 = hdf5, target_dataset_name=target_dataset_name)
                    return metadata, [dataset]
        else:

            raise ValueError(f'Path {path} does not exist.')


class RowGenerator():
    # to iteratively return info required for environments from a dataset.
    def __init__(
        self, 
        dataset_metadata: DatasetMetadata, 
        datasets: List[h5py.Dataset]
        start_index = None,
        end_index = None,
        batch_size: Optional[int] = None,
        ) -> None:

        self.dataset_metadata = dataset_metadata
        self.start_index = start_index
        self.end_index = end_index
        self.batch_size = batch_size


    def __iter__:
        for 
    # returns mutually exclusive generators each covering a continous section of dataset.
    def divide(self, n: int):


class PeekableDataWrapper:
    # A wrapper that gives peek and reset ability to generator like objects
    def __init__(self, data: DataFrame | RowGenerator):

        if not isinstance(data, pd.DataFrame) and not isinstance(
            data, RowGenerator):

            log.logger.error(
                'Can only wrap pd.DataFrame or a RowGenerator objects.')
            
            raise ValueError

        self.data = data
        self.generator = None
        self.reset()
        return None

    def reset(self):
        self.generator = peekable(self.data.get_generator())
        return None

    def peek(self):
        return self.generator.peek()

    def __iter__(self):
        yield from self.generator

    def __next__(self):
        return next(self.generator)

    def __len__(self):
        return len(self.data)
    
# converts dictionaries of enum objects into dataframe
def dicts_enum_to_df(
    info: Iterable[Dict[str, str]]
    ) -> DataFrame:
    
    for dict_ in info:
        for key, val in dict_.items():
            dict_[key] = val.value if isinstance(val, Enum) else val

    df = DataFrame(info)
    return df

def to_datetime(date: str):
    try:
        date_format = "%Y-%m-%d"
        dt = datetime.strptime(date, date_format)
    except:
        ValueError('Invalid date. Valid examples: 2022-03-20, 2015-01-01')
    return dt

def to_timeframe(time_frame: str):

    match = re.search(r'(\d+)(\w+)', time_frame)

    if match:

        amount = int(match.group(1))
        unit = match.group(2)

        map = {
            'Min': TimeFrameUnit.Minute,
            'Hour': TimeFrameUnit.Hour,
            'Day': TimeFrameUnit.Day,
            'Week': TimeFrameUnit.Week,
            'Month': TimeFrameUnit.Month}

        return TimeFrame(amount, map[unit])
    else:
        raise ValueError(
            "Invalid timeframe. Valid examples: 59Min, 23Hour, 1Day, 1Week, 12Month")
    
def combine_column_type_masks(*args: Dict[]):
    # aggregates column type masks suitable for filtering concatenated rows. 