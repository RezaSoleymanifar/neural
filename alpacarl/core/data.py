import os
from datetime import datetime
from enum import Enum
import re
from typing import (List, Optional, Iterable, Type,
    Generator, Any, Dict, Tuple)
from dataclasses import dataclass

import pandas_market_calendars as market_calendars
import pandas as pd
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from more_itertools import peekable

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

@dataclass
class Dataset:
    path: str | List[str]
    dataset_type: DatasetType | DatasetChainType
    start: datetime
    end: datetime
    symbols: List[str] | List[List[str]]
    resolution: str
    n_rows: int
    n_columns: int | List[int]
    csv: str | List[str] = 'dataset.csv'

    def __or__(self, other):

        if isinstance(self.dataset_type, DatasetTypeChain):
            if self.dataset_type.chain_type == DatasetChainType.VERTICAL:
                raise ValueError(
                    'Cannot horizontally chain a vetically chained dataset.')

        if isinstance(other.dataset_type, DatasetTypeChain):
            raise TypeError(
                'Left-associative operator | : in x | y, y can only be DatasetType, not DatasetTypeChain.')

        # checking compatibility
        if not isinstance(other, Dataset):
            raise ValueError('Only Dataset objects can be added.')

        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        if self.n_rows != other.n_rows:
            raise ValueError('Datasets must have the same number of rows.')

        if self.start != other.start:
            raise ValueError('Datasets do not have the same start dates.')

        if self.end != other.end:
            raise ValueError('Datasets do not have the same end dates.')

        path = self.path + \
            other.path if isinstance(self.path, List) else [
                self.path, other.path]
        
        dataset_type = self.dataset_type + other.dataset_type

        symbols = self.symbols + \
            other.symbols if isinstance(self.symbols, List[List]) else [
                self.symbols, other.symbols]
        
        n_columns = self.n_columns + other.n_columns if isinstance(
            self.n_columns, List) else [self.n_columns, other.n_columns]
        
        csv = self.csv + \
            other.csv if isinstance(self.n_columns, List) else [
                self.csv, other.csv]

        return Dataset(
            path=path,
            dataset_type=dataset_type,
            start=self.start,
            end=self.end,
            symbols=symbols,
            resolution=self.resolution,
            n_rows=self.n_rows,
            n_columns=n_columns,
            csv=csv
            )

    def __and__(self, other):
        
        if isinstance(self.dataset_type, DatasetTypeChain):
            if self.dataset_type.chain_type == DatasetChainType.HORIZONTAL:
                raise ValueError('Cannot vertically chain a horizontally chained dataset.')
        
        if isinstance(other.dataset_type, DatasetTypeChain):
            raise TypeError('Left-associative operator + : in x + y, y can only be DatasetType, not DatasetTypeChain.')
                            
        # checking compatibility
        if not isinstance(other, Dataset):
            raise ValueError('Only Dataset objects can be added.')
        
        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')
        
        if self.symbols != other.symbols:
            raise ValueError('Vertical chaining requires datasets to have same symbols.')
        
        if other.start <= self.end:
            raise ValueError('Datasets cannot have overlapping time spans in vertical chaining.')
        
        if self.n_columns != other.n_columns:
            raise ValueError('Datasets must have the same number of rows.')
         
        
        path = self.path + other.path if isinstance(self.path, List) else [self.path, other.path]
        dataset_type = self.dataset_type + other.dataset_type
        end = other.end
        n_rows = self.n_rows + other.n_rows if isinstance(self.n_rows, List) else [self.n_rows, other.n_rows]

        return Dataset(
            path=paths,
            dataset_type=dataset_type,
            start=self.start,
            end=self.end,
            symbols=self.symbols,
            resolution=self.resolution,
            n_rows=n_rows,
            n_columns=self.n_columns,
            csv= csvs
        )
    
class DatasetChainType(Enum):
    HORIZONTAL = 'HORIZONTAL'
    VERTICAL = 'VERTICAL'


class DatasetTypeChain:
    def __init__(
            self,
            *dataset_types: DatasetType,
            chain_type: DatasetChainType
            ) -> None:
        self.dataset_types = dataset_types
        self.chain_type = chain_type
    
    def __add__(self, other: DatasetType | DatasetChainType):

        if isinstance(other.dataset_type, DatasetTypeChain):
            if self.chain_type != other.chain_type:
                raise ValueError(
                    'Cannot chain two incompatible chain types.')
        
        elif isinstance(self.dataset_types)
            
        # checking compatibility
        if not isinstance(other, Dataset):
            raise ValueError('Only Dataset objects can be added.')


    
class Calendar:

    NYSE = 'NYSE'
    ALWAYS_OPEN = '24/7'

    def __init__(self, calendar_type=Type[Calendar]) -> None:
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

class RowGenerator():
    # TODO: takes list of dirs and iterates and concates them on the fly.
    # creates generator to iterate through large CSV file by loading chunks
    # into RAM
    def __init__(
        self, 
        dir: str, 
        chunk: Optional[str] = None, 
        skiprows = ...,
        nrows = ...
        ) -> None:

        self.dir = dir
        self.chunk = chunk
        self.skiprows = skiprows
        self.nrows = nrows

    def __len__(self):
        # counts number of rows in RowGenerator object
        count = -1  # skipping header row
        # create pointer to file

        with open(self.dir) as file:
            for _ in file:
                count += 1
        return count

    def iterrows(self) -> Generator[Any, None, None]:

        # returns a generator object to iterate through rows similar to
        # pd.DataFrame
        chunk_iterator = pd.read_csv(self.dir, chunksize=self.chunk)
        idx = -1

        # Loop over the chunks and yield each row from the current chunk
        for chunk in chunk_iterator:
            yield from chunk.iterrows()

    # multiplies into RowGenrator instnaces each dedicated to one part of dataframe
    def multiply(self):
        pass

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
        self.generator = peekable(self.data.iterrows())
        return None

    def peek(self):
        return self.generator.peek()

    def __iter__(self):
        yield from self.generator

    def __next__(self):
        return next(self.generator)

    def __len__(self):
        return len(self.data)
    
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
        
        self._assets = _dicts_enum_to_df(assets_)
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
        self._positions = _dicts_enum_to_df(positions_)
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
    
    def create_dataset(self,
        dataset_type: DatasetType,
        start_date: str,
        end_date: str,
        resolution: str,
        symbols: str | List[str] = None,
        dir: str = None,
        ) -> str | pd.DataFrame:
        
        # converts to expected input formats
        start_date, end_date = to_datetime(start_date), to_datetime(end_date)
        resolution = to_timeframe(resolution)

        # check if symbols are valid names and of the same asset class type
        asset_class = self.validate_symbols(symbols)

        downloader, request = Downloader(meta_client = self, asset_class = asset_class)

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

class Downloader():
    def __init__(self, meta_client: AlpacaMetaClient, asset_class: str) -> None:
        self.meta_client = meta_client
        self.asset_class = asset_class

    def get_downloader_and_request(self, dataset_type: DatasetType):

        # choose relevant client
        if self.asset_class == 'US_EQUITY':
            client = self.meta_client.clients['stocks']
            downloader = client.get_stocks_bars
            request = StockBarsRequest
            

        elif self.asset_class == 'CRYPTO':
            client = self.meta_client.clients['crypto']
            downloader = client.get_crypto_bars
            request = CryptoBarsRequest
            
        return downloader, request
        
# converts dictionaries of enum objects into dataframe
def _dicts_enum_to_df(
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

def _resample_and_ffil(data, open: datetime, close: datetime, interval: str):
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
