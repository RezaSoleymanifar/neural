import os
from datetime import datetime
from enum import Enum
import re
from typing import (List, Optional, Iterable, Type,
    Generator, Any, Dict)

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
    
    @staticmethod
    def _resample_and_ffil(open, close, interval):
        # resample and forward fills missing rows

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
    
    def create_dataset(self,
        start_date: str,
        end_date: str,
        resolution: str,
        symbols=None,
        dir: str = None,
        file_name: str = 'data.csv',
        ) -> str | pd.DataFrame:
        
        # converts to expected input formats
        start_date, end_date = to_datetime(start_date), to_datetime(end_date)
        resolution = to_timeframe(resolution)

        # check if symbols are valid names and of the same asset class type
        asset_class = self.validate_symbols(symbols)

        downloader, request = Downloader(meta_client = self, asset_class = asset_class)

        calendar = Calendar(Calendar.ALWAYS_OPEN
            ) if asset_class == 'CRYPTO' else Calendar(Calendar.NYSE)

        # if dir does not exist create it
        if dir is not None:
            if not os.path.exists(dir):
                os.makedirs(dir)

            # if file already exists just appends rows
            else:
                if os.path.exists(os.path.join(dir, file_name)):
                    header = False
                else:
                    header = True

        schedule = calendar.get_schedule(start_date=start_date, end_date=end_date)
        time_zone = calendar.get_time_zone()

        progress_bar = tqdm(total=len(len(schedule)),
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}')

        # use memory if dir is not provided
        if dir is None:
            data = list()

        log.logger.info(
            f"Downloading data for {len(self.symbols)} symbols | resolution: {resolution} |"
            f" {len(schedule)} working days from {start_date} to {end_date}"
        )

        for _, day in schedule:
            
            start = day['market_open']
            end = day['market_close']

            bars = downloader(request(symbol_or_symbols=symbol, timeframe=resolution, start=start, end=end))
            bars = bars.tz_convert('America/New_York')

            features = pd.concat([AlpacaMetaClient._resample_and_ffil(
                group[1], day, resolution) for group in bars.groupby('symbol')], axis=1)
            features = features.select_dtypes(include=np.number)
            if dir is not None:
                features.to_csv(os.path.join(dir, 'data.csv'),
                                index=True, mode='a', header=header)
                header = False
            else:
                data.append(features)
            progress_bar.update(1)
        progress_bar.close()
        return pd.concat(data) if dir is None else None

class Downloader():
    def __init__(self, meta_client: AlpacaMetaClient, asset_class: str) -> None:
        self.meta_client = meta_client
        self.asset_class = asset_class

    def get_downloader_and_request(self):

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
    
class Calendar:
    NYSE = 'NYSE'
    ALWAYS_OPEN = '24/7'

    def __init__(self, calendar_type = Type[Calendar]) -> None:
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
    # creates generator to iterate through large CSV file by loading chunks
    # into RAM
    def __init__(self, dir: str, chunk: Optional[str] = None) -> None:

        self.dir = dir
        self.chunk = chunk

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
            for _, row in chunk.iterrows():
                idx += 1
                yield idx, row


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
