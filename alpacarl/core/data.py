from typing import List, Union
import pandas as pd
import numpy as np
import os
import exchange_calendars
from pytickersymbols import PyTickerSymbols
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET
from alpaca_trade_api.rest import REST
from alpacarl.meta import log
from tqdm import tqdm
from typing import Generator, Union, Any
import inspect
from alpacarl.meta import log
from more_itertools import peekable


class DataHandler:
    def __init__(self) -> None:
        self.key = ALPACA_API_KEY
        self.secret = ALPACA_API_SECRET
        self._symbols = None
        self.api = None

    def connect(self, endpoint: str) -> None:
        self.api = REST(self.key, self.secret, endpoint, api_version='v2')
        # check API connection
        try:
            account = self.api.get_account()
            if account.status == 'ACTIVE':
                log.logger.info("Connection successful: {}".format(endpoint))
        except Exception as e:
            log.logger.exception("Error connecting to API: {}".format(str(e)))
        return None
    
    @property
    def symbols(self) -> List[str]:
        return self._symbols
    
    @symbols.setter
    def symbols(self, identifier: Union[str, List[str]]) -> List[str]:
        # identifier is index or list of symbols
        stocks_info = PyTickerSymbols()
        if isinstance(identifier, str):
            indices = [index.lower() for index in stocks_info.get_all_indices()]
            if identifier.lower() in indices:
                self._symbols = [stock['symbol'] for stock in list(stocks_info.get_stocks_by_index(identifier))]
            else:
                log.logger.exception("Supported indices: {}".format(indices))
                raise ValueError(f"{identifier} is not found in indices.")
        else:
            self._symbols = identifier

    @staticmethod
    def _preprocess(data, date, interval):
        # API returns no row for intervals with have no price change
        # forward filling will recover missing data
        start = pd.Timestamp(date, tz='America/New_York') + pd.Timedelta('9:30:00')
        end = pd.Timestamp(date, tz='America/New_York') + pd.Timedelta('15:59:00')
        index = pd.date_range(start=start, end=end, freq=interval)
        # creates rows for missing intervals
        resampled = data.reindex(index, method='ffill')
        if resampled.isna().all().all():
            log.logger.exception('Data does not have entries in NYSE market hours.')
            raise ValueError
        # backward fills if first row is nan
        if resampled.isna().any().any():
            resampled = resampled.bfill()
        return resampled

    def download(self, start: str, end: str, interval: str, symbols = None, dir: str = None) -> Union[None, pd.DataFrame]:
        symbols = symbols if symbols else self.symbols
        if dir is not None:
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                if os.path.exists(os.path.join(dir, 'data.csv')):
                    header = False
                else:
                    header = True

        nyse = exchange_calendars.get_calendar("NYSE")
        working_days = nyse.sessions_in_range(start, end).strftime('%Y-%m-%d')
        progress_bar = tqdm(total=len(working_days), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}')
        
        if dir is None:
            data = list()
        log.logger.info(f"Downloading data for {len(self.symbols)} symbols // frequency: {interval} //"\
                         f" {len(working_days)} working days from {start} to {end}")
        for day in working_days:
            # start and end are in UTC. ET is -04:00 from March to November and -05:00 otherwise.
            # We pad start by one hour to account for daylight saving time.
            # after tz conversion from Nov. to March, 8:30-9:30 is extra and from March to Nov. 16:00-17:00 is extra.
            # padded 1 hour is automatically dropped in resampling
            start = f'{day}T8:30:00-05:00'
            end = f'{day}T16:00:00-05:00'
            bars = self.api.get_bars(self.symbols, interval, start, end, adjustment='raw', limit=None).df
            bars = bars.tz_convert('America/New_York')
            features = pd.concat([DataHandler._preprocess(group[1], day, interval) for group in bars.groupby('symbol')], axis = 1)
            features = features.select_dtypes(include=np.number)
            if dir is not None:
                features.to_csv(os.path.join(dir, 'data.csv'), index=True, mode='a', header = header)
                header = False
            else:
                data.append(features)
            progress_bar.update(1)
        progress_bar.close()
        return pd.concat(data) if dir is None else None

def row_counter(dir = None):
    # counts number of rows in large data file
    # useful for getting number of steps in env without loading entire file.
    count = -1 # skipping header row
    # create pointer to file
    with open(dir) as file:
        for line in file:
            count += 1
    return count

def row_generator(dir: str = None, chunksize: str = None) -> Generator[Any, None, None]:
    # convenience function to iterate through rows of large CSV file
    # Create an iterator that generates chunks of the CSV file
    chunk_iterator = pd.read_csv(dir, chunksize=chunksize)
    idx = -1
    # Loop over the chunks and yield each row from the current chunk
    for chunk in chunk_iterator:
        for _, row in chunk.iterrows():
            idx += 1
            yield idx, row

class PeekableDataWrapper(DataWrapper):
    # sublcass of data wrapper with peek funcionality
    def __init__(self, data: Union[pd.DataFrame, Generator]):
        super().__init__(data)
        self.data = peekable(DataWrapper(data))
    def __iter__(self):
        return self.data.next()
    def __len__(self):
        return super().__len__()
    def peek(self):
        return self.data.peek()

class DataWrapper:
    # convenience class to provide a common generator-like interface for row_generator and pd.DataFrame objects
    def __init__(self, data: Union[pd.DataFrame, Generator]):
        if not isinstance(self.data, pd.DataFrame) and not inspect.isgenerator(self.data):
            log.logger.error('Can only wrap pd.DataFrame or a generator objects.')
            raise ValueError
        
        self.data = peekable(data)
        self.len_ = None

    def __iter__(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.iterrows()
        else:
            return self.data.next()

    def __len__(self):
        if isinstance(self.data, pd.DataFrame):
            self.len_ = len(self.data)
        elif inspect.isgenerator(self.data):
            sig = inspect.signature(self.data)
            if 'dir' in sig.parameters:
                dir = sig.bind_partial().arguments['dir']
                self.len_ = row_counter(dir)
        return self.len_
