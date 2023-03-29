from typing import List, Union, Optional
import pandas as pd
import numpy as np
import os
import pandas_market_calendars as market_calendars
from pytickersymbols import PyTickerSymbols
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_PAPER_URL
from alpaca_trade_api.rest import REST
from alpacarl.meta import log
from tqdm import tqdm
from typing import Generator, Union, Any
from alpacarl.meta import log
from more_itertools import peekable


class DataFetcher:
    def __init__(self) -> None:
        self.key = ALPACA_API_KEY
        self.secret = ALPACA_API_SECRET
        self._symbols = None
        self.api = None
        stocks_info = PyTickerSymbols()
        self.markets = market_calendars.get_calendar_names()
        self.get_all_assets
        self.get_all_indices = stocks_info.get_all_indices
        self.get_symbols_by_index = stocks_info.get_stocks_by_index
        self.get_market_schedule = lambda market: market_calendars.get_calendar(market).schedule
        self.get_market_timezone = lambda market: market_calendars.get_calendar(market).tz
        
        
        # self.get_all_industries = pass
        self.get_all_markets = market_calendars.get_calendar_names
        # self.get_symbols_by_market = pass
        self.get_symbols_by_industry = lambda industry: [stock['symbol'] for stock in stocks_info.get_stocks_by_industry(industry)]

        self.assets = self.fetch_assets()   

    def get_all_equity(self):
        pass

    def get_all_crypto(self):
        pass

    def get_all_industries(self):
        pass

    def get_all_exchanges(self):
        pass

    def get_exchange_schedule(self):
        pass

    def get_exchange_timezone(self):
        pass

    def get_symbols_by_exchange(self):
        pass

    def get_symbols_by_index(self):
        pass

    def get_symbols_by_industry(self):
        pass

    def validate_symbols(self):
        # sanity check for symbols before creating dataset
        # checks symbols for: 1) being a valid asset 2) being active 3) having same time trades
        pass

    def get_
    def get_exchange_timezone
    def fetch_assets(self):
        assets = self.api.list_assets()
        self.assets = pd.DataFrame([list(asset.__dict__.values())[0] for asset in assets])

    def get_symbol_start_date(self, symbol:str):
        pass

    def get_all_assets() -> pd.DataFrame:
        

    def connect_to_endpoint(self, endpoint: str, key: Union[str, None] = None, secret: Union[str, None] = None) -> None:
        key = key if key is not None else self.key
        secret = secret if secret is not None else self.secret
        self.api = REST(key, secret, endpoint, api_version='v2')
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
        if isinstance(identifier, str):
            indices = [index.lower() for index in self.get_all_indices()]
            if identifier.lower() in indices:
                self._symbols = [stock['symbol'] for stock in list(self.get_symbols_by_index(identifier))]
            else:
                log.logger.exception("Supported indices: {}".format(indices))
                raise ValueError(f"{identifier} is not found in indices.")
        else:
            # sets symbols equal to provided list of symbols
            self._symbols = identifier
        print(len(self._symbols))

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
        
        # Prefix column names with symbol
        symbol = resampled['symbol'][0]
        resampled.columns = [f'{symbol}_{col}' for col in data.columns]
        return resampled

    def create_dataset(self, start_date: str, end_date: str, time_interval: str, market: str,\
                  symbols = None, dir: str = None, auto_clip = True) -> Union[str, pd.DataFrame]:
        symbols = symbols if symbols else self.symbols
        if dir is not None:
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                if os.path.exists(os.path.join(dir, 'data.csv')):
                    header = False
                else:
                    header = True

        nyse = market_calendars.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        progress_bar = tqdm(total=len(len(schedule)), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}')
        
        if dir is None:
            data = list()
        log.logger.info(f"Downloading data for {len(self.symbols)} symbols // frequency: {time_interval} //"\
                         f" {len(schedule)} working days from {start_date} to {end_date}")
        for _, day in schedule:
            # start and end are in UTC. ET is -04:00 from March to November and -05:00 otherwise.
            # We pad start by one hour to account for daylight saving time.
            # after tz conversion from Nov. to March, 8:30-9:30 is extra and from March to Nov. 16:00-17:00 is extra.
            # padded 1 hour is automatically dropped in resampling
            start = day['market_open']
            end = day['market_close']
            bars = self.api.get_bars(symbols, time_interval, start, end, adjustment='raw').df
            bars = bars.tz_convert('America/New_York')
            features = pd.concat([DataFetcher._preprocess(group[1], day, time_interval) for group in bars.groupby('symbol')], axis = 1)
            features = features.select_dtypes(include=np.number)
            if dir is not None:
                features.to_csv(os.path.join(dir, 'data.csv'), index=True, mode='a', header = header)
                header = False
            else:
                data.append(features)
            progress_bar.update(1)
        progress_bar.close()
        return pd.concat(data) if dir is None else None

class RowGenerator():
    # creates generator to iterate through large CSV file by loading chunks into RAM
    def __init__(self, dir: str, chunk: Optional[str] = None) -> None:
        self.dir = dir
        self.chunk = chunk

    def __len__(self):
        # counts number of rows in RowGenerator object
        count = -1 # skipping header row
        # create pointer to file
        with open(self.dir) as file:
            for _ in file:
                count += 1
        return count

    def iterrows(self) -> Generator[Any, None, None]:
        # returns a generator object to iterate through rows similar to pd.DataFrame
        chunk_iterator = pd.read_csv(self.dir, chunksize=self.chunk)
        idx = -1
        # Loop over the chunks and yield each row from the current chunk
        for chunk in chunk_iterator:
            for _, row in chunk.iterrows():
                idx += 1
                yield idx, row

class PeekableDataWrapper:
    # A wrapper that gives peek and reset ability to generator like objects
    def __init__(self, data: Union[pd.DataFrame, RowGenerator]):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, RowGenerator):
            log.logger.error('Can only wrap pd.DataFrame or a RowGenerator object.')
            raise ValueError
        
        self.data = data
        self.generator = None
        self.reset()

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
