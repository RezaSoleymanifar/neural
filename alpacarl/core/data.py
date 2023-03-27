from typing import List, Union
import pandas as pd
import numpy as np
import os
import exchange_calendars
from pytickersymbols import PyTickerSymbols
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET
from alpaca_trade_api.rest import REST
from alpacarl.meta import log


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
    def preprocess(df, date, interval):
        # API returns no row for intervals with have no price change
        # forward filling will recover missing data
        start = pd.Timestamp(date, tz='America/New_York') + pd.Timedelta('9:30:00')
        end = pd.Timestamp(date, tz='America/New_York') + pd.Timedelta('15:59:00')
        index = pd.date_range(start=start, end=end, freq=interval)
        # creates rows for missing intervals
        resampled = df.reindex(index, method='ffill')
        if resampled.isna().all().all():
            log.logger.exception('Data does not have entries in NYSE market hours.')
            raise ValueError
        # backward fills if first row is nan
        if resampled.isna().any().any():
            resampled = resampled.bfill()
        return resampled

    def download(self, start: str, end: str, interval: str, dir: str = None) -> Union[None, pd.DataFrame]:
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
        if dir is None:
            data = list()

        log.logger.info(f"Downloading {interval} data for {len(self.symbols)} symbols, from {start} to {end}")
        for day in working_days:
            # start and end are in UTC. ET is -04:00 from March to November and -05:00 otherwise.
            # We pad start by one hour to account for daylight saving time.
            # after tz conversion from Nov. to March, 8:30-9:30 is extra and from March to Nov. 16:00-17:00 is extra.
            # padded 1 hour is automatically dropped in resampling
            start = f'{day}T8:30:00-05:00'
            end = f'{day}T16:00:00-05:00'
            bars = api.get_bars(self.symbols, interval, start, end, adjustment='raw', limit=None).df
            bars = bars.tz_convert('America/New_York')
            features = pd.concat([preprocess(group[1], day, interval) for group in bars.groupby('symbol')], axis = 1)
            features = features.select_dtypes(include=np.number)
            if dir is not None:
                features.to_csv(os.path.join(dir, 'data.csv'), index=True, mode='a', header = header)
                header = False
            else:
                data.append(features)
        
        return pd.concat(data) if dir is None else None

    def update(self, dir):
        # updates dataset in directory
        pass