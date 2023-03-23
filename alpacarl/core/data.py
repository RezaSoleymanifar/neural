from typing import List, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET, NYSE_START, NYSE_END
from pytickersymbols import PyTickerSymbols
import ta
import exchange_calendars
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
    def _indicators(df: pd.DataFrame) -> pd.DataFrame:
        # trend
        df['macd'] = ta.trend.macd(df['close'])
        df['cci'] = ta.trend.cci(high = df['high'], low= df['low'], close=df['close'])
        # momentum
        df['rsi'] = ta.momentum.rsi(close = df['close'])
        return df

    @staticmethod
    def _resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        # resamples and forward fills missing intervals.
        map = {'1Min':'1T', '5Min':'5T', '15Min': '15T', '1H': '1H'}
        resampled = df.resample(map[interval]).ffill()
        return resampled

    def featurize(self, start: str, end: str , interval: str='15Min', symbols: Union[str, List[str]] = None)\
          -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
        # get price data
        symbols = symbols if symbols else self.symbols
        log.logger.info("Downloading data for {} symbols...".format(len(symbols)))
        bars = self.api.get_bars(symbols, interval, start, end, adjustment='raw').df
        bars = bars.tz_convert('America/New_York')
        log.logger.info("Downloading {:,.2f} records successful.".format(len(bars)))


        # resample intervals then add indicators
        log.logger.info("Adding technical indicators and resampling...")
        df = pd.concat([self._indicators(self._resample(group[1], interval)) for group in bars.groupby('symbol')], axis = 1)

        # filter NYSE working days
        nyse = exchange_calendars.get_calendar("NYSE")
        working_days = nyse.sessions_in_range(start, end).normalize()
        df = df[df.index.floor('D').tz_localize(None).isin(working_days)]

        # filter NYSE trading hours (all symbols must trade in NYSE hours)
        df = df[(pd.Timestamp(NYSE_START).time() <= df.index.time) &\
         (df.index.time < pd.Timestamp(NYSE_END).time())]
        
        # drop null and non-number values
        df = df.select_dtypes(include=np.number)
        df.dropna(inplace=True)

        # save prices
        prices = df['close']
        prices.columns = symbols

        # standardize 
        scaler = StandardScaler()
        features = scaler.fit_transform(df)
        n, m = features.shape
        log.logger.info("Feature engineering successful. quantity:{:,.2f}, dimensionality:{:,.2f}".format(n, m))
        return prices, features, scaler
