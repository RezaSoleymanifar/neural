from typing import List, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL, NYSE_START, NYSE_END
from pytickersymbols import PyTickerSymbols
from alpaca_trade_api.rest import REST
from alpacarl.meta import log

class DataHandler:
    def __init__(self) -> None:
        self.key = 'a'
        self.secret = ALPACA_API_SECRET
        self._symbols = None
        self.api = None

    def connect(self) -> None:
        self.api = REST(self.key, self.secret, ALPACA_API_BASE_URL, api_version='v2')
        # check API connection
        try:
            account_info = self.api.get_account()
            if account_info.status_code == 200:
                log.logger.info("Connection successful: {}".format(ALPACA_API_BASE_URL))
            else:
                log.logger.warning("Connection returned status code: {}".format(account_info.status_code))
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
        indices = [index.lower() for index in stocks_info.get_all_indices()]
        if identifier.lower() in indices:
            self._symbols = [stock['symbol'] for stock in list(stocks_info.get_stocks_by_index(identifier))]
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
        # relies on daily data having prices before and after trade hours, so forward filling works.
        map = {'1Min':'1T', '5Min':'5T', '15Min': '15T', '1H': '1H'}
        resampled = df.resample(map[interval]).ffill()
        return resampled

    def featurize(self, start: str, end: str , interval: str='15Min', symbols: Union[str, List[str]] = None)\
          -> Tuple[pd.DataFrame, np.ndarray, StandardScaler]:
        symbols = symbols if symbols else self.symbols
        bars = api.get_bars(symbols, interval, start, end, adjustment='raw').df
        bars = df.tz_convert('America/New_York')
        # resample and forward fill missing intervals then add indicators
        df = pd.concat([self._indicators(self._resample(group[1], interval)) for group in bars.groupby('symbol')], axis = 1)
        # filter NYSE working days
        nyse = ec.get_calendar("NYSE")
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
        # standardize 
        scaler = StandardScaler()
        features = scaler.fit_transform(df)
        return prices, features, scaler
