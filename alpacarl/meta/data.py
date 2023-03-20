from typing import List, Union
import pandas as pd
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL
from pytickersymbols import PyTickerSymbols
from alpaca_trade_api.rest import REST

class DataHandler:
    def __init__(self) -> None:
        self.key = ALPACA_API_KEY
        self.secret = ALPACA_API_SECRET
        self.symbols = None
        self.api = REST(self.key, self.secret, ALPACA_API_BASE_URL, api_version='v2')
        print('Successfully connected to Alpaca API.')


    def set_symbols(self, index: str) -> List[str]:
        stock_data = PyTickerSymbols()
        self.symbols = [stock['symbol'] for stock in list(stock_data.get_stocks_by_index(index))]

    @staticmethod
    def _prices(self, start: str, end: str , interval: str='1Min', symbols: Union[str, List[str]] = None) -> pd.DataFrame:
        symbols = symbols if symbols else self.symbols
        bars = self.api.get_bars(symbols, interval, start, end, adjustment='raw').df
        return bars

    @staticmethod
    def _trades(self) -> pd.DataFrame:
        pass
    
    @staticmethod
    def _quotes(self) -> pd.DataFrame:
        pass

    @staticmethod
    def _news(self) -> pd.DataFrame:
        pass

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
        try:
            for feature in feature_list:
                pass
        except:
            raise KeyError('Data incompatible with features.')