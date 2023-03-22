from typing import List, Union
import pandas as pd
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL, NYSE_START, NYSE_END
from pytickersymbols import PyTickerSymbols
from alpaca_trade_api.rest import REST
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self) -> None:
        self.key = ALPACA_API_KEY
        self.secret = ALPACA_API_SECRET
        self.symbols = None
        self.api = None

    def connect(self):
        try:
            self.api = REST(self.key, self.secret, ALPACA_API_BASE_URL, api_version='v2')
            logger.info("Successfully connected to endpoint: {}".format(ALPACA_API_BASE_URL))
        except Exception as e:
            logger.error("Connection failed.: {}".format(e))

    def set_symbols(self, index: str) -> List[str]:
        stock_data = PyTickerSymbols()
        self.symbols = [stock['symbol'] for stock in list(stock_data.get_stocks_by_index(index))]

    @staticmethod
    def _bars(self, start: str, end: str , interval: str='1Min', symbols: Union[str, List[str]] = None) -> pd.DataFrame:
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

    def _add_indicators(self, data: np.array) -> np.ndarray:
        pass

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        pass

    def _trades_et(self, symbol):
        #checks if symbol trades at Eastern Time.
        pass

    def prices(self) -> np.ndarray:
        pass

    def save_scaler(self):
        pass

    def featurize(self, add_indicators: bool = False, add_quotes: bool = False, add_trades: bool = False)