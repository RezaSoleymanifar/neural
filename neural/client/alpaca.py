from typing import Optional, Dict, List, Callable, Tuple

import numpy as np
import pandas as pd

from alpaca.trading.enums import (AccountStatus, AssetExchange, AssetClass,
                                  AssetStatus)
from alpaca.common.rest import RESTClient
from alpaca.common.websocket import BaseStream
from alpaca.data.historical import (StockHistoricalDataClient,
                                    CryptoHistoricalDataClient)
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.trading import TradingClient, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order, TradeAccount, Position
from alpaca.trading.models import Asset as AlpacaAsset
from alpaca.data.requests import BaseTimeseriesDataRequest

from alpaca.data.requests import (CryptoBarsRequest, CryptoTradesRequest,
                                  StockBarsRequest, StockQuotesRequest,
                                  StockTradesRequest)

from neural.common.log import logger
from neural.common.constants import API_KEY, API_SECRET
from neural.client.base import (AbstractClient, AbstractTradeClient,
                                AbstractDataClient)
from neural.data.base import AlpacaDataSource, AlpacaAsset
from neural.data.enums import AssetType
from neural.utils.misc import objects_to_dataframe


class AlpacaClient(AbstractClient):
    """
    Option 1: Instantiate an instance of the AlpacaClient class with
    your API key and secret.

    >>> client = AlpacaClient(key=..., secret=...)
    >>> assets = client.assets()

    Option 2: Instantiate an instance of the AlpacaClient by passing
    values to constants.

    >>> from neural.common.constants import API_KEY, API_SECRET
    >>> API_KEY = ...
    >>> API_SECRET = ...
    >>> client = AlpacaClient()

    Option 3: Instantiate an instance of the AlpacaClient class with
    environment variables.

    # Set the environment variables for API key and secret # on
    Unix-like operating systems (Linux, macOS, etc.): BASH: export
    API_KEY = <your_api_key> BASH: export API_SECRET = <your_secret_key>

    # Instantiate an instance of the AlpacaClient class >>> from
    neural.connect.alpaca import AlpacaClient >>> client =
    AlpacaClient()
    """

    def __init__(self,
                 key: Optional[str] = None,
                 secret: Optional[str] = None,
                 paper: bool = False) -> None:

        self.key = key if key is not None else API_KEY
        self.secret = secret if secret is not None else API_SECRET
        self.paper = paper

        self._clients = None
        self._account = None
        self._assets = None
        self._symbols = None
        self._asset_classes = None
        self._exchanges = None

        super().__init__()

        return None

    def connect(self):
        self._validate_credentials()
        self._clients = self._get_clients()
        self._account = self._get_account()

        return None

    def _validate_credentials(self) -> bool:

        if self.key is None or self.secret is None:
            raise ValueError(
                'API key and secret are required to connect to Alpaca API.')

        return None

    def _get_clients(self) -> RESTClient:

        # crypto does not need key, and secret but will be faster if
        # provided
        clients = dict()

        clients['crypto'] = CryptoHistoricalDataClient(api_key=self.key,
                                                       secret_key=self.secret)
        clients['stocks'] = StockHistoricalDataClient(api_key=self.key,
                                                      secret_key=self.secret)
        clients['trade'] = TradingClient(api_key=self.key,
                                         secret_key=self.secret,
                                         paper=self.paper)

        return clients

    def _get_account(self) -> TradeAccount:

        try:
            account = self.clients['trading'].get_account()
            if not self.account.status == AccountStatus.ACTIVE:

                logger.warning(f'Account Status: {self.account.status}')

        except Exception as e:
            logger.exception(f'Account setup failed: {e}')

        return account

    @property
    def clients(self) -> Dict[str:RESTClient]:

        return self._clients

    @property
    def account(self) -> TradeAccount:

        return self._account

    @property
    def assets(self) -> pd.DataFrame:
        """
        Returns a dataframe of all assets available on Alpaca API.
        """

        if self._assets is None:
            self._assets = self.clients['trade'].get_all_assets()

        assets_dataframe = objects_to_dataframe(self._assets)

        return assets_dataframe

    @property
    def symbols(self) -> Dict[str, AlpacaAsset]:
        """
        Returns a dictionary of all symbols available on Alpaca API. The
        corresponding values are the Asset objects.
        """
        if self._symbols is None:
            self._symbols = {asset.symbol: asset for asset in self.assets}

        return self._symbols

    @property
    def asset_classes(self) -> List[str]:

        if self._asset_classes is None:
            self._asset_classes = [item for item in AssetClass]

        asset_classes = [item.value for item in self._asset_classes]
        return asset_classes

    @property
    def exchanges(self) -> List[str]:

        if self._exchanges is None:
            self._exchanges = [item for item in AssetExchange]

        exchanges = [item.value for item in self._exchanges]
        return exchanges


class AlpacaDataClient(AlpacaClient, AbstractDataClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def data_source(self):

        return AlpacaDataSource

    @staticmethod
    def safe_method_call(object, method_name):

        if hasattr(object, method_name):
            return getattr(object, method_name)
        else:
            raise AttributeError(f"Client does not have method '{method_name}'")

    def get_downloader_and_request(
            self,
            dataset_type: AlpacaDataSource.DatasetType,
            asset_type=AssetType) -> Tuple[Callable, BaseTimeseriesDataRequest]:

        client_map = {
            AssetType.STOCK: self.clients['stocks'],
            AssetType.CRYPTO: self.clients['crypto']
        }

        client = client_map[asset_type]

        downloader_request_map = {
            AlpacaDataSource.DatasetType.TRADE: {
                AssetType.STOCK: ('get_stock_bars', StockBarsRequest),
                AssetType.CRYPTO: ('get_crypto_bars', CryptoBarsRequest)
            },
            AlpacaDataSource.DatasetType.QUOTE: {
                AssetType.STOCK: ('get_stock_quotes', StockQuotesRequest),
                AssetType.CRYPTO: ('get_crypto_quotes', CryptoQuotesRequest)
            },
            AlpacaDataSource.DatasetType.TRADE: {
                AssetType.STOCK: ('get_stock_trades', StockTradesRequest),
                AssetType.CRYPTO: ('get_crypto_trades', CryptoTradesRequest)
            }
        }

        downloader_method_name, request = downloader_request_map[dataset_type][
            asset_type]
        downloader = AlpacaDataClient.safe_method_call(
            object=client, method_name=downloader_method_name)

        return downloader, request

    def get_streamer(
        self,

        # callable take an async handler that receiVes the data and a
        # list of symbols async def handler(data): print(data)
        stream_type: AlpacaDataSource.StreamType,
        asset_type: AssetType,
    ) -> Callable:

        stream_map = {
            AlpacaDataSource.StreamType.BAR: {
                AssetType.STOCK: ('subscribe_bars', StockDataStream),
                AssetType.CRYPTO: ('subscribe_bars', CryptoDataStream)
            },
            AlpacaDataSource.StreamType.QUOTE: {
                AssetType.STOCK: ('subscribe_quotes', StockDataStream),
                AssetType.CRYPTO: ('subscribe_quotes', CryptoDataStream)
            },
            AlpacaDataSource.StreamType.TRADE: {
                AssetType.STOCK: ('subscribe_trades', StockDataStream),
                AssetType.CRYPTO: ('subscribe_trades', CryptoDataStream)
            }
        }

        stream_method_name, stream = stream_map[stream_type][asset_type]

        streamer = AlpacaDataClient.safe_method_call(
            object=stream, method_name=stream_method_name)

        return streamer

    def _get_assets(self, symbols: List[str]):

        asset_type_map = {
            AssetClass.US_EQUITY: AssetType.STOCK,
            AssetClass.CRYPTO: AssetType.CRYPTO
        }

        assets = list()

        asset_types = set(asset_type_map[self.symbols[symbol].asset_class]
                          for symbol in symbols)

        # checks if symbols have the same asset class
        if len(asset_types) != 1:
            raise ValueError(f'Non-homogenous asset types: {asset_types}.')

        for symbol in symbols:

            alpaca_asset = self.client.symbols[symbol]

            if alpaca_asset is None:
                raise ValueError(f'Symbol {symbol} is not a known symbol.')

            if not alpaca_asset.tradable:
                logger.warning(f'Symbol {symbol} is not a tradable symbol.')

            if alpaca_asset.status != AssetStatus.ACTIVE:
                logger.warning(f'Symbol {symbol} is not an active symbol.')

            if not alpaca_asset.fractionable:
                logger.warning(f'Symbol {symbol} is not a fractionable symbol.')

            if not alpaca_asset.easy_to_borrow:
                logger.warning(f'Symbol {symbol} is not easy to borrow (ETB).')

            asset_type = asset_type_map[alpaca_asset.asset_class]

            assets.append(
                AlpacaAsset(symbol=symbol,
                            asset_type=asset_type,
                            marginable=alpaca_asset.marginable,
                            fractionable=alpaca_asset.fractionable,
                            shortable=alpaca_asset.shortable,
                            required_margin=alpaca_asset.initial_margin,
                            maintenance_margin=alpaca_asset.maintenance_margin))

        return assets


class AlpacaTradeClient(AlpacaClient, AbstractTradeClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._cash = None
        self._asset_quantities = None
        self._net_worth = None
        self._longs = None
        self._shorts = None

    @property
    def cash(self):
        """
        The current amount of cash available to the trader.

        Returns:
        ---------
            float: The current amount of cash available to the trader.
        """

        self._cash = self.client.account.cash

        return self._cash

    @property
    def asset_quantities(self, assets: List[AlpacaAsset]) -> np.ndarray[float]:
        """
        Returns a dictionary of symbols and asset quantities for the
        trader's current positions.

        Returns:
            dict: A dictionary mapping symbols to asset quantities.
        """

        asset_quantities = dict()

        positions_dataframe = self.get_positions_dataframe()
        symbols = positions_dataframe['symbol'].unique()

        for symbol in symbols:
            row = positions_dataframe.loc[positions_dataframe['symbol'] ==
                                          symbol].iloc[0]
            quantity = row['qty'] if row['side'] == 'long' else -1 * row['qty']

            asset_quantities[symbol] = quantity

        return asset_quantities

    @property
    def net_worth(self) -> float:
        """
        The current net worth of the trader.

        Returns:
            float: The current net worth of the trader.
        """

        self._net_worth = self.client.account.portfolio_value

        return self._net_worth

    @property
    def longs(self) -> float:
        """
        The current long positions held by the trader.

        Returns:
            float: The current long positions held by the trader.
        """

        self._longs = self.client.account.long_market_value

        return self._longs

    @property
    def shorts(self) -> float:
        """
        The total value of all short positions held by the trader.

        Returns:
            float: The total value of all short positions held by the
            trader.
        """

        self._shorts = self.client.account.short_market_value

        return self._shorts

    def get_positions_dataframe(self) -> pd.DataFrame:
        """
        Get all current positions in the account.

        :return: A DataFrame containing the positions.
        """

        self._positions = self.clients['trading'].get_all_positions()

        positions_dataframe = objects_to_dataframe(self._positions)

        return positions_dataframe

    def check_connection(self):
        """
        Check the connection to the Alpaca API.

        :return: True if the account status is active, False otherwise.
        """

        status = True if self.account.status == AccountStatus.ACTIVE else False
        return status

    def place_order(
        self,
        symbol: str,
        quantity: Optional[float] = None,
        notional: Optional[float] = None,
        time_in_force: str = 'fok',
    ) -> Order:

        # this is a market order. Other order types should be
        # implemented by user. time in force options: Day order = "day"
        # Good 'til cancelled = "gtc" Opoening order = "opg" Closing
        # order = "cls" Immediate or cancel = "ioc" Fill or kill = "fok"

        if quantity is None and notional is None:
            raise ValueError('Either quantity or notional must be specified.')
        if quantity is not None and notional is not None:
            raise ValueError(
                'Only one of quantity or notional can be specified.')

        sign = np.sign(quantity) if quantity is not None else np.sign(notional)

        side = OrderSide.BUY if sign > 0 else OrderSide.SELL
        time_in_force = TimeInForce(time_in_force)

        quantity = abs(quantity) if quantity is not None else None
        notional = abs(notional) if notional is not None else None

        market_order_request = MarketOrderRequest(symbol=symbol,
                                                  qty=quantity,
                                                  notional=notional,
                                                  side=side,
                                                  time_in_force=time_in_force)

        market_order = self.clients['trade'].submit_order(
            order_data=market_order_request)

        return market_order
