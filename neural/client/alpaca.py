from typing import Optional, Dict, List, Callable, Tuple

import pandas as pd

from alpaca.trading.enums import AccountStatus, AssetExchange, AssetClass, AssetStatus
from alpaca.common.rest import RESTClient
from alpaca.common.websocket import BaseStream
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.trading import TradingClient, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Order, TradeAccount, Asset, Position
from alpaca.data.requests import BaseTimeseriesDataRequest

from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoQuotesRequest,
    CryptoTradesRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest
)

from neural.common.log import logger
from neural.common.constants import API_KEY, API_SECRET 
from neural.client.base import AbstractClient, AbstractTradeClient, AbstractDataClient
from neural.data.enums import AlpacaDataSource
from neural.data.base import AssetType
from neural.tools.misc import objects_to_df



class AlpacaClient(AbstractClient):

    """
    Option 1: Instantiate an instance of the AlpacaClient class with your API key and secret.

    >>> client = AlpacaClient(key=..., secret=...)
    >>> assets = client.assets()

    Option 2: Instantiate an instance of the AlpacaClient by passing values to constants.

    >>> from neural.common.constants import API_KEY, API_SECRET
    >>> API_KEY = ...
    >>> API_SECRET = ...
    >>> client = AlpacaClient()

    Option 3: Instantiate an instance of the AlpacaClient class with environment variables.

    # Set the environment variables for API key and secret
    # on Unix-like operating systems (Linux, macOS, etc.):
    BASH: export API_KEY = <your_api_key>
    BASH: export API_SECRET = <your_secret_key>

    # Instantiate an instance of the AlpacaClient class
    >>> from neural.connect.alpaca import AlpacaClient
    >>> client = AlpacaClient()
    """

    def __init__(
        self,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        paper: bool = False
        ) -> None:


        self.key = key if key is not None else API_KEY
        self.secret = secret if secret is not None else API_SECRET
        self.paper = paper

        self._validate_credentials()
        super().__init__()

        self._clients = None
        self._account = None
        self._assets = None
        self._symbols = None
        self._asset_classes = None
        self._exchanges = None

        return None


    def connect(self):
        # super class runs this method at contructor.
        self._clients = self._get_clients()
        self._account = self._get_account()

        return None


    def _validate_credentials(self) -> bool:

        if self.key is None or self.secret is None:
            raise ValueError(
                'API key and secret are required to connect to Alpaca API.')
        
        return None
    
    def _get_clients(self) -> RESTClient:

        # crypto does not need key, and secret but will be faster if provided
        clients = dict()

        clients['crypto'] = CryptoHistoricalDataClient(
            api_key=self.key, secret_key=self.secret)
        clients['stocks'] = StockHistoricalDataClient(
            api_key=self.key, secret_key=self.secret)
        clients['trade'] = TradingClient(
            api_key=self.key, secret_key=self.secret, paper=self.paper)
        
        return clients


    def _get_account(self) -> TradeAccount:

        try:
            self.account = self.clients['trading'].get_account()
            if not self.account.status == AccountStatus.ACTIVE:

                logger.warning(
                    f'Account Status: {self.account.status}')

        except Exception as e:
            logger.exception(
                f'Account setup failed: {e}')
            
        return None
    

    @property
    def clients(self) -> Dict[str: RESTClient]:


        return self._clients


    @property
    def account(self) -> TradeAccount:
        
        return self._account
            

    @property
    def assets(self) -> pd.DataFrame:

        """
        Get all assets available through the Alpaca API.

        :return: A DataFrame containing the assets.
        """

        if self._assets is None:
            self._assets = self.clients['trade'].get_all_assets()

        assets_dataframe =  objects_to_df(self._assets)
        
        return assets_dataframe



    @property
    def symbols(self) -> Dict[str, Asset]:

        """
        Get the symbols of assets fetched from the Alpaca API.

        Returns
        -------
        Dict
            A dictionary of symbols mapped to their respective assets.
        """

        if self._symbols is None:
            self._symbols = {asset.symbol: asset for asset in self.assets}

        return self._symbols


    @property
    def exchanges(self) -> List[str]:

        """
        Get a list of all supported exchanges.

        :return: A list of exchange names.
        """

        if self._exchanges is None:
            self._exchanges = [item for item in AssetExchange]

        exchanges =  [item.value for item in self._exchanges]
        return exchanges


    @property
    def asset_classes(self) -> List[str]:

        """
        Get a list of all supported asset classes.

        :return: A list of asset class names.
        """

        if self._asset_classes is None:
            self._asset_classes = [item for item in AssetClass]

        asset_classes =  [item.value for item in self._asset_classes]
        return asset_classes



class AlpacaDataClient(AlpacaClient, AbstractDataClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    
    @property
    def data_source(self):

        return AlpacaDataSource

    def get_downloader_and_request(
        self,
        dataset_type: AlpacaDataSource.DatasetType,
        asset_class=AssetType
        ) -> Tuple[Callable, BaseTimeseriesDataRequest]:

        """
        Returns the appropriate data downloader and request object based on the provided dataset type
        and asset class.

        Parameters:
        -----------
        dataset_type: DatasetType
            The type of dataset being downloaded, one of ['BAR', 'QUOTE', 'TRADE'].
        asset_class: AssetClass, optional
            The asset class being downloaded, defaults to `AssetClass.US_EQUITY`.

        Returns:
        --------
        Tuple[Any, Any]
            A tuple containing the appropriate downloader and request objects.
        """

        client_map = {
            AssetType.STOCK: self.clients['stocks'],
            AssetType.CRYPTO: self.clients['crypto']}

        client = client_map[asset_class]

        def safe_method_call(client, method_name):
            if hasattr(client, method_name):
                return getattr(client, method_name)
            else:
                raise AttributeError(
                    f"Client does not have method '{method_name}'")

        downloader_request_map = {
            AlpacaDataSource.DatasetType.TRADE: {
                AssetClass.US_EQUITY: ('get_stock_bars', StockBarsRequest),
                AssetClass.CRYPTO: ('get_crypto_bars', CryptoBarsRequest)},
            AlpacaDataSource.DatasetType.QUOTE: {
                AssetClass.US_EQUITY: ('get_stock_quotes', StockQuotesRequest),
                AssetClass.CRYPTO: ('get_crypto_quotes', CryptoQuotesRequest)},
            AlpacaDataSource.DatasetType.TRADE: {
                AssetClass.US_EQUITY: ('get_stock_trades', StockTradesRequest),
                AssetClass.CRYPTO: ('get_crypto_trades', CryptoTradesRequest)}}

        downloader_method_name, request = downloader_request_map[dataset_type][asset_class]
        downloader = safe_method_call(
            client=client, method_name=downloader_method_name)

        return downloader, request


    def get_streamer(self, 
        stream_types: AlpacaDataSource.StreamType,
        asset_class: AssetType,
        ) -> BaseStream:


        stream_map = {
            AlpacaDataSource.StreamType,
            AssetClass.US_EQUITY: StockDataStream,
            AssetClass.CRYPTO: CryptoDataStream}

        stream = stream_map[asset_class]

        return stream


    def _validate_symbols(self, symbols: List[str]):


        for symbol in symbols:

            symbol_data = self.client.symbols[symbol]

            if symbol_data is None:
                raise ValueError(f'Symbol {symbol} is not a known symbol.')

            if not symbol_data.tradable:
                logger.warning(f'Symbol {symbol} is not a tradable symbol.')

            if symbol_data.status != AssetStatus.ACTIVE:
                logger.warning(f'Symbol {symbol} is not an active symbol.')

            if not symbol_data.fractionable:
                logger.warning(
                    f'Symbol {symbol} is not a fractionable symbol.')

            if not symbol_data.easy_to_borrow:
                logger.warning(
                    f'Symbol {symbol} is not easy to borrow (ETB).')

        asset_classes = set(
            self.symbols.get(symbol).asset_class for symbol in symbols)

        # checks if symbols have the same asset class
        if len(asset_classes) != 1:
            raise ValueError('Symbols are not of the same asset class.')

        asset_class = asset_classes.pop()

        return asset_class
    


class AlpacaTradeClient(AbstractTradeClient):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self._cash = None



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
    def asset_quantities(self) -> Dict[str, float]:

        """
        Returns a dictionary of symbols and asset quantities for 
        the trader's current positions.

        Returns:
            dict: A dictionary mapping symbols to asset quantities.
        """

        asset_quantities = dict()

        positions_dataframe = self.get_positions_dataframe()
        symbols = positions_dataframe['symbol'].unique()

        for symbol in symbols:
            row = positions_dataframe.loc[positions_dataframe['symbol'] == symbol].iloc[0]
            quantity = (row['qty'] if row['side'] == 'long' else -1 * row['qty'])

            asset_quantities[symbol] = quantity

        return asset_quantities


    @property
    def net_worth(self):

        """
        The current net worth of the trader.

        Returns:
            float: The current net worth of the trader.
        """

        self._net_worth = self.client.account.portfolio_value

        return self._net_worth


    @property
    def longs(self):

        """
        The current long positions held by the trader.

        Returns:
            float: The current long positions held by the trader.
        """

        self._longs = self.client.account.long_market_value

        return self._longs


    @property
    def shorts(self):

        """
        The total value of all short positions held by the trader.

        Returns:
            float: The total value of all short positions held by the trader.
        """

        self._shorts = self.client.account.short_market_value

        return self._shorts


    def get_positions_dataframe(self) -> pd.DataFrame:

        """
        Get all current positions in the account.

        :return: A DataFrame containing the positions.
        """

        self._positions = self.clients['trading'].get_all_positions()

        positions_dataframe =  objects_to_df(self._positions)

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
        quantity: float,
        time_in_force: str,
        ) -> Order:


        assert time_in_force in ['ioc', 'fok'], 'Invalid time in force. options: ioc, fok}'

        side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
        quantity = abs(quantity)
        time_in_force = TimeInForce(time_in_force)


        market_order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.DAY)

        market_order = self.clients['trade'].submit_order(order_data=market_order_request)

        return market_order
