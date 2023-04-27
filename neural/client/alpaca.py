from typing import Optional, Dict, List

import pandas as pd

from alpaca.trading.enums import AccountStatus, AssetExchange, AssetClass
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.trading import TradingClient

from neural.common.log import logger
from neural.common.constants import API_KEY, API_SECRET
from neural.client.base import AbstractClient
from neural.tools.ops import objects_to_df


class AlpacaClient(AbstractClient):


    """
    AlpacaClient is a Python class that allows you to connect to the Alpaca API to trade, 
    get stock and crypto data, manage your account and assets, and more. To use this class, 
    you will need to set up an API key and secret, which can be obtained from the Alpaca website. 
    You can find instructions on how to do this in the Alpaca documentation 
    [https://alpaca.markets/learn/connect-to-alpaca-api/]. Once you have your API key and secret, 
    you can instantiate an instance of the AlpacaClient class and begin accessing the API.

    Attributes:
    ------------
    key (str): The API key to be used for authentication (optional, defaults to None).
    secret (str): The secret key to be used for authentication (optional, defaults to None).
    clients (dict): A dictionary of clients for crypto, stocks, and trading.
    account (Account): The account information retrieved from the Alpaca API.
    _assets (list): A list of assets fetched from the Alpaca API.
    _symbols (dict): A dictionary of symbols mapped to their respective assets.
    _positions (list): A list of current positions in the account.
    _asset_classes (list): A list of all supported asset classes.
    _exchanges (list): A list of all supported exchanges.

    Methods:
    --------
    assets() -> pandas.DataFrame: Get all assets available through the Alpaca API.
    exchanges() -> list: Get a list of all supported exchanges.
    asset_classes() -> list: Get a list of all supported asset classes.
    positions() -> pandas.DataFrame: Get all current positions in the account.
    check_connection() -> bool: Check the connection to the Alpaca API.

    Example:
    ------------
    Option 1: Instantiate an instance of the AlpacaClient class with your API key and secret.

    >>> from neural.connect.alpaca import AlpacaClient
    >>> client = AlpacaClient(key=..., secret=...)
    >>> assets = client.assets()
    >>> positions = client.positions()

    Option 2: Instantiate an instance of the AlpacaClient class without your API key and secret.

    >>> from neural.connect.alpaca import AlpacaClient
    >>> from neural.common.constants import API_KEY, API_SECRET
    >>> API_KEY = ...
    >>> API_SECRET = ...
    >>> client = AlpacaClient()

    Option 3: Instantiate an instance of the AlpacaClient class with environment variables.

    # Set the environment variables for API key and secret
    export API_KEY=your_api_key
    export API_SECRET=your_secret_key

    # Instantiate an instance of the AlpacaClient class
    python
    >>> from neural.connect.alpaca import AlpacaClient
    >>> import os
    >>> client = AlpacaClient(key=os.environ['API_KEY'], secret=os.environ['API_SECRET'])
    """

    def __init__(
        self,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        ) -> None:
        super.__init__

        self.key = key if key is not None else API_KEY
        self.secret = secret if secret is not None else API_SECRET

        self.clients = None
        self.account = None
        self._assets = None
        self._symbols = None
        self._positions = None
        self._asset_classes = None
        self._exchanges = None

        self._connect()

        return None


    @property
    def __symbols(self) -> Dict:

        """
        Get the symbols of assets fetched from the Alpaca API.

        Returns
        -------
        Dict
            A dictionary of symbols mapped to their respective assets.
        """

        if self._assets is None: # fetch assets first
            self.assets

        if self._symbols is None:
            self._symbols = {
                asset.symbol: asset for asset in self._assets}

        return self._symbols



    @property
    def assets(self) -> pd.DataFrame:

        """
        Get all assets available through the Alpaca API.

        :return: A DataFrame containing the assets.
        """

        if self._assets is None:
            self._assets = self.clients['trading'].get_all_assets()

        return objects_to_df(self._assets)



    @property
    def exchanges(self) -> List[str]:

        """
        Get a list of all supported exchanges.

        :return: A list of exchange names.
        """

        if self._exchanges is None:
            self._exchanges = [item for item in AssetExchange]

        return [item.value for item in self._exchanges]



    @property
    def asset_classes(self) -> List[str]:

        """
        Get a list of all supported asset classes.

        :return: A list of asset class names.
        """

        if self._asset_classes is None:
            self._asset_classes = [item for item in AssetClass]

        printed_asset_classes =  [item.value for item in self._asset_classes]
        return printed_asset_classes



    @property
    def positions(self) -> pd.DataFrame:

        """
        Get all current positions in the account.

        :return: A DataFrame containing the positions.
        """

        self._positions = self.clients['trading'].get_all_positions()

        positions_dataframe =  objects_to_df(self._positions)

        return positions_dataframe


    def _connect(self) -> None:

        """
        Set up clients for crypto, stocks, and trading, and retrieve account information.

        :raises Error: If login fails.
        """

        if self.key is None or self.secret is None:
            raise ValueError('Key and secret are required for account login.')
        
        self.clients = dict()

        # crypto does not need key, and secret but will be faster if provided
        self.clients['crypto'] = CryptoHistoricalDataClient(
            api_key = self.key, secret_key =self.secret)
        self.clients['stocks'] = StockHistoricalDataClient(
            api_key=self.key, secret_key=self.secret)
        self.clients['trading'] = TradingClient(
            api_key=self.key, secret_key=self.secret)

        try:

            self.account = self.clients.get('trading').get_account()

            logger.info(
                f'Account setup successful.')

            if self.check_connection():

                logger.info(
                    f'Account Status: {self.account.status}')
            else:
                logger.warning(
                    f'Account Status: {self.account.status}')

        except Exception as e:

            logger.exception(
                f'Account setup failed: {e}')

        return None
    

    def check_connection(self):

        """
        Check the connection to the Alpaca API.

        :return: True if the account status is active, False otherwise.
        """

        return True if self.account.status == AccountStatus.ACTIVE else False


    def submit_order(
        self, 
        symbol: str, 
        qty: int, 
        side, 
        type, 
        time_in_force, 
        limit_price=None, 
        stop_price=None, 
        client_order_id=None
        ) -> None:

        """
        Submit an order to the Alpaca API.

        :param symbol: The symbol of the asset to be traded.
        :param qty: The quantity of the asset to be traded. If fraction only market orders are allowed.
        :param side: The side of the order, either 'buy' or 'sell'.
        :param type: The type of order, either 'market', 'limit', 'stop', or 'stop_limit'.
        :param time_in_force: The time in force for the order, either 'day', 'gtc', 'opg', 'ioc', or 'fok'.
        :param limit_price: The limit price for the order (optional, defaults to None).
        :param stop_price: The stop price for the order (optional, defaults to None).
        :param client_order_id: The client order ID for the order (optional, defaults to None).
        :return: The order ID.
        """

        try:

            order = self.clients['trading'].submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id)

            logger.info(
                f'Order submitted successfully: {order}')

            return order.id

        except Exception as e:

            logger.exception(
                f'Order submission failed: {e}')

            return None