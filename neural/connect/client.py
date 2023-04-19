from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import pandas as pd

from alpaca.trading.enums import AccountStatus, AssetExchange, AssetClass
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.trading import TradingClient

from neural.common.log import logger
from neural.common.constants import API_KEY, API_SECRET
from neural.tools.ops import objects_to_df



class AbstractClient(ABC):

    """
    Abstract base class for a client that defines the required methods for setting credentials
    and checking connection.
    """

    @abstractmethod
    def set_credentials(self, *args, **kwargs):

        """
        Set the credentials for the client.

        This method should be implemented by derived classes to set the required
        credentials (e.g., API keys, tokens, usernames, passwords) for connecting
        to a specific service.

        :param args: Positional arguments to be passed to the implementation.
        :param kwargs: Keyword arguments to be passed to the implementation.
        :raises NotImplementedError: If the method is not implemented in the derived class.
        """

        raise NotImplementedError


    @abstractmethod
    def check_connection(self, *args, **kwargs):

        """
        Check the connection to the service.

        This method should be implemented by derived classes to test the connection
        to a specific service, usually by sending a request and verifying the response.

        :param args: Positional arguments to be passed to the implementation.
        :param kwargs: Keyword arguments to be passed to the implementation.
        :raises NotImplementedError: If the method is not implemented in the derived class.
        """

        raise NotImplementedError



class AlpacaMetaClient(AbstractClient):

    """
    AlpacaMetaClient is a concrete implementation of the AbstractClient class.
    It provides access to the Alpaca API for trading, stocks, and crypto data,
    as well as account and asset management.
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

        return None


    @property
    def __symbols(self) -> Dict:

        """
        Get the symbols of assets fetched from the Alpaca API.

        :return: A dictionary of symbols mapped to their respective assets.
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
            self._assets = self.clients[
                'trading'].get_all_assets()

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

        return [item.value for item in self._asset_classes]


    @property
    def positions(self):

        """
        Get all current positions in the account.

        :return: A DataFrame containing the positions.
        """

        self._positions = self.clients['trading'].get_all_positions()

        return objects_to_df(self._positions)
    

    def setup_clients_and_account(self) -> None:

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
            if self.check_connection():

                logger.info(
                    f'Clients and account setup successful. Status: {self.account.status}')

        except Exception as e:

            logger.exception(
                f'Account setup failed: {e}')

        return None


    def set_credentials(
        self, 
        key: str, 
        secret: str
        ) -> None:

        """
        Set the API key and secret for the AlpacaMetaClient.

        :param key: API key for the Alpaca API.
        :param secret: Secret key for the Alpaca API.
        """


        self.key = key
        self.secret = secret

        return None
    

    def check_connection(self):

        """
        Check the connection to the Alpaca API.

        :return: True if the account status is active, False otherwise.
        """

        return True if self.account.status == AccountStatus.ACTIVE else False
