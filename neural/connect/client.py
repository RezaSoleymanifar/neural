from typing import Optional

from alpaca.trading.enums import AccountStatus, AssetExchange, AssetClass
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.trading import TradingClient

from neural.common.log import logger
from neural.common.constants import API_KEY, API_SECRET
from neural.tools.ops import objects_to_df
from abc import ABC, abstractmethod



class AbstractClient(ABC):

    @abstractmethod
    def set_credentials(self, *args, **kwargs):

        raise NotImplementedError

    @abstractmethod
    def check_connection(self, *args, **kwargs):
        raise NotImplementedError




class AlpacaMetaClient(AbstractClient):
    def __init__(
        self,
        key: Optional[str] = None,
        secret: Optional[str] = None,
        ) -> None:
        super.__init__
        # if sandbox = True tries connecting to paper account endpoint
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
    def __symbols(self):
        
        # makes sure assets are fetched.
        if self._assets is None:
            self.assets

        if self._symbols is None:
            self._symbols = {
                asset.symbol: asset for asset in self._assets}

        return self._symbols

    @property
    def assets(self):

        if self._assets is None:
            self._assets = self.clients[
                'trading'].get_all_assets()

        return objects_to_df(self._assets)

    @property
    def exchanges(self):

        if self._exchanges is None:
            self._exchanges = [item for item in AssetExchange]

        return [item.value for item in self._exchanges]

    @property
    def asset_classes(self):

        if self._asset_classes is None:
            self._asset_classes = [item for item in AssetClass]

        return [item.value for item in self._asset_classes]

    @property
    def positions(self):

        self._positions = self.clients['trading'].get_all_positions()

        return objects_to_df(self._positions)
    

    def setup_clients_and_account(self) -> None:

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

            self.account = self.clients['trading'].get_account()
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

        if not isinstance(key, str) or not isinstance(secret, str):
            raise ValueError(f'key and secret must of type {str}')

        self.key = key
        self.secret = secret

        return None
    
    def check_connection(self, *args, **kwargs):

        return True if self.account.status == AccountStatus.ACTIVE else False
