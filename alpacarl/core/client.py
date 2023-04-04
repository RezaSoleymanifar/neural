from alpaca.trading.enums import AssetExchange, AssetClass, AccountStatus, AssetStatus
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.trading import TradingClient
from alpacarl.meta.constants import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_ENDPOINT,
    ALPACA_API_ENDPOINT_PAPER)

from alpacarl.meta import log

class AlpacaMetaClient:
    def __init__(
            self,
            key: str,
            secret: str,
            sandbox=False
            ) -> None:

        # if sandbox = True tries connecting to paper account endpoint
        self.key = key if key else ALPACA_API_KEY
        self.secret = secret if secret else ALPACA_API_SECRET
        self.endpoint = ALPACA_API_ENDPOINT if not sandbox else ALPACA_API_ENDPOINT_PAPER
        self.clients = None
        self.account = None

        self._assets = None
        self._asset_classes = None
        self._exchanges = None

        return None

    def setup_clients_and_account(self) -> None:

        # crypto does not need key, and secret but will be faster if provided
        self.clients['crypto'] = CryptoHistoricalDataClient(
            self.key, self.secret)
        self.clients['stocks'] = StockHistoricalDataClient(
            self.key, self.secret)
        self.clients['trading'] = TradingClient(self.key, self.secret)

        try:
            self.account = self.clients['trading'].get_account()

            if self.account.status == AccountStatus.ACTIVE:
                log.logger.info(
                    f'Clients and account setup successful. Account is active.')

        except Exception as e:
            log.logger.exception("Account setup failed: {}".format(str(e)))

        return None

    @property
    def symbols(self):

        if self._symbols is None:
            self._symbols = {
                asset.pop('symbol'): asset for asset in self._assets}

        return self._symbols

    @property
    def assets(self):

        if self._assets is None:
            assets_ = self.clients['trading'].get_all_assets()
            # keep tradable active assets only.
            self._assets = [asset for asset in assets_ if
                            asset.status == AssetStatus.ACTIVE and asset.tradable]

        return dicts_enum_to_df(self._assets)

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

        return dicts_enum_to_df(self._positions)

    def set_credentials(self, key: str, secret: str) -> None:

        if not isinstance(key, str) or not isinstance(secret, key):
            raise ValueError(f'key and secret must of type {str}')

        self.key = key
        self.secret = secret

        return None

    def set_endpoint(self, endpoint: str) -> None:

        if not isinstance(endpoint, str):
            raise ValueError(f'endpoint must of type {str}')

        return None

# converts dictionaries of enum objects into dataframe
def dicts_enum_to_df(
        info: Iterable[Dict[str, str]]
        ) -> DataFrame:

    for dict_ in info:
        for key, val in dict_.items():
            dict_[key] = val.value if isinstance(val, Enum) else val

    df = DataFrame(info)
    return df
