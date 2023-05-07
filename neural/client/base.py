from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np


if TYPE_CHECKING:
    from neural.data.enums import AbstractDataSource


class AbstractClient(ABC):

    """
    Abstract base class for API clients. Provides facility for connecting to the API at construction.
    Child classes are expected to implement connection logic in the `_connect` method. Credentials
    should be passed to the constructor. This client can provide trading and data functionality.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the `_connect` method.
    **kwargs : dict
        Keyword arguments to be passed to the `_connect` method.

    Notes
    -----
    This abstract class defines the common interface and functionality for API clients.
    Subclasses must implement the `_connect` method to establish a connection to the API.

    Raises
    ------
    NotImplementedError
        If the `_connect` method is not implemented in the subclass.

    Examples
    --------
    To create a new API client:

    >>> class MyClient(AbstractClient):
    ...     def connect(self, *args, **kwargs):
    ...         # Connect to the API
    ...         pass

    >>> client = MyClient()

    """

    def __init__(self, *args, **kwargs):
        # this can be used post initialization to automaitcally connect to the API
        self.connect(*args, **kwargs)

    @abstractmethod
    def connect(self, *args, **kwargs):
        """
        Connect to the API. This method must be implemented in the subclass.
        """
        raise NotImplementedError


class AbstractTradeClient(AbstractClient):

    """
    Abstract base class for a client that connects to a trading service or API.

    This class defines the required methods for setting credentials and checking the connection to the service. 
    Derived classes must implement these methods to provide the necessary functionality for connecting to 
    a specific service.

    Attributes:
    ------------
        cash (float): The current amount of cash available to the trader.
        asset_quantities (ndarray[float]): The current quantity of each asset held by the trader.
        positions (float): The current positions (notional base currency value) of each asset held by the trader.
        net_worth (float): The current net worth of the trader.
        longs (float): Sum of current notional value of long positions held by the trader.
        shorts (float): Sum of current notional value of short positions held by the trader.

    Methods:
    ------
        check_connection(*args, **kwargs) -> bool:
            Check the connection to the service. If the connection is successful, the method should return True, 
            otherwise False. The Trader class will use this method to check the connection before executing 
            any trades.
        place_order(*args, **kwargs):
            Abstract method for placing an order for a single asset.

    Raises:

        NotImplementedError: If the method is not implemented in the derived class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def cash(self) -> float:
        """
        The current amount of cash available to the trader.

        Raises:
        --------
            NotImplementedError: This property must be implemented by a subclass.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def asset_quantities(self) -> np.ndarray[float]:
        """
        The current quantity of each asset held by the trader.

        Raises:
        --------
            NotImplementedError: This property must be implemented by a subclass.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def net_worth(self) -> float:
        """
        The current net worth of the trader.

        Raises:
        --------
            NotImplementedError: This property must be implemented by a subclass.
        """

        raise NotImplementedError

    @abstractmethod
    def check_connection(self, *args, **kwargs):
        """
        check the connection to the service. If the connection is successful,
        the method should return True, otherwise False. The Trader class will use this method to check
        the connection before execution of trading process.
        """

        raise NotImplementedError

    @abstractmethod
    def place_order(self, *args, **kwargs):
        """
        Abstract method for placing an order for a single asset.
        """

        raise NotImplementedError


class AbstractDataClient(AbstractClient):

    """
    Abstract base class for a client that connects to a data service or API.
    This class defines a blueprint for clients that provide data functionality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def data_source(self) -> AbstractDataSource:
        """
        The name of the data source. Data clients are enforced
        to have a data source attribute. This helps mapping clients to
        constituents of stream metadata that specify a data source.
        """

        raise NotImplementedError
