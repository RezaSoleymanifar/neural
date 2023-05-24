from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np


if TYPE_CHECKING:
    from neural.data.base import AbstractDataSource


class AbstractClient(ABC):

    """
    Abstract base class for API clients. Provides facility for
    connecting to the API at construction. Child classes are expected to
    implement connection logic in the `connect` method. Credentials
    should be passed to the constructor. This client can provide trading
    and/or data functionality.

    Parameters
    ----------
    *args : tuple
        Positional arguments to be passed to the `_connect` method.
    **kwargs : dict
        Keyword arguments to be passed to the `_connect` method.

    Raises
    ------
    NotImplementedError
        If the `connect` method is not implemented in the subclass.

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
        """
        Initialize the client and connect to the API.
        """
        self.connect(*args, **kwargs)

    @abstractmethod
    def connect(self, *args, **kwargs):
        """
        Connect to the API. This method must be implemented in the
        subclass.
        """
        raise NotImplementedError


class AbstractTradeClient(AbstractClient):

    """
    Abstract base class for a client that connects to a trading service
    or API. Trade clients are enforced to provide the bare minimum
    functionality required for agent to make trading decisions. This is 
    an extension of the AbstractClient class that in addition to
    connectivity, provides trading functionality. The Trader class
    expects the client to provide the following information:
        - cash
        - asset_quantities

    And also following functionality:
        - check_connection
        - place_order


    Attributes
    ----------
    cash : float
        The current amount of cash available to the trader.
    asset_quantities : np.ndarray[float]
        The current quantity of each asset held by the trader.

    Raises
    ------
    NotImplementedError
        If the `cash` or `asset_quantities` properties are not
        implemented in the subclass.

    Methods
    -------
    check_connection
        Check the connection to the service. If the connection is
        successful, the method should return True, otherwise False. The
        Trader class will use this method to check the connection before
        execution of trading process.
    place_order
        Place an order for a single asset.

    Examples
    --------
    To create a new trade client:

    >>> class MyTradeClient(AbstractTradeClient):
    ...     def connect(self, *args, **kwargs):
    ...         # Connect to the API
    ...         pass
    ...     @property
    ...     def cash(self) -> float:
    ...         # Return the current amount of cash available to the
    ...         # trader.
    ...         pass
    ...     @property
    ...     def asset_quantities(self) -> np.ndarray[float]:
    ...         # Return the current quantity of each asset held by the
    ...         # trader.
    ...         pass
    ...     def check_connection(self, *args, **kwargs):
    ...         # Check the connection to the service. If the connection    
    ...         # is successful, the method should return True, otherwise   
    ...         # False. The Trader class will use this method to check the
    ...         # connection before execution of trading process.
    ...         pass
    ...     def place_order(self, *args, **kwargs): 
    ...         # Place an order for a single asset.    
    ...         pass
    
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
            NotImplementedError: This property must be implemented by a
            subclass.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def asset_quantities(self, *args, **kwargs) -> np.ndarray[float]:
        """
        The current quantity of each asset held by the trader.

        Raises:
        --------
            NotImplementedError: This property must be implemented by a
            subclass.
        """

        raise NotImplementedError


    @abstractmethod
    def check_connection(self, *args, **kwargs):
        """
        check the connection to the service. If the connection is
        successful, the method should return True, otherwise False. The
        Trader class will use this method to check the connection before
        execution of trading process.
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
    Abstract base class for a client that connects to a data service or
    API. This class defines a blueprint for clients that provide data
    functionality.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def data_source(self) -> AbstractDataSource:
        """
        The name of the data source. Data clients are enforced to have a
        data source attribute. This helps mapping clients to
        constituents of stream metadata that specify a data source.
        """

        raise NotImplementedError
