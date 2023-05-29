"""
base.py

Description:
------------
    This module provides basic functionality for API clients. The
    inheritance structure of classes provided are as follows:
        
        AbstractClient 
            - AbstractTradeClient 
            - AbstractDataClient
    
    AbstractClient is an abstract base class for API clients that
    provides connectivity functionality. Concrete classes are expected
    to implement connection logic in the `connect` method.
    AbstractTradeClient and AbstractDataClient extend the AbstractClient
    to handle trading and data functionality respectively.


License:
--------
    MIT License. See LICENSE.md file.

Author(s):
-------
    Reza Soleymanifar, Email: Reza@Soleymanifar.com

Classes:
--------
    AbstractClient(ABC):
        Abstract base class for API clients. Provides facility for
        connecting to the API at construction. Child classes are
        expected to implement connection logic in the `connect` method.
        Credentials should be passed to the constructor. This client can
        provide trading and/or data functionality.

    AbstractTradeClient(AbstractClient):
        Abstract base class for a client that connects to a trading
        service or API. Trade clients are enforced to provide the bare
        minimum functionality required for agent to make trading
        decisions. This is an extension of the AbstractClient class that
        in addition to connectivity, provides trading functionality. The
        Trader class expects the client to provide the following
        information:
            - cash
            - asset_quantities  

        And also following functionality:
            - check_connection
            - place_order

    AbstractDataClient(AbstractClient):
        Abstract base class for a client that connects to a data service
        or API. This class defines a blueprint for clients that provide
        data functionality. data_source property will be used to map
        clients to constituents of stream metadata that has data_source
        attribute. For example a Trader class may use multiple data
        clients at construction (e.g. one borker data client and one
        twitter data client). The Trader class will use the data source
        attribute of each client to map the client to the corresponding
        stream metadata, for live streaming of data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np

from neural.data.base import AbstractAsset

if TYPE_CHECKING:
    from neural.data.base import AbstractDataSource


class AbstractClient(ABC):
    """
    Abstract base class for API clients. Provides facility for
    connecting to the API at construction. Child classes are expected to
    implement connection logic in the `connect` method. Credentials
    should be passed to the constructor. This client can provide trading
    and/or data functionality.

    Args
    ----------
        *args : Tuple
            Positional arguments to be passed to the `connect` method.
        **kwargs : Dict
            Keyword arguments to be passed to the `connect` method.

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

        Args
        ------
        *args : Tuple
            Positional arguments to be passed to the `_connect` method.
        **kwargs : Dict
            Keyword arguments to be passed to the `connect` method.
        """
        self.connect(*args, **kwargs)

    @abstractmethod
    def connect(self, *args, **kwargs):
        """
        Connect to the API. This method must be implemented in the
        subclass.

        Args
        ----
        *args : Tuple   
            Positional arguments to be passed to the `connect` method.
        **kwargs : Dict
            Keyword arguments to be passed to the `connect` method.
        """
        raise NotImplementedError


class AbstractDataClient(AbstractClient):
    """
    Abstract base class for a client that connects to a data service or
    API. This class defines a blueprint for clients that provide data
    functionality. data_source property will be used to map clients to
    constituents of stream metadata that has data_source attribute. For
    example a Trader class may use multiple data clients at construction
    (e.g. one borker data client and one twitter data client). The
    Trader class will use the data source attribute of each client to
    map the client to the corresponding stream metadata, for live
    streaming of data.
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

    Properties
    ----------
        cash (float):
            The current amount of cash available to the trader. Cash can
            be positive or negative. Negative cash indicates that the
            trader is in debt.

    Raises
    ------
        NotImplementedError
            If the `cash` or `asset_quantities` properties are not
            implemented in the subclass.

    Methods
    -------
        asset_quantities(assets: List[AbstractAsset], *args, **kwargs)
        -> np.ndarray[float]:
            Returns numpy array containing quantities of the assets
            provided as argument. Asset quantities can be positive or
            negative. Negative quantities indicate that the trader has
            shorted the asset, namely the trader owes the asset to the
            broker.
        check_connection(*args, **kwargs):
            Check the connection to the service. If the connection is
            not successful, an error should be raised. The Trader class
            will use this method to check the connection before
            execution of trading process.
        place_order(*args, **kwargs):
            Place an order for a single asset.

    Examples
    --------
    To create a new trade client:

    >>> class MyTradeClient(AbstractTradeClient):
    ...    def __init__(self, *args, **kwargs):
    ...        super().__init__(*args, **kwargs)
    ...     @property
    ...     def cash(self):
    ...         # Return the current amount of cash available to the
    ...         # trader.
    ...         pass
    ...     def connect(self, *args, **kwargs):
    ...         # Connect to the API
    ...         pass
    ...     def get_asset_quantities(self, assets, *args, **kwargs):
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
        """
        Initialize the client and connect to the API.

        Args
        ----------
            *args : Tuple
                Positional arguments to be passed to the `connect`
                method.
            **kwargs : Dict
                Keyword arguments to be passed to the `connect` method.
        """
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def cash(self) -> float:
        """
        The current amount of cash available to the trader. Cash can be
        positive or negative. Negative cash indicates that the trader is
        in debt.

        Raises:
        --------
            NotImplementedError: 
                This property must be implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_asset_quantities(self, assets: List[AbstractAsset], *args,
                             **kwargs) -> np.ndarray[float]:
        """
        The current quantity of each asset held by the trader. Asset
        quantities can be positive or negative. Negative quantities
        indicate that the trader has shorted the asset, namely the
        trader owes the asset to the broker.

        Args:
        ----
            assets (List[AbstractAsset]):
                A list of assets to get the quantities for.

        Raises:
        --------
            NotImplementedError: 
                This property must be implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def check_connection(self, *args, **kwargs):
        """
        check the connection to the service. If the connection is not
        successful it raises an error. The Trader class will use this
        method to check the connection before execution of trading
        process.

        Raises:
        --------
            NotImplementedError: 
                This method must be implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def place_order(self, *args, **kwargs):
        """
        Abstract method for placing an order for a single asset. The
        restrictions of the API should be enforced in this method.
        """
        raise NotImplementedError
