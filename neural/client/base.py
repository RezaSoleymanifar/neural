from abc import ABC, abstractmethod



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
    ...     def _connect(self, *args, **kwargs):
    ...         # Connect to the API
    ...         pass

    >>> client = MyClient()

    """

    def __init__(self, *args, **kwargs):
        self._connect(*args, **kwargs)


    @abstractmethod
    def _connect(self, *args, **kwargs):

        raise NotImplementedError



class AbstractDataClient(AbstractClient):
    pass


class AbstractTradeClient(AbstractClient):

    """
    Abstract base class for a client that connects to a trading service or API.

    This class defines the required methods for setting credentials and checking the connection to the service. 
    Derived classes must implement these methods to provide the necessary functionality for connecting to a specific service.

    Methods:
        check_connection(*args, **kwargs) -> bool: Check the connection to the service. If the connection is successful, 
        the method should return True, otherwise False. The Trader class will use this method to check 
        the connection before executing any trades.

    Raises:
        NotImplementedError: If the method is not implemented in the derived class.
    """
    

    @abstractmethod
    def check_connection(self, *args, **kwargs):

        raise NotImplementedError


    @abstractmethod
    def place_order(self, *args, **kwargs):

        raise NotImplementedError
