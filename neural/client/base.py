from abc import ABC, abstractmethod



class AbstractClient(ABC):

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
        """
        Check the connection to the service. If the connection is successful, the method
        should return True, otherwise False. The Trader class will use this method to check
        the connection before executing any trades.

        This method should be implemented by derived classes to test the connection
        to a specific service, usually by sending a request and verifying the response.

        Parameters:
        ------------
            *args: Positional arguments to be passed to the implementation.
            **kwargs: Keyword arguments to be passed to the implementation.

        Raises:
        --------
            NotImplementedError: If the method is not implemented in the derived class.
        """

        raise NotImplementedError
