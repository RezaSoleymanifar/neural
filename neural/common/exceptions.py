"""
exceptions.py

Description:
------------
    This is a module for storing constants used in the library.

License:
--------
    MIT License. See LICENSE.md file.

Author(s):
-------
    Reza Soleymanifar, Email: Reza@Soleymanifar.com

Exceptions:
-----------
    CorruptDataError: 
        Raised when data is found to be corrupt or inconsistent.
        Typically used to match dataset metadata with actual data.
    IncompatibleWrapperError: 
        Raised when a wrapper is found to be incompatible with other
        enclosed wrappers. Can be used to enforce proper ordering of
        wrappers.
    TradeConstraintViolationError:
        Raised when certain trade constraints or rules are violated.
        This can include a variety of constraints, pattern day trading
        minimum violation, receiving margin calls, etc.
"""
class CorruptDataError(Exception):
    """
    Custom exception for handling corrupt data errors. Raised when data
    is found to be corrupt or inconsistent. Typically used to match
    dataset metadata with actual data.
    """

    def __init__(self, message) -> None:
        """
        Initialize the CorruptDataError with an error message.

        Args:
        -----
            message (str): 
                A string describing the corrupt data error.
        """

        self.message = message
        super().__init__(self.message)

        return None

class IncompatibleWrapperError(Exception):
    """
    Custom exception for handling incompatible wrapper errors. Raised
    when a wrapper is found to be incompatible with other enclosed
    wrappers. Used to enforce proper ordering of wrappers.
    """
    def __init__(self, message) -> None:
        """
        Initialize the IncompatibleWrapperError with an error message.

        Args:
        ------
            message (str): 
                A string describing the incompatible wrapper error.
        """

        self.message = message
        super().__init__(self.message)

        return None


class TradeConstraintViolationError(Exception):
    """
    Custom exception for handling trade constraint violation errors.
    Raised when certain constraints or rules are violated. This can
    include a variety of constraints, pattern day trading minimum
    violation, receiving margin calls, etc.
    """
    def __init__(self, message):
        """
        Initialize the TradeConstraintViolationError with an error
        message.

        Args:
        -----
            message (str): 
                A string describing the trade constraint violation
                error.
        """
        self.message = message
        super().__init__(self.message)

        return None
