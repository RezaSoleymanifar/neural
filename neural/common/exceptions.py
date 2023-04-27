class CorruptDataError(Exception):

    """
    Custom exception for handling corrupt data errors.
    Raised when data is found to be corrupt or inconsistent.

    Args:
        message (str): A string describing the corrupt data error.
    """

    def __init__(self, message):

        """
        Initialize the CorruptDataError with an error message.

        Args:
            message (str): A string describing the corrupt data error.
        """

        self.message = message
        super().__init__(self.message)



class IncompatibleWrapperError(Exception):

    """
    Custom exception for handling incompatible wrapper errors.
    Raised when a wrapper is found to be incompatible with other enclosed wrappers.

    Args:
        message (str): A string describing the incompatible wrapper error.
    """

    def __init__(self, message):

        """
        Initialize the IncompatibleWrapperError with an error message.

        Args:
            message (str): A string describing the incompatible wrapper error.
        """

        self.message = message
        super().__init__(self.message)



class TradeConstraintViolationError(Exception):

    """
    Custom exception for handling trade constraint violation errors.
    Raised when certain constraints or rules are violated.

    Args:
        message (str): A string describing the trade constraint violation error.
    """

    def __init__(self, message):
        """
        Initialize the TradeConstraintViolationError with an error message.

        Args:
            message (str): A string describing the trade constraint violation error.
        """
        self.message = message
        super().__init__(self.message)
