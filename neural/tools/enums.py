from enum import Enum


class CalendarType(Enum):

    """
    Enum representing the different types of calendar types for trading purposes.
    Attributes:
        US_EQUITY: A string representing the US Equity market calendar type, specifically 
        the New York Stock Exchange (NYSE).
        CRYPTO: A string representing the cryptocurrency market calendar type, which operates 24/7.
    """
    
    US_EQUITY = 'NYSE'
    CRYPTO = '24/7'
    FOREX = '24/7'
