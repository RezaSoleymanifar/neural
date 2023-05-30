"""" 
enums.py

Contains enums for different types of data and assets. These enums are
used to create a schema for datasets and streams. The schema is a
dictionary that contains boolean masks for different types of data and
assets. The schema is used to filter data and assets for training
environments. The schema is also used to create feature vectors for
each asset.

Classes:
---------
    CalendarType: 
        An enum class representing different types of trading calendars.
        Each calendar type has a schedule property that returns a
        schedule function for the calendar type. The schedule function
        takes a start_date and end_date and returns a dataframe with
        trading dates and times.
    FeatureType:
        An enum class representing different types of features in data.
        Typically used to define the feature schema of a dataset or
        stream. This means that a boolean mask is created for each
        feature type that indicates the location corresponding columns
        is in the dataset or stream. features types are universal and
        can be found in any data source and dataset.This enumeration is
        also useful for feature engineering where a certain type of
        input is required to commpute a derivative feature such as MACD 
        or RSI financial indicators.
    AssetType:
        An enum class representing different categories of financial
        instruments.
    
Notes:
---------
    This module is not a representative list of all the feature types
    that can be used in datasets and streams. This is just a list of
    the most common feature types that are frequently used in a way
    that needs to be distinguished them from other feature types, for
    example for feature engineering, using OHLC prices, or for
    filtering text columns for passing to large language models. This
    list can be extended to include other feature types that are not
    included here using the FeatureType.MY_FEATURE_TYPE = 'KEY_WORD'
    syntax. Feature schema will automatically look for columns that
    contain name 'KEY_WORD' and create a boolean mask for them in
    feature shema.
"""
from enum import Enum
from neural.common.constants import CALENDAR


class CalendarType(Enum):

    """
    An enum object that standardizes different types of trading
    calendars.

    Properties:
    ------------
        schedule (Callable[[Any, Any], pd.DataFrame]):
            A function that returns a dataframe with trading dates and
            times. Start and end dates are any type that can be handle
            by to_datetime() method of pandas. The dataframe will have
            output similar to the following. It Rows correspond to
            trading dates and columns correspond to start and end times
            of trading intervals. For instance, if the calendar is 24/7,
            then the start and end times will be 00:00:00 and 00:00:00
            (next day) respectively.

            >>> schedule(start_date, end_date)
                        start                       end
            2022-01-03  2022-01-03 00:00:00+00:00   2022-01-04 00:00:00+00:00
            2022-01-04  2022-01-04 00:00:00+00:00   2022-01-05 00:00:00+00:00
            2022-01-05  2022-01-05 00:00:00+00:00   2022-01-06 00:00:00+00:00
            2022-01-06  2022-01-06 00:00:00+00:00   2022-01-07 00:00:00+00:00
            2022-01-07  2022-01-07 00:00:00+00:00   2022-01-08 00:00:00+00:00
            2022-01-10  2022-01-10 00:00:00+00:00   2022-01-11 00:00:00+00:00
    Notes:
    ----------
    If an asset does not fall under these calendar categories it can be
    handled by user speciying the CalendarType.MY_CALENDAR_TYPE =
    'VALID_PANDAS_CALENDAR' and providing a valid
    pandas_market_calendars calendar. This is the calendar used by
    default and CalendarType enum values correpond to valid strings for
    the pandas_market_calendars get_calendar() method. More information
    here:
    https://pandas-market-calendars.readthedocs.io/en/latest/modules.html.
    If not happy with the default calendar, user can specify a different
    calendar by setting the CALENDAR constant in
    neural/common/constants.py to a calendar following the interface of
    AbstractCalendar class in neural.utils.time. This allows handling
    local calendars that are not supported by pandas_market_calendars.

    Examples:
    ----------
    Option 1: use default calendar and pandas_market_calendars:
    
    >>> CalendarType.MY_CALENDAR_TYPE = 'VALID_PANDAS_CALENDAR'

    Option 2: use custom calendar and custom calendar type that is not
    supported by pandas_market_calendars:
    
    >>> from neural.data.enums import CalendarType 
    >>> from neural.common.constants import CALENDAR 
    >>> CalendarType.MY_CALENDAR_TYPE = 'MY_CALENDAR' 
    >>> CALENDAR = CustomCalendarClass 
    >>> CalendarType.MY_CALENDAR_TYPE.schedule(start_date, end_date)
    """

    TWENTY_FOUR_SEVEN = '24/7'
    TWENTY_FOUR_FIVE = '24/5'
    NEW_YORK_STOCK_EXCHANGE = 'NYSE'
    NATIONAL_ASSOCIATION_OF_SECURITIES_DEALERS_AUTOMATED_QUOTATIONS = 'NASDAQ'
    CHICAGO_MERCANTILE_EXCHANGE = 'CME'
    INTERCONTINENTAL_EXCHANGE = 'ICE'
    LONDON_STOCK_EXCHANGE = 'LSE'
    TOKYO_STOCK_EXCHANGE = 'TSE'
    SINGAPORE_EXCHANGE = 'SGX'
    AUSTRALIAN_SECURITIES_EXCHANGE = 'ASX'
    MOSCOW_EXCHANGE = 'MOEX'
    BME_SPANISH_EXCHANGES = 'BM'
    BOVESPA = 'FBOVESPA'
    JOHANNESBURG_STOCK_EXCHANGE = 'JSE'
    SHANGHAI_STOCK_EXCHANGE = 'SSE'
    SHENZHEN_STOCK_EXCHANGE = 'SZSE'
    HONG_KONG_EXCHANGES_AND_CLEARING = 'HKEX'
    NATIONAL_STOCK_EXCHANGE_OF_INDIA = 'NSE'
    BOMBAY_STOCK_EXCHANGE = 'BSE'
    KOREA_EXCHANGE = 'KRX'
    TAIWAN_STOCK_EXCHANGE = 'TWSE'

    @property
    def schedule(self):
        """
        Schedule property that returns a schedule function for the
        calendar type. The schedule function takes a start_date and
        end_date and returns a dataframe with trading dates and times.

        Returns:
        ---------
            function: 
                A function that returns a dataframe with trading dates
                and times.
            
        Examples:
        ---------
        >>> CalendarType.NEW_YORK_STOCK_EXCHANGE.schedule(start_date, end_date)
        """
        schedule =  CALENDAR(self.value).schedule
        schedule = lambda start_date, end_date: schedule(
            self, start_date, end_date)
        return schedule

class FeatureType(Enum):

    """
    Enumeration class that defines constants for the different types of
    features in data. Typically used to define the feature schema of a
    dataset or stream. This means that a boolean mask is created for
    each feature type that indicates the location corresponding columns
    is in the dataset or stream. features types are universal and can be
    found in any data source and dataset.This enumeration is also useful
    for feature engineering where a certain type of input is required to
    commpute a derivative feature such as MACD or RSI financial
    indicators. Downloaders use the value of feature enums as a filter
    to generate boolean mask for corresponding features. This means that
    downloaders look for columns with name equal to the value of the
    enum, for example 'OPEN' corresponding to ASSET_OPEN_PRICE enum and
    then return a boolean mask that indicates the location of any column
    names in the dataset/stream that matches the name 'OPEN'. Column
    names in Output of downloaders are guaranteed to be consistent with
    this method to create valid boolean masks.


    Attributes:
    --------------
        ASSET_OPEN_PRICE (str): 
            The type of column for opening price of an asset within an
            interval. 
        ASSET_HIGH_PRICE (str): 
            The type of column for high price data of asset within an
            interval.
        ASSET_LOW_PRICE (str): 
            The type of column for low price data.
        ASSET_CLOSE_PRICE (str): 
            The type of column for closing price data. by default
            training environments use this column as the price of asset
            for placing orders at the end of each time interval.
        ASSET_BID_PRICE (str): 
            The type of column for bid price data.
        ASSET_ASK_PRICE (str): 
            The type of column for ask price data.
        SENTIMENT (str): 
            The type of column for sentiment data. usualy a float
            between -1 and 1. 1 meaning very positive and -1 meaning
            very negative. However it is possible to have a different
            range of values, or multiple sentiment columns. 
        EMBEDDING (str): The type of column for word embedding data.
        This is a vector of floats that is the result of a word
        embedding algorithm such as BERT, word2vec or GloVe aggregated 
        over a time interval. 
        TEXT (str): The type of column for text data. This is a string
        that is the result of concatenation of all the text data in a
        time interval.
    
    Notes:
    --------------
        This is not a representative list of all the feature types that
        can be used in datasets and streams. This is just a list of the
        most common feature types that are frequently used in a way that
        needs to be distinguished them from other feature types, for
        example for feature engineering, using OHLC prices, or for
        filtering text columns for passing to large language models.
        This list can be extended to include other feature types that
        are not included here using the FeatureType.MY_FEATURE_TYPE =
        'KEY_WORD' syntax. Feature schema will automatically look for
        columns that contain name 'KEY_WORD' and create a boolean mask
        for them in feature shema.
    
    Examples:
    --------------
    >>> from neural.data.enums import FeatureType
    >>> FeatureType.MY_FEATURE_TYPE = 'KEY_WORD'
    """

    ASSET_OPEN_PRICE = 'OPEN'
    ASSET_HIGH_PRICE = 'HIGH'
    ASSET_LOW_PRICE = 'LOW'
    ASSET_CLOSE_PRICE = 'CLOSE'
    ASSET_BID_PRICE = 'BID'
    ASSET_ASK_PRICE = 'ASK'
    ASSET_TEXT_EMBEDDING = 'EMBEDDING'
    ASSET_SENTIMENT_SCORE = 'SENTIMENT'
    ASSET_TEXT = 'TEXT'


class AssetType(str, Enum):

    """
    An enum class representing different categories of financial
    instruments amenable to high frequency trading. This enum is used to
    standardize the asset type of different financial instruments in 
    the library.

    STOCK: Represents ownership in a publicly traded corporation. Stocks
    can be traded on stock exchanges, and their value can fluctuate
    based on various factors such as the company's financial performance
    and market conditions.

    CURRENCY: Represents a unit of currency, such as the US dollar,
    Euro, or Japanese yen. Currencies can be traded on the foreign
    exchange market (Forex), and their value can fluctuate based on
    various factors such as interest rates, g eopolitical events, and
    market sentiment.

    CRYPTOCURRENCY: Represents a digital or virtual currency that uses
    cryptography for security and operates independently of a central
    bank. Cryptocurrencies can be bought and sold on various
    cryptocurrency exchanges, and their value can fluctuate based on
    various factors such as supply and demand, adoption rates, and
    regulatory developments.

    FUTURES: Represents a standardized contract to buy or sell an
    underlying asset at a predetermined price and date in the future.
    Futures can be traded on various futures exchanges, and their value
    can fluctuate based on various factors such as supply and demand,
    geopolitical events, and market sentiment.

    OPTIONS: Represents a financial contract that gives the buyer the
    right, but not the obligation, to buy or sell an underlying asset at
    a specified price on or before a specified date. Options can be
    traded on various options exchanges, and their value can fluctuate
    based on various factors such as the price of the underlying asset,
    the time until expiration, and market volatility.

    EXCHANGE_TRADED_FUND: Represents a type of investment fund traded on
    stock exchanges, similar to mutual funds, but with shares that can
    be bought and sold like individual stocks. ETFs can provide exposure
    to a wide range of asset classes, such as stocks, bonds, and
    commodities, and their value can fluctuate based on various factors
    such as the performance of the underlying assets and market
    conditions.

    COMMODITY: Represents a physical or virtual product that can be
    bought or sold, such as gold, oil, or currencies. Commodities can be
    traded on various commodity exchanges, and their value can fluctuate
    based on various factors such as supply and demand, geopolitical
    events, and market sentiment.
    """

    STOCK = 'STOCK'
    CURRENCY = 'CURRENCY'
    CRYPTOCURRENCY = 'CRYPTOCURRENCY'
    FUTURES = 'FUTURES'
    OPTIONS = 'OPTIONS'
    EXCHANGE_TRADED_FUND = 'ETF'
    COMMODITY = 'COMMODITY'