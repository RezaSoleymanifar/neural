from enum import Enum
from neural.common.constants import CALENDAR
from abc import ABC
from neural.client.alpaca import AlpacaDataClient


class AbstractDataSource(ABC):

    """
    Abstract base class for a data source that standardizes the interface for accessing data from different sources.
    A data source is typically an API or a database that provides access to data. This class defines the interface
    for accessing data from a data source. The data source can either provide a static data namely dataset that 
    aggreates old data streams useful for training. Or it can provide a live stream of data that is used for high frequency trading.
    It also defines a set of nested enums for standardizing the available dataset and stream types.

    Attributes:
    -----------
    DatasetType : Enum
        Enumeration class that defines the available dataset types for the data source. This can include
        types such as stock prices, weather data, social media data, etc. Dataset types are used to organize
        the data and to ensure that consistent data processing methods are used across datasets.

    StreamType : Enum
    ----------------
        Enumeration class that defines the available stream types for the data source. This can include types
        such as tick data, volume data, order book data, tweets etc. Stream types are used to stream live data for 
        high frequency trading. Usually an algorithm is trained usinsg a static data and it's dataset metadata is
        mappped to a stream type for streaming and aggregating the type of data that was used to train the agent on.
        Any dataset type should logically have a corresponding stream type, otherwise trained agent will not be deployable
        in a live trading environment.
    
    Methods:
    -----------
        stream(dataset_type: DatasetType) -> StreamType
            Returns the stream type corresponding to the specified dataset type. By default maps 
            to corresponding stream type using the value of the dataset type enum. This means a
            dataset type with value 'STOCK' will be mapped to a stream type with value 'STOCK'.
            This behavior can be overriden to provide custom mapping between dataset and stream types.

    Example:
    -----------
        >>> class MyDataSource(AbstractDataSource):
        ...     class DatasetType(Enum):
        ...         MY_DATASET = 'MY_DATASET'
        ...     class StreamType(Enum):
        ...         MY_STREAM = 'MY_STREAM'
        ...     def stream(self, dataset_type):
        ...         # your custom stream mapping logic here.
        ...         ...
    """ 

    class DatasetType(Enum):

        @property
        def data_source(self):

             # returns pointer to data source in the outer scope.
            self._data_source = globals()[self.__class__.__qualname__.split('.')[0]]
             
            return self._data_source
        

        @property
        def stream(self):
            # uses the stream implementation of data source to map dataset type to stream type.
            return self.data_source.stream(self)


        def __eq__(self, other):

            # convenience method to check if other is a dataset type.
            return isinstance(other, AbstractDataSource.DatasetType)


    class StreamType(Enum):

        # convenience method to check if other is a stream type.
        @property
        def data_source(self):

            # recovers data source from the outer scope.
            self._data_source = globals()[self.__class__.__qualname__.split('.')[0]]
             
            return self._data_source
        

        def __eq__(self, other):

            # convenience method to check if other is a stream type.
            return isinstance(other, AbstractDataSource.StreamType)


    @classmethod
    def stream(cls, dataset_type: DatasetType) -> StreamType:

        """
        Returns a StreamType enum member corresponding to the given DatasetType enum member.
        If a different behavior is intended subclasses can override this method to provide custom
        mapping between dataset and stream types.

        Args:
        ---------
            dataset_type (DatasetType): A member of the DatasetType enum that represents the type of dataset.

        Returns:
        --------
            StreamType: A member of the StreamType enum that corresponds to the given dataset_type.

        Raises:
        ---
            ValueError: If dataset_type is not a valid member of the DatasetType enum, 
            or if there is no corresponding member of the StreamType enum.

        Notes:
        ------
            The other direction of mapping from stream to dataset type is not valid because
            there can be multiple dataset types that can be mapped to the same stream type.
        """

        stream_map = {
            dataset_type: cls.StreamType(dataset_type.value) for dataset_type in cls.DatasetType}

        try:
            stream_type = stream_map[dataset_type]

        except KeyError:
            raise KeyError(f"Corresponding stream type of {dataset_type} is not found.")
        
        return stream_type




class AlpacaDataSource(AbstractDataSource):


    class DatasetType(AbstractDataSource.DatasetType):

        """
        Enumeration class that defines constants for the different types of datasets.

        Attributes
        ----------
        TRADE : str
            The type of dataset for aggregated trade stream data. Also known as bars data.
        QUOTE : str
            The type of dataset for aggregated quote stream.
        ORDER_BOOK : str
            The type of dataset for aggregated order book data.
        """
        BAR = 'BAR'
        TRADE = 'TRADE'
        QUOTE = 'QUOTE'
        ORDER_BOOK = 'ORDER_BOOK'

    class StreamType(AbstractDataSource.StreamType):

        """
        Enumeration class that defines constants for the different types of data streams.

        Attributes
        ----------
        QUOTE : str
            The type of data stream for quotes.
        TRADE : str
            The type of data stream for trades.
        ORDER_BOOK : str
            The type of data stream for order book data.
        """
        BAR = 'BAR'
        TRADE = 'TRADE'
        QUOTE = 'QUOTE'


class CalendarType(Enum):

    """

    If an asset does not fall under these three categories it can be handled by user speciying the
    CalendarType.MY_CALENDAR_TYPE = 'VALID_PANDAS_CALENDAR' and providing a valid pandas_market_calendars
    calendar. This is the calendar used by default and CelndarType enum values correponds to valid
    strings for the pandas_market_calendars get_calendar() method.
    More information here: https://pandas-market-calendars.readthedocs.io/en/latest/modules.html.

    Examples:
    ----------
    >>> CalendarType.MY_CALENDAR_TYPE = 'VALID_PANDAS_CALENDAR'
    """

    NEW_YORK_STOCK_EXCHANGE = 'NYSE'
    TWENTY_FOUR_SEVEN = '24/7'
    TWENTY_FOUR_FIVE = '24/5'
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
        return CALENDAR(self.value).schedule


class AssetType(Enum):

    """
    Enum class that defines the type of asset. Note that supported calendars 
    are 
    """

    STOCK = 'STOCK'
    ETF = 'ETF'
    CRYPTO = 'CRYPTO'
    CURRENCY = 'CURRENCY'
    COMMODITY = 'COMMODITY'
    BOND = 'BOND'
    INDEX = 'INDEX'
    FUTURE = 'FUTURE'
    OPTION = 'OPTION'
    FUND = 'FUND'


class FeatureType(Enum):

    """
    Enumeration class that defines constants for the different types of features in data.
    Typically used to define the feature schema of a dataset or stream. This means that a boolean mask
    is created for each feature type that indicates the location corresponding columns is in the dataset or stream.
    features types are universal and can be found in any data source and dataset.This enumeration is also useful for 
    feature engineering where a certain type of input is required to commpute a derivative feature such as MACD or RSI financial indicators.
    Downloaders use the value of feature enums as a filter to generate boolean mask for corresponding features.
    This means that dowlnloaders look for columns with name equal to the value of the enum, for example 'OPEN'
    corresponding to ASSET_OPEN_PRICE enum and then return a boolean mask that indicates the location of any 
    column names in the dataset/stream that matches the name 'OPEN'. Column names in Output of downloaders are 
    guaranteed to be consistent with this method to create valid boolean masks.


    Attributes:
    --------------
        ASSET_OPEN_PRICE (str): The type of column for opening price of an asset within an interval.
        ASSET_HIGH_PRICE (str): The type of column for high price data of asset within an interval.
        ASSET_LOW_PRICE (str): The type of column for low price data.
        ASSET_CLOSE_PRICE (str): The type of column for closing price data. by default training 
            environments use this column as the price of asset for placing orders at the end of each time interval.
        ASSET_BID_PRICE (str): The type of column for bid price data.
        ASSET_ASK_PRICE (str): The type of column for ask price data.
        SENTIMENT (str): The type of column for sentiment data. usualy a float between -1 and 1.
        1 meaning very positive and -1 meaning very negative. However it is possible to have
        a different range of values, or multiple sentiment columns.
        EMBEDDING (str): The type of column for word embedding data. This is a vector of floats
        that is the result of a word embedding algorithm such as word2vec or GloVe aggregated
        over a time interval.
        TEXT (str): The type of column for text data. This is a string that is the result of
        concatenation of all the text data in a time interval.
    
    Notes:
    --------------
        This is not a representative list of all the feature types that can be used in datasets and streams.
        This is just a list of the most common feature types that are frequently used in a way that
        needs to be distinguished them from other feature types, for example for feature engineering, 
        using OHLC prices, or for filtering text columns for passing to large language models.
        This list can be extended to include other feature types that are not included here using
        the FeatureType.MY_FEATURE_TYPE = 'KEY_WORD' syntax. Feature schema will automatically
        look for columns that contain name 'KEY_WORD' and create a boolean mask for them in feature shema.
    """

    ASSET_OPEN_PRICE = 'OPEN'
    ASSET_HIGH_PRICE = 'HIGH'
    ASSET_LOW_PRICE = 'LOW'
    ASSET_CLOSE_PRICE = 'CLOSE'
    ASSET_BID_PRICE = 'BID'
    ASSET_ASK_PRICE = 'ASK'
    EMBEDDING = 'EMBEDDING'
    SENTIMENT = 'SENTIMENT'
    TEXT = 'TEXT'
