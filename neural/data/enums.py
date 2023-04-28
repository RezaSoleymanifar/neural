from enum import Enum
from datetime import datetime
from typing import List, Dict, Tuple
from abc import ABC
from dataclasses import dataclass
import pickle

from neural.tools.base import Calendar


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


    @property
    def calendar(self):
        return AssetType.calendar_map[self]
    

@dataclass
class Symbol:

    """
    A dataclass that represents a symbol of an asset.
    """

    symbol: str
    asset_name: str
    asset_type: AssetType
    CalendarType: CalendarType



class DataType(Enum):

    """
    Enumeration class that defines constants for the different types of data in datasets.
    Typically used to define the column schema of a dataset. This means that a boolean mask
    is created for each data type that indicates the location corresponding columns is in the dataset.
    Data types are universal and can be used in any data source and dataset. Data is unifromly
    formatted in a way that each row corresponds to a time interval with legnth equal to the resolution
    specified by the user. This enumeration is also useful for feature engineering where a certain
    type of data is required to commpute a feature such as MACD or RSI financial indicators.
    Downloaders use the value of enums as a filter to grab boolean mask for corresponding columns.
    This means that dowlnloaders look for columns with name equal to the value of the enum, for example 'OPEN'
    and then return a boolean mask that indicates the location of any column names in the dataset that matches the
    name 'OPEN'. Column names in Output of downloaders are guaranteed to be consistent with this method to create valid 
    boolean masks.


    Attributes:
    --------------
        ASSET_OPEN_PRICE (str): The type of column for opening price data.
        ASSET_HIGH_PRICE (str): The type of column for high price data.
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
        This is not a representative list of all the data types that can be used in datasets.
        This is just a list of the most common data types that are frequently used in a way that
        needs to be distinguished them from other data types, for example for feature engineering, 
        using OHLC prices, or for filtering text columns for passing to large language models.
        This list can be extended to include other data types that are not included here.
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


class AbstractDataSource(ABC):

    """
    Abstract base class for a data source that provides access to time-series data.

    This class defines a set of nested enums for specifying the available dataset and stream types
    for the data source. These enums can be used to standardize the types of data that can be accessed
    from the source and to ensure consistency in data handling and processing.

    Attributes:
    -----------
    DatasetType : Enum
        Enumeration class that defines the available dataset types for the data source. This can include
        types such as stock prices, weather data, social media data, etc. Dataset types are used to organize
        the data and to ensure that consistent data processing and analysis methods are used across datasets.

    StreamType : Enum
    ----------------
        Enumeration class that defines the available stream types for the data source. This can include types
        such as tick data, volume data, order book data, tweets etc. Stream types are used to stream live data for 
        high frequency trading. Usually an algorithm is trained usinsg a static data and it's dataset metadata is
        mappped to a stream type for streaming and aggregating the type of data that was used to train the algorithm.
        Any dataset type should logically have a corresponding stream type, otherwise trained agent will not be deployable
        in a live trading environment.
    
    Methods:
    -----------
        stream(dataset_type: DatasetType) -> StreamType
            Returns the stream type corresponding to the specified dataset type. By default maps 
            to corresponding stream type using the value of the dataset type enum. This method can be
            overriden to provide custom mapping between dataset and stream types.
    """ 

    class DatasetType(Enum):

        @property
        def data_source(self):

             # recovers data source from the outer scope.
            self._data_source = globals()[self.__class__.__qualname__.split('.')[0]]
             
            return self._data_source
        

        @property
        def stream(self):

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

        dataset_to_stream_map = {
            dataset_type: cls.StreamType(dataset_type.value) for dataset_type in cls.DatasetType}

        try:
            stream_type = dataset_to_stream_map[dataset_type]

        except ValueError:
            raise ValueError(f"Corresponding stream type of {dataset_type} is not defined.")
        
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


        TRADE = 'TRADE'
        QUOTE = 'QUOTE'
        ORDER_BOOK = 'ORDER_BOOK'



class AbstractDataMetaData:

    # Data has a universal representation a two dimensional array of objects throughout the framework. Each row 
    # corresponds to a time interval with a fixed length called resolution. Each column corresponds to a feature of the data for a time interval. 
    # The boolean mask indicates where the columns of the corresponding data types are located in the data.
    # Lenght of boolean mask is equal to the number columns in the data. Difference between dataset and stream is that
    # dataset is static and can be loaded in memory, while stream is dynamic and can only be accessed in an iterator like
    # fashion, where each iteration returns a new row of data and takes time equal to the resolution of the data.
    # The metadata allows fusion of data from multiple data sources into a coehsive representation. This is useful for
    # market simulations and trading abstracting away the construction of data from the receiver of data. Metadata also
    # automatic validation and updating of joined or appended data making data construction a self-contained process.

    data_types: List[AbstractDataSource.DatasetType | AbstractDataSource.StreamType]
    data_schema: Dict[DataType, Tuple[bool]]
    symbols: Tuple[Symbol]
    resolution: str

    def __post_init__(self):

        assert isinstance(self.data_types, list), "dataset_types argument must be a list of data types."


    def __or__(self, other, **kwargs):

        if not self._validate_data_types(other.data_types):

            raise ValueError(
                f'Metadata {other} has data type {other.data_types} which is not compatible with {self.data_types}.')

        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        data_types = self.data_types + other.data_types
        data_schema = self._join_data_schemas(other)


        return self.__class__(
            dataset_type=data_types,
            data_schema=data_schema,
            symbols=self.symbols,
            resolution=self.resolution,
            **kwargs)



    def __add__(self, other, **kwargs):

        # stream metadata child cannot use this method. appending stream metadata would not make sense.
        # only 
        if pickle.dumps(self.data_schema) != pickle.dumps(other.column_schema):
            raise ValueError(f'Datasets must have identical data schemas.')

        if self.symbols != other.symbols:
            raise ValueError('Datasets must have the same symbols.')

        if self.resolution != other.resolution:
            raise ValueError(
                f'Dataset resolutions{self.resolution} and {other.resolution} are mismatched.')

        data_schema = self._join_data_schemas(other)

        return self._class__(
            data_types=self.data_types,
            data_schema=data_schema,
            symbols=self.symbols,
            resolution=self.resolution,
            **kwargs)


    def _validate_data_types(self, data_types):

        valid = True
        for data_type in data_types:
            valid = valid and all(
                data_type_ == data_type for data_type_ in self.data_types)
            
        return True
    
    def _join_data_schemas(self, other):

        if set(self.data_schema.keys()) != set(other.data_schema.keys()):

            raise ValueError(
                'Datasets do not have matching data schemas.')

        merged_schema = dict()

        for key in self.data_schema.keys():
            
            merged_schema[key] = self.data_schema[key] + other.data_schema[key]

        return merged_schema
    


@dataclass
class StreamMetaData(AbstractDataMetaData):


    stream_type: List[AbstractDataSource.StreamType]
    column_schema: Dict[DataType, Tuple[bool]]
    symbols: Tuple[Symbol]
    resolution: str

    def __add__(self, other, **kwargs):
        raise NotImplementedError('Stream metadata cannot be appended.')


@dataclass
class DatasetMetadata(AbstractDataMetaData):
    dataset_type: List[AbstractDataSource.DatasetType]
    column_schema: Dict[DataType, Tuple[bool]]
    symbols: Tuple[Symbol]
    resolution: str
    start: datetime
    end: datetime
    n_rows: int


    @property
    def n_columns(self) -> int:

        """
        Returns the number of columns in the dataset based on the column_schema attribute.
        """

        return len(next(iter(self.column_schema.values())))


    def __or__(self, other: AbstractDataMetaData) -> AbstractDataMetaData:

        # This is useful for joining datasets that are large to download in one go. Each sub-dataset
        # is downloaded for a fixed time interval and each can correponds to differnt data sources, data types
        # and symbols. Joining datasets and validating the process is done automatically using this method.

        # check symbols 

        if self.start != other.start:
            raise ValueError(
                f'Current start time: {self.start}, does not match joined dataset start time {other.start}.')
        
        if self.end != other.end:
            raise ValueError(
                f'Current end time: {self.end}, does not match joined dataset end time {other.end}.')
        
        if self.n_rows != other.n_rows:
            raise ValueError('Datasets must have the same number of rows.')
        
        return super().__or__(other, start=self.start, end=self.end, n_rows=self.n_rows)


    def __add__(self, other: DatasetMetadata) -> DatasetMetadata:

        # this is useful for appending datasets that are large to downolad in one go.
        # At each iteration the user can download the data in chunks corresponding to 
        # a fixed time interval shared between all other chunks and automatically validate 
        # the process and update the metadata. For example downloading tradde data for 
        # S&P500 stocks for a fixed time interval can happens by downloading the data for
        # a list of symbols at a time.

        if not self._check_dates(prev_end = self.end, cur_start = other.start):

            raise ValueError(
                f'Non-consecutive market days between end time {self.end} and {self.start}.')

        return super().__add__(other, start=self.start, end=other.end, n_rows=self.n_rows)


    
    def _check_dates(self, prev_end, cur_start):

        """
        Checks if two dates are consecutive market days. This is used to validate the process of
        appending datasets to  ensure temporal continuity of the data.
        """

        asset_type = self.symbols[0].asset_type
        start_date = prev_end.date()
        end_date = cur_start.date()

        calendar = Calendar(asset_type)
        schedule = calendar.schedule(start_date= start_date, end_date= end_date)
        
        return True if len(schedule) == 2 else False