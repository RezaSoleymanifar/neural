from enum import Enum
from functools import reduce
from datetime import datetime
from typing import Dict, Tuple
from abc import ABC
from dataclasses import dataclass
import pickle
import pandas as pd

from neural.data.time import Calendar
from neural.common.constants import CALENDAR



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



@dataclass
class AbstractDataMetaData:

    # Data has a universal representation of a two dimensional array of objects throughout the framework. Each row 
    # corresponds to a time interval with a fixed length called resolution. Each column corresponds to a feature of the data for a time interval. 
    # The boolean mask indicates where the columns of the corresponding feature types are located in the data.
    # Lenght of boolean mask is equal to the number columns in the data. Difference between dataset and stream is that
    # dataset is static and can be loaded in memory, while stream is dynamic and can only be accessed in an iterator like
    # fashion, where each iteration returns a new row of data and takes time equal to the resolution of the data.
    # The metadata allows fusion of data from multiple data sources into a coehsive representation. This is useful for
    # market simulations and trading abstracting away the construction of data from the representation of data. Metadata also
    # offers automatic validation and updating of joined or appended data making joining multiple source of data a self-contained process.
    # Note that market environments require price mask to be present for all symbols in the data schema.
    # if this property is violated an error will be raised by the data feeder that uses this metadata before the simulation starts.

    data_schema: Dict[AbstractDataSource.DatasetType: Tuple[str]] | Dict[AbstractDataSource.StreamType: Tuple[str]]
    feature_schema: Dict[FeatureType, Tuple[bool]]
    resolution: str


    @property
    def symbols (self):
        # returns a set of all unique symbols in the data schema.
        symbols = set(reduce(lambda x, y: x + y, self.data_schema.values()))
        return symbols
    

    @property
    def asset_prices_mask(self):

        # returns a mask for the asset close price feature type. This price is
        # used by market environments as the point of reference for placing orders. 
        # when a time interval is over and features are observed the closing price of interval is used to 
        # immediately place orders. The order of price mask matches the order of symbols in the data schema.

        mask_ = [mask for feature_type, mask in self.feature_schema.items() 
            if feature_type == FeatureType.ASSET_CLOSE_PRICE]
        
        mask =  mask_.pop() if mask_ else None
        return mask


    @property
    def valid(self) -> bool:

        # ensures that all symbols have a price mask associated with them.
        # This property can be violated during merging, since some feature types
        # may not have a price mask associated with them, due to not being a price related feature type.
        # However post merging the metadata can validate itself using this property.
        # used by data feeders to validaete input before feeding data to the market environments.

        valid = True if len(self.symbols) == self.asset_prices_mask.count(True) else False
        return valid


    @staticmethod
    def create_feature_schema(data: pd.DataFrame):

        """
        Creates a feature schema dictionary for a given DataFrame, with DataType as keys and boolean masks as values.
        The boolean masks indicate where the columns of the corresponding feature types are located in the data.
        By default downloaders provide downloaded data in a pandas Dataframe format.

        Args:
            data (pd.DataFrame): The input DataFrame for which the feature schema is to be created. By defaulat
            all feature types in FeatureType are enumerated and their value is matched against the column names of the input DataFrame.
            If a column name contains the vluae of a feature type, the corresponding boolean mask is set to True. this process
            is case insensitive. For example if dataframe has the column name 'AAPL_close_price' the boolean mask for FeatureType.ASSET_CLOSE_PRICE
            will be set to True at the position of the column name. Downloaders and streamers should ensure that the column names of the data
            they provide are consistent with this procedure.

        Returns:
            Dict[FeatureType, List[bool]]: A dictionary with FeatureType as keys and boolean masks as values.
        """

        feature_schema = dict()

        for feature_type in FeatureType:
            # if column name contains the feature type value set the corresponding boolean mask to True.
            # this check is case insensitive.
            mask = data.columns.str.lower().str.match('.*'+feature_type.value.lower()+'.*')
            feature_schema[feature_type] = mask

        return feature_schema

    def __or__(self, other, **kwargs):
        
        # i.e. all streams or all datasets
        if not self._validate_data_schema(other.data_schema):

            raise ValueError(
                f'Metadata {other} has feature type {other.feature_types} which is not compatible with {self.feature_types}.')
        
        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        data_schema = self.data_schema.update(other.data_schema)
        feature_schema = self._join_feature_schemas(other)


        return self.__class__(
            data_schema= data_schema,
            feature_schema= feature_schema,
            resolution=self.resolution,
            **kwargs)


    def _validate_data_schema(self, data_schema):

        # checks if all stream or all datasets.
        valid = True
        for data_type in data_schema:
            valid = valid and all(
                data_type == data_type_ for data_type_ in self.data_schema)
            
        return True
    
    def _join_feature_schemas(self, other):

        if set(self.feature_schema.keys()) != set(other.data_schema.keys()):

            raise ValueError(
                'Datasets do not have matching feature schemas.')

        merged_schema = dict()

        for key in self.feature_schema.keys():
            
            merged_schema[key] = self.feature_schema[key] + other.data_schema[key]

        return merged_schema
    


@dataclass
class StreamMetaData(AbstractDataMetaData):

    data_schema: Dict[AbstractDataSource.StreamType: Tuple[str]]
    feature_schema: Dict[FeatureType, Tuple[bool]]
    resolution: str




@dataclass
class DatasetMetadata(AbstractDataMetaData):

    data_schema: Dict[AbstractDataSource.DatasetType: Tuple[str]]
    feature_schema: Dict[FeatureType, Tuple[bool]]
    resolution: str
    resolution: str
    start: datetime
    end: datetime
    n_rows: int


    @property
    def n_columns(self) -> int:

        """
        Returns the number of columns in the dataset based on the feature_schema attribute.
        """

        return len(next(iter(self.feature_schema.values())))

        
    def __or__(self, other: AbstractDataMetaData) -> AbstractDataMetaData:

        # This is useful for joining datasets that are large to download in one go. Each sub-dataset
        # is downloaded for a fixed time interval and each can correponds to differnt data sources, feature types
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


    def __add__(self, other, **kwargs):

        # stream metadata child cannot use this method. appending stream metadata would not make sense.
        # if used with stream metadata it will raise a not implemented error.


        # ensures data schemas are identical.
        if pickle.dumps(self.data_schema) != pickle.dumps(other.data_schema):
            raise ValueError(f'Datasets must have identical data schemas.')
           
        # ensures data schemas are identical.
        if pickle.dumps(self.feature_schema) != pickle.dumps(other.feature_schema):
            raise ValueError(f'Datasets must have identical feature schemas.')

        if self.resolution != other.resolution:
            raise ValueError(
                f'Dataset resolutions{self.resolution} and {other.resolution} are mismatched.')
        
        if not self.calendar_type != other.calendar_type:

            raise ValueError(
                f'Metadata {other} has calendar type {other.calendar_type} which is not compatible with {self.calendar_type}.')

        if not self._check_dates(prev_end=self.end, cur_start=other.start):

            raise ValueError(
                f'Non-consecutive market days between end time {self.end} and {self.start}.')


        data_schema = self._join_feature_schemas(other)

        return self._class__(
            feature_types=self.feature_types,
            data_schema=data_schema,
            symbols=self.symbols,
            resolution=self.resolution,
            **kwargs)

    def __add__(self, other: AbstractDataMetaData) -> AbstractDataMetaData:

        # this is useful for appending datasets that are large to downolad in one go.
        # At each iteration the user can download the data in chunks corresponding to 
        # a fixed time interval shared between all other chunks and automatically validate 
        # the process and update the metadata. For example downloading tradde data for 
        # S&P500 stocks for a fixed time interval can happens by downloading the data for
        # a list of symbols at a time.

        return super().__add__(other, start=self.start, end=other.end, n_rows=self.n_rows)


    
    def _check_dates(self, prev_end, cur_start):

        """
        Checks if two dates are consecutive market days. This is used to validate the process of
        appending datasets to ensure temporal continuity of the data.
        """

        start_date = prev_end.date()
        end_date = cur_start.date()

        calendar = Calendar(self.calendar_type)
        schedule = Calendar.schedule(start_date= start_date, end_date= end_date)
        
        return True if len(schedule) == 2 else False
    
    @property
    def stream(self):
        stream = StreamMetaData(
            stream_types=self.dataset_types)