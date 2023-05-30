"""
base.py

Base data module for defining data sources, data feeders and data
metadata. Data sources are responsible for providing access to data from
different sources. Data feeders are responsible for feeding data to
market environments. Data metadata is responsible for providing
information about the data such as the data schema, feature schema,
resolution, calendar type, etc. Data metadata is also responsible for
validating the data and ensuring that the data is consistent with the
metadata. Data metadata is also responsible for joining and appending
datasets and streams. This is useful for downloading large datasets in
chunks and validating the process. Data metadata is also responsible for
providing a universal representation of data throughout the framework.
This is useful for abstracting away the construction of data from the
representation of data. Data metadata also provides a consistent
interface for accessing data from different sources.

Classes:
--------
    AbstractDataSource:
        Abstract base class for a data source that standardizes the
        interface for accessing data from different sources. A data
        source is typically an API or a database that provides access to
    
    AlpacaDataSource:
        Represents Alpaca API as a data source. Provides standardized
        enums for historical and live data from Alpaca API.
    
    AbstractDataMetaData: abstract base class for data metadata that
        provides a universal representation of data throughout the
        framework. This is useful for abstracting away the construction
        of data from the representation of data. Data metadata also
        provides a consistent interface for accessing data from
        different sources.
    
    StreamMetaData:
        Subclass of AbstractDataMetaData that provides metadata for
        streaming data.
    
    DatasetMetadata:
        Subclass of AbstractDataMetaData that provides metadata for
        static data.
    
    AbstractDataFeeder:
        Abstract base class for defining a data feeder that is
        responsible for feeding data to a market environment,
        iteratively. A data feeder can feed data in a static or
        asynchronous manner. A static data feeder is responsible for
        feeding data from a static source, such as a HDF5 file, while an
        asynchronous data feeder is responsible for feeding data from a
        live stream.
    
    StaticDataFeeder:
        Subclass of AbstractStaticDataFeeder that iteratively returns
        data required for the environment from a static source.
    
    AbstractAsyncDataFeeder: abstract base class for defining an
        asynchronous data feeder that is responsible for feeding data
        from a live stream to a market environment.
    
    AlpacaDataFeeder: subclass of AbstractAsyncDataFeeder that feeds
        data from Alpaca API to a market environment.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import reduce
from typing import Dict, Tuple, Iterable, Optional, List

import dill
import h5py as h5
import numpy as np
import pandas as pd

from neural.client.base import AbstractDataClient
from neural.data.enums import FeatureType, AssetType, CalendarType


@dataclass(frozen=True)
class AbstractAsset(ABC):
    """
    A generic financial asset. This class standardizes the
    representation of assets throughout the framework.

    Attributes:
    -----------
        symbol: str
            A string representing the symbol or ticker of the asset.
        asset_type: AssetType
            An instance of the `AssetType` enum class representing the
            type of asset.
        fractionable: bool
            A boolean indicating whether the asset can be traded in
            fractional shares. This is useful for trading for example
            cryptocurrencies or stocks that are expensive to buy as a
            whole share.
    """
    symbol: str
    asset_type: AssetType
    fractionable: bool


@dataclass(frozen=True)
class AlpacaAsset(AbstractAsset):
    """
    A dataclass representing a financial asset in Alpaca API:
    https://alpaca.markets/. This can be a stock, or cryptocurrency.
    This class standardizes the representation of assets in Alpaca API.
    Natively encodes the mechanics for opening and closing positions.
    When creating nonmarginable assets, maintenance_margin, shortable,
    and easy_to_borrow attributes are set to None by default.

    Attributes:
    ----------
        symbol: str
            A string representing the symbol or ticker of the asset.
        asset_type: AssetType
            An instance of the `AssetType` enum class representing the
            type of asset.
        fractionable: bool
            A boolean indicating whether the asset can be traded in
            fractional shares. This is useful for trading for example
            cryptocurrencies or stocks that are expensive to buy as a
            whole share.
        marginable: bool
            A boolean indicating whether the asset is a marginable
            asset. Marginable assets can be used as collateral for
            margin trading. Margin trading is a process where the
            brokerage lends money to the trader to buy more assets than
            the trader can afford. More info here:
            https://www.investopedia.com/terms/m/margin.asp.
        shortable: bool
            A boolean indicating whether the asset can be sold short.
            When asset is sold short you sell the asset when you do not
            own it. This way an asset debt is recorded in your account.
            You can then buy the asset at a lower price and return it to
            the brokerage. This form of trading allows making profit in
            a bear market. More info here:
            https://www.investopedia.com/terms/s/shortselling.asp.
        easy_to_borrow: bool
            A boolean indicating whether the asset can be borrowed
            easily. Alpaca API has restrictive rules for hard to borrow
            assets and in general HTB assets cannot be shorted.
        maintenance_margin: float | None
            A float representing the maintenance margin of the asset.
            This means that maintenace_margin * position_value should be
            available in marginable equity at all times. In practice the
            gross maintenance margin for entire portfolio is used to
            measure maintenance margin requirement. If maintenance
            margin requirement is violated the brokerage will issue a
            margin call. More info here:
            https://www.investopedia.com/terms/m/maintenance_margin.asp.
            Alpaca API in reality enforces this at the end of day or
            when it is violated by a greate extent.

    Properties:
    -----------
        shortable: bool | None
            A boolean indicating whether the asset can be sold short
            (i.e., sold before buying to profit from a price decrease).
        easy_to_borrow: bool | None
            A boolean indicating whether the asset can be borrowed
            easily.

    Methods:
    --------
        get_initial_margin(self, short: bool = False) -> float | None
            A float representing the initial margin of the asset.
        get_maintenance_margin(self, price: float, short: bool = False)
            -> float | None A float representing the maintenance margin
            of the asset.
    Notes:
    ------
        The easy_to_borrow, intial_margin, and maintenance_margin
        properties are only valid for assets that are marginable. For
        example, cryptocurrencies are not marginable and therefore do
        not need to set these attributes. For consistency for
        nonmarginable assets default boolean valued attributes are
        returned as False and maintenance margin and initial margin are
        returned as 0 and 1 respectively. nonmarginable assets can only
        be purchased using cash and we assume they cannot be shorted.
        There are rare cases where non-marginable assets can be shorted,
        but this is not supported by this library due to the complexity
        of the process. A mix of marginable and non-marginable assets in
        portoflio is not supporeted either due to the same level of
        irregularities.
    """

    marginable: bool
    maintenance_margin: Optional[float] = None
    shortable: Optional[bool] = None
    easy_to_borrow: Optional[bool] = None

    @property
    def shortable(self) -> bool | None:
        """
        A boolean indicating whether the asset can be sold short (i.e.,
        sold before buying to profit from a price decrease). In Alpaca
        API shorted assets cannot have faractional quantities. Also if
        asset is not marginable it cannot be shorted. There are rare
        cases where non-marginable assets can be shorted, but this is
        not supported by this library due to the complexity of the
        process.

        Returns:
        --------
            bool:
                A boolean indicating whether the asset can be sold
                short.
        """
        return self.shortable if self.marginable else False

    @property
    def easy_to_borrow(self) -> bool | None:
        """
        A boolean indicating whether the asset can be borrowed easily.
        Alpaca API has restrictive rules for hard to borrow assets and
        in general HTB assets cannot be shorted. This library only
        allows easy to borrow assets to be shorted.

        Returns:
        --------
            bool:
                A boolean indicating whether the asset can be borrowed
                easily.
        """
        return self.easy_to_borrow if self.marginable else False

    def get_initial_margin(self, short: bool = False) -> float | None:
        """
        A float representing the initial margin of the asset. 25% margin
        for long positions and 150% margin for short positions is a
        FINRA requirement:
        https://www.finra.org/filing-reporting/regulation-t-filings.
        Alpaca API has a 50% margin requirement for opening positions,
        by default. Initial margin for nonmarginable assets is 1.0
        namely entire value of trade needs to be available in cash.
        Since nonmarginable assets cannot be margined this is an abuse
        of terminalogy to provide a convenient interface for working
        with marginable and onnmarginable assets in a consistent way.

        Args:
        -----
            short (bool):
                A boolean indicating whether the initial margin is for a
                short position. By default False. If asset is
                nonmarginable returns 1.0.
        
        Returns:
        --------
            float:
                A float representing the initial margin of the asset.

        TODO:
        -----
            Investigate why Alpaca API does not follow FINRA 150% margin
            requirement for short positions. This still works since
            Alpaca requires 50% initial margin for short positions.
            Reducing this to 50% can unlock more leverage for short
            positions.
        """
        if not self.marginable:
            return 1
        elif not short:
            return 0.5
        elif short:
            return 1.5

    def get_maintenance_margin(self,
                               price: float,
                               short: bool = False) -> float | None:
        """
        A float representing the maintenance margin of the asset. This
        means that maintenace_margin * position_value should be
        available in marginable equity. Maintenance margin is cumulative
        for all assets and needs to be satisfied at all times. Alpaca
        API in reality enforces this at the end of day or when it is
        violated by a greate amount intraday. We enforce this at all
        times in a conservative manner. Maintenance margin for
        nonmarginable assets is 0. Since nonmarginable assets cannot be
        margined this is an abuse of terminalogy to provide a convenient
        interface for working with marginable and onnmarginable assets.
        
        Default maintenance marigin is the maintenance margin that
        Alpaca API reports by default. maintenance margin attribute is
        the value received from Alpaca API, however it is subject to
        change given price change and position change. Thus taking max
        of default maintenace margin and maintenance margin attribute
        ensures that the most conservative value is used for calculating
        the maintenance margin. Because Alpaca API checks both initial
        and maintenance margin at the end of day, we set the maintenance
        margin to be the maximum of the two. This is not a common
        behavior since typically initial margin is used for opening
        positions and maintenance margin is used for maintaining
        positions. However Alpaca API enforces both at end of day. In
        the end maximum of default margin, initial margin, and
        maintenance margin is used for final calculation of the
        maintenance margin.

        Args:
        -----
            price (float):
                A float representing the price of the asset.
            short (bool):
                A boolean indicating whether the maintenance margin is
                for a short position. By default False. 
        
        Returns:
        --------
            float:
                A float representing the maintenance margin of the
                asset.
        """

        def default_maintenance_margin(price: float,
                                       short: bool = False) -> float:
            """
            Link: https://alpaca.markets/docs/introduction/. The
            maintenance margin is by default calculated based on the
            following table:

            | Pos | Cond      | Margin Req        |
            | --- | --------- | ----------------- |
            | L   | SP < $2.5 | 100% of EOD MV    |
            | L   | SP >= $2.5| 30% of EOD MV     |
            | L   | 2x ETF    | 100% of EOD MV    |
            | L   | 3x ETF    | 100% of EOD MV    |
            | S   | SP < $5.0 | Max $2.50/S or 100% |
            | S   | SP >= $5.0| Max $5.00/S or 30% |

            where SP is the stock price and ETF is the exchange traded
            fund. L and S are long and short positions respectively. EOD
            MV is the end of day market value of the position.

            TODO: Add support for 2x and 3x ETFs. Currently there is no
            way in API to distinguish between ETFs and stocks. This
            means that 2x and 3x ETFs are treated as stocks.

            Args:
            -----
                price (float):
                    A float representing the price of the asset.
                short (bool):
                    A boolean indicating whether the maintenance margin
                    is for a short position. By default False.
            """

            if not self.marginable:
                return 0

            elif not short:
                if price >= 2.50:
                    return 0.3
                else:
                    return 1.0

            elif short:
                if price < 5.00:
                    return max(2.5 / price, 1.0)
                elif price >= 5.00:
                    return max(5.0 / price, 0.3)
                
        required_margin = max(self.maintenance_margin,
                              default_maintenance_margin(price, short),
                              self.get_initial_margin(short))
        return required_margin


class AbstractDataSource(ABC):
    """
    Abstract base class for a data source that standardizes the interface for
    accessing data from different sources. A data source is typically an API or
    a database that provides access to data. This class defines the interface
    for accessing data from a data source. The data source can either provide a
    static data namely dataset that aggreates old data streams useful for
    training. Or it can provide a live stream of data that is used for high
    frequency trading. It also defines a set of nested enums for standardizing
    the available dataset and stream types.

    Attributes:
    -----------
        DatasetType : Enum
            Enumeration class that defines the available historical dataset
            types for the data source. This can include types such as stock
            prices, weather data, social media data, etc. Dataset types are
            used to organize the types of datasets that are available for
            training. Since these types are source dependent, they are defined
            in the data source class.

        StreamType : Enum
            Enumeration class that defines the available stream types for the
            data source. This can include types such as price data, volume data,
            order book data, tweets etc. Stream types are used to stream live
            data for high frequency trading. Usually an algorithm is trained
            usinsg a static historical dataset and it's dataset metadata is
            mappped to a stream type for streaming and aggregating the type of
            data that was used to train the agent on. Any dataset type should
            logically have a corresponding stream type, otherwise agent
            will not be deployable in a live trading environment, if trained
            on that dataset type.
    
    Methods:
    -----------
        stream(dataset_type: DatasetType) -> StreamType
            Returns the stream type corresponding to the specified dataset
            type. By default maps to corresponding stream type using the value
            of the dataset type enum. This means a dataset type with value
            'STOCK' will be mapped to a stream type with value 'STOCK'. This
            behavior can be overriden to provide custom mapping between dataset
            and stream types.

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
        """
        Enumeration class that defines constants for the different types
        of datasets.

        Attributes:
        -----------
            data_source:
                A property that returns a pointer to the data source
                class in the outer scope. This allows the enumeration
                constants to access the data source's methods and
                properties.
            stream:
                A property that uses the stream implementation of the
                data source to map the dataset type to a stream type.
                This is useful for obtaining a stream of data for a
                particular dataset type.
        
        Methods:
        --------
            __eq__(self, other):
                Convenience method to check if other is a dataset type.
        """

        @property
        def data_source(self):
            """
            Returns a pointer to the data source class in the outer
            scope. This allows the enumeration constants to access the
            data source's methods and properties.

            Returns:
            --------
                AbstractDataSource: 
                    A pointer to the data source class in the outer
                    scope.
            """
            data_source = globals()[self.__class__.__qualname__.split(
                '.', maxsplit=1)[0]]

            return data_source

        @property
        def stream(self):
            """
            Uses the stream implementation of the data source to map the
            dataset type to a stream type. This is useful for obtaining
            a stream of data for a particular dataset type.

            Returns:
            --------
                AbstractDataSource.StreamType:
                    A stream type corresponding to the dataset type.
            """
            return self.data_source.stream(self)

        def __eq__(self, other):
            """
            convenience method to check if other is a dataset type.
            """

            return isinstance(other, AbstractDataSource.DatasetType)

    class StreamType(Enum):
        """
        Enumeration class that defines constants for the different types
        of data streams.

        Attributes:
        ----------
            data_source:
                A property that returns a pointer to the data source
                class in the outer scope.
        Methods:
        --------
            __eq__(self, other):    
                Convenience method to check if another object is a
                stream type.
        """

        @property
        def data_source(self):
            """
            Returns a pointer to the data source class in the outer
            scope.
            
            Returns:
            --------
                AbstractDataSource:
                    A pointer to the data source class in the outer
                    scope.
            """
            data_source = globals()[self.__class__.__qualname__.split(
                '.', maxsplit=1)[0]]

            return data_source

        def __eq__(self, other):
            """
            Convenience method to check if another object is a stream
            type.
            
            Args:
            ------
                other (object):
                    The object to be compared with the current stream
                    type.
            
            Returns:
            --------
                bool:
                    True if the other object is a stream type, False
                    otherwise.
            """
            return isinstance(other, AbstractDataSource.StreamType)

    @classmethod
    def stream(cls, dataset_type: DatasetType) -> StreamType:
        """
        Returns a StreamType enum member corresponding to the given
        DatasetType enum member. If a different behavior is intended
        subclasses can override this method to provide custom mapping
        between dataset and stream types.

        Args:
        ---------
            dataset_type (DatasetType): 
                A member of the DatasetType enum that represents the
                type of dataset.

        Returns:
        --------
            StreamType: 
                A member of the StreamType enum that corresponds to the
                given dataset_type.

        Raises: 
        -------
            ValueError: If dataset_type is not a valid member of the
            DatasetType enum, or if there is no corresponding member of
            the StreamType enum.

        Notes:
        ------
            The other direction of mapping from stream to dataset type
            is not valid because there can be multiple dataset types
            that can be mapped to the same stream type.
        """

        stream_map = {
            dataset_type: cls.StreamType(dataset_type.value)
            for dataset_type in cls.DatasetType
        }

        try:
            stream_type = stream_map[dataset_type]

        except KeyError as key_error:
            raise KeyError(
                f'Corresponding stream type of {dataset_type} is not found.'
            ) from key_error

        return stream_type


class AlpacaDataSource(AbstractDataSource):
    """
    Represents Alpaca API as a data source. Provides standardized enums
    for historical and live data from Alpaca API.
    """

    class DatasetType(AbstractDataSource.DatasetType):
        """
        Enumeration class that defines constants for the different types
        of datasets.

        Attributes
        ----------
        BAR : str
            The type of dataset for aggregated trade stream data. Also
            known as bars data.
        TRADE : str
            The type of dataset for aggregated trade stream data.
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
        Enumeration class that defines constants for the different types
        of data streams.

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


@dataclass
class AbstractDataMetaData:
    """
    Data has a universal representation of a two dimensional array of
    objects throughout the framework. Each row corresponds to a time
    interval with a fixed length called resolution. Each column
    corresponds to a feature of the data for a time interval. The
    boolean mask indicates where the columns of the corresponding
    feature types are located in the data. Lenght of boolean mask is
    equal to the number columns in the data. Difference between dataset
    and stream is that dataset is static and can be loaded in memory,
    while stream is dynamic and can only be accessed in an iterator like
    fashion, where each iteration returns a new row of data and takes
    time equal to the resolution of the data. The metadata allows fusion
    of data from multiple data sources into a coehsive representation.
    This is useful for market simulations and trading abstracting away
    the construction of data from the representation of data. Metadata
    also offers automatic validation and updating of joined or appended
    data making joining multiple source of data a self-contained
    process. Note that market environments require price mask to be
    present for all symbols in the data schema. if this property is
    violated an error will be raised by the data feeder that uses this
    metadata before the simulation starts.

    Attributes:
    -----------
    data_schema:
        A dictionary mapping data source types to tuples of assets. The
        data schema defines the structure and format of the data for
        each asset in the dataset or stream.
    feature_schema Dict[FeatureType, Tuple[bool]]]:
        A dictionary mapping feature types to tuples of booleans, where
        each boolean indicates whether the corresponding feature is
        present in the data.
    resolution (str):
        A string representing the resolution of the data, which is the
        fixed length of each time interval in the dataset or stream.
    calendar_type (CalendarType):
        An instance of the `CalendarType` enum class that represents the
        type of calendar used to organize the data. The calendar type
        can affect how data is grouped and sorted in the dataset or
        stream.

    Properties:
    -----------
    n_columns: int
        Returns the number of columns in the dataset. This is useful for
        checking if the dataset has been downloaded correctly.
    assets: List[Asset]
        Returns a list of unique assets in the data schema. Order is
        preserved.
    asset_prices_mask:
        Returns a mask for the asset close price feature type. This
        price is used by market environments as the point of reference
        for placing orders. When a time interval is over and features
        are observed, the closing price of the interval is used to
        immediately place orders. The order of price mask matches the
        order of symbols in the data schema.
    valid: bool
        Ensures that all symbols have a price mask associated with them.
        This property can be violated during merging, since some feature
        types may not have a price mask associated with them, due to not
        being a price-related feature type. However, post-merging, the
        metadata can validate itself using this property. This is used
        by data feeders to validate input before feeding data to the
        market environments.
    schedule:
        Returns a DataFrame representing the schedule of the dataset.
        This is useful for checking if the dataset has been downloaded
        correctly.

    Methods:
    --------
    create_feature_schema(dataframe: pd.DataFrame) -> Dict[FeatureType,
    List[bool]]
        Creates a feature schema dictionary for a given DataFrame, with
        DataType as keys and boolean masks as values. The boolean masks
        indicate where the columns of the corresponding feature types
        are located in the data. By default, downloaders provide
        downloaded data in a pandas DataFrame format. The created
        feature schema can be used to instantiate the feature_schema
        attribute
    __or__(self, other: AbstractDataMetaData, **kwargs) ->
    AbstractDataMetaData
        Merges two metadata objects. This is useful for joining datasets
        that are large to download in one go. Each sub-dataset is
        downloaded for a fixed time interval and each can correspond to
        different data sources, feature types, and assets. Joining
        datasets and validating the process is done automatically using
        this method.
    __add__(self, other: AbstractDataMetaData, **kwargs) ->
    AbstractDataMetaData
        Appends two metadata objects. This is useful for appending
        datasets that are large to download in one go. At each
        iteration, the user can download the data in chunks
        corresponding to a fixed time interval shared between all other
        chunks and automatically validate the process and update the
        metadata. For example, downloading trade data for S&P500 stocks
        for a fixed time interval can happen by downloading the data for
        a list of symbols at a time.
    _validate_data_schema(self, data_schema) -> bool
        Checks if all stream or all datasets. This is useful for
        validating the data schema of the metadata object before joining
        or appending. This is used by the __or__ and __add__ methods to
        validate the data schema of the metadata object before joining
        or appending.
    _join_feature_schemas(self, other) -> Dict[FeatureType, List[bool]]
        Joins feature schemas of two datasets or streams. The boolean
        masks are simply concatenated to indicate the features type
        locations in the joined dataset/stream.
    """
    data_schema: Dict[AbstractDataSource.DatasetType:Tuple[AlpacaAsset]] | Dict[
        AbstractDataSource.StreamType:Tuple[AlpacaAsset]]
    feature_schema: Dict[FeatureType, Tuple[bool]]
    resolution: str
    calendar_type: CalendarType

    @property
    def n_columns(self) -> int:
        """
        Returns the number of columns in the dataset. This is useful for
        checking if the dataset has been downloaded correctly. 
        """

        n_columns = len(self.feature_schema.values()[0])
        return n_columns

    @property
    def assets(self) -> List[AlpacaAsset]:
        """
        Returns a list of unique assets in the data schema. Order is
        preserved.

        Returns:
        --------
            List[Asset]: a list of unique assets in the data schema.
        """
        assets = reduce(lambda x, y: x + y, self.data_schema.values())
        symbols = OrderedDict()
        for asset in assets:
            symbols[asset.symbol] = asset
        assets = list(symbols.values())
        return assets

    @property
    def asset_prices_mask(self):
        """
        Returns a mask for the asset close price feature type. This
        price is used by market environments as the point of reference
        for placing orders. when a time interval is over and features
        are observed the closing price of interval is used to
        immediately place orders. The order of price mask matches the
        order of assets in the data schema.
        """

        mask_in_list = [
            mask for feature_type, mask in self.feature_schema.items()
            if feature_type == FeatureType.ASSET_CLOSE_PRICE
        ]

        asset_price_mask = mask_in_list.pop() if mask_in_list else None
        return asset_price_mask

    @property
    def valid(self) -> bool:
        """
        ensures that all assets have a price mask associated with them.
        This property can be violated during merging, since some feature
        types may not have a price mask associated with them, due to not
        being a price related feature type. However post merging the
        metadata can validate itself using this property. used by data
        feeders to validaete input before feeding data to the market
        environments.

        Returns:
        --------
            bool: 
                True if all assets have a price mask associated with
                them, False otherwise.
        """

        valid = True if len(
            self.assets) == self.asset_prices_mask.count(True) else False
        return valid

    @property
    def schedule(self):
        """
        Returns a DataFrame representing the schedule of the dataset.
        This is useful for checking if the dataset has been downloaded
        correctly.
        """

        start_date = self.start.date()
        end_date = self.end.date()

        schedule = self.calendar_type.schedule(start_date=start_date,
                                               end_date=end_date)

        return schedule

    @staticmethod
    def create_feature_schema(dataframe: pd.DataFrame):
        """
        Creates a feature schema dictionary for a given DataFrame, with
        DataType as keys and boolean masks as values. The boolean masks
        indicate where the columns of the corresponding feature types
        are located in the data. By default downloaders provide
        downloaded data in a pandas Dataframe format.

        Args:
        ------
            dataframe (pd.DataFrame): 
                The input DataFrame for which the feature schema is to
                be created. By defaulat all feature types in FeatureType
                are enumerated and their value is matched against the
                column names of the input DataFrame. If a column name
                contains the vluae of a feature type, the corresponding
                boolean mask is set to True. this process is case
                insensitive. For example if dataframe has the column
                name 'AAPL_close_price' the boolean mask for
                FeatureType.ASSET_CLOSE_PRICE will be set to True at the
                position of the column name. Downloaders and streamers
                should ensure that the column names of the data they
                provide are consistent with this procedure.

        Returns:
        --------
            Dict[FeatureType, List[bool]]: 
                A dictionary with FeatureType as keys and boolean masks
                as values.
        """

        feature_schema = dict()

        for feature_type in FeatureType:

            mask = dataframe.columns.str.lower().str.match(
                '.*' + feature_type.value.lower() + '.*')
            feature_schema[feature_type] = mask

        return feature_schema

    def __or__(self, other: AbstractDataMetaData,
               **kwargs) -> AbstractDataMetaData:
        """
        This is useful for joining datasets that are large to download
        in one go. Each sub-dataset is downloaded for a fixed time
        interval and each can correponds to differnt data sources,
        feature types and symbols. Joining datasets and validating the
        process is done automatically using this method.

        Args:
        ------
            other (AbstractDataMetaData):
                The metadata object to be joined with the current
                metadata object.
        Returns:
        --------
            AbstractDataMetaData: 
                a new metadata object that is the result of joining the
                current metadata object with the other metadata object.
        Raises:
        -------
            ValueError: 
                if the data schemas of the two metadata objects are not
                compatible.
            ValueError: 
                if the resolutions of the two metadata objects are not
                the same.
            ValueError: 
                if the calendar types of the two metadata objects are
                not the same.
        """

        if not self._validate_data_schema(other.data_schema):

            raise ValueError(
                f'Metadata {other} has data schema {other.data_schema} '
                'which is not compatible with {self.data_schema}.')

        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        if not self.calendar_type != other.calendar_type:

            raise ValueError(
                f'Metadata {other} has calendar type {other.calendar_type} '
                'which is not compatible with {self.calendar_type}.')

        data_schema = self.data_schema.update(other.data_schema)
        feature_schema = self._join_feature_schemas(other)

        joined_metadata = self.__class__(data_schema=data_schema,
                                         feature_schema=feature_schema,
                                         resolution=self.resolution,
                                         **kwargs)

        return joined_metadata

    def __add__(self, other: AbstractDataMetaData,
                **kwargs) -> AbstractDataMetaData:
        """
        appends two metadata objects. This is useful for appending
        datasets that are large to downolad in one go. At each iteration
        the user can download the data in chunks corresponding to a
        fixed time interval shared between all other chunks and
        automatically validate the process and update the metadata. For
        example downloading tradde data for S&P500 stocks for a fixed
        time interval can happens by downloading the data for a list of
        symbols at a time. stream metadata child cannot use this method.
        appending stream metadata would not make sense.if used with
        stream metadata it will raise a not implemented error.

        Args:
        ------
            other (AbstractDataMetaData): another metadata object to be
            appended to the current metadata object.
        Returns:
        --------
            AbstractDataMetaData: 
                a new metadata object that is the result of appending
                the current metadata object with the other metadata
                object.
        Raises:
        -------
            ValueError: 
                if the data schemas of the two metadata objects are not
                compatible.
            ValueError:
                if the feature schemas of the two metadata objects are
                not compatible.
            ValueError:
                if the resolutions of the two metadata objects are not
                the same.
            ValueError:
                if the calendar types of the two metadata objects are
                not the same.
        """

        if dill.dumps(self.data_schema) != dill.dumps(other.data_schema):
            raise ValueError('Datasets must have identical data schemas.')

        if dill.dumps(self.feature_schema) != dill.dumps(other.feature_schema):
            raise ValueError('Datasets must have identical feature schemas.')

        if self.resolution != other.resolution:
            raise ValueError(
                f'Dataset resolutions{self.resolution} and {other.resolution} '
                'are mismatched.')

        if not self.calendar_type != other.calendar_type:

            raise ValueError(
                f'Metadata {other} has calendar type {other.calendar_type} '
                'which is not compatible with {self.calendar_type}.')

        appended_metadata = self.__class__(data_schema=self.data_schema,
                                           feature_schema=self.feature_schema,
                                           resolution=self.resolution,
                                           calendar_type=self.calendar_type,
                                           **kwargs)

        return appended_metadata

    def _validate_data_schema(self, data_schema):
        """
        Checks if all stream or all datasets. This is useful for
        validating the data schema of the metadata object before joining
        or appending. This is used by the __or__ and __add__ methods to
        validate the data schema of the metadata object before joining
        or appending.

        Args:
        ------
            data_schema
            (Dict[AbstractDataSource.DatasetType:Tuple[Asset]] |
            Dict[AbstractDataSource.StreamType:Tuple[Asset]]):
                The data schema to be validated.
        Returns:
        --------
            bool: True if all stream or all datasets, False otherwise.
        """
        #
        valid = True
        for data_type in data_schema:
            valid = valid and all(data_type == self_data_type
                                  for self_data_type in self.data_schema)

        return valid

    def _join_feature_schemas(self, other):
        """
        joins feature schemas of two datasets or streams. The boolean
        masks are simply concatenated to indicate the features type
        locations in the joind dataset/stream.

        Args:
        ------
            other (AbstractDataMetaData):
                The metadata object to be joined with the current
                metadata object.
        Returns:
        --------
            Dict[FeatureType, List[bool]]:
                A dictionary with FeatureType as keys and boolean masks
                as values.
        
        Raises:
        -------
            ValueError: if the feature schemas of the two metadata
            objects are not compatible.
        
        Notes:
        ------
            This method is used by the __or__ method to join two
            metadata objects. It is not used by the __add__ method
            because the feature schemas of the two metadata objects must
            be identical for appending to be valid.
        """
        if set(self.feature_schema.keys()) != set(other.data_schema.keys()):
            raise ValueError('Datasets do not have matching feature schemas.')

        merged_feature_schema = dict()
        for key in self.feature_schema.keys():

            merged_feature_schema[
                key] = self.feature_schema[key] + other.data_schema[key]

        return merged_feature_schema


@dataclass
class StreamMetaData(AbstractDataMetaData):
    """
    Subclass of AbstractDataMetaData that provides metadata for
    streaming data.

    Attributes:
    -----------
        data_schema: Dict[AbstractDataSource.StreamType:Tuple[str]]
            A dictionary mapping stream types to tuples of symbols. The
            data schema defines the structure and format of the data for
            each symbol in the stream.
        feature_schema: Dict[FeatureType, Tuple[bool]]
            A dictionary mapping feature types to tuples of booleans,
            where each boolean indicates whether the corresponding
            feature is present in the data.
        resolution: str
            A string representing the resolution of the data, which is
            the fixed length of each time interval in the stream.
        calendar_type: CalendarType 
            An instance of the `CalendarType` enum class that represents
            the type of calendar used to organize the data. The calendar
            type can affect how data is grouped and sorted in the
            stream.
    
    Properties:
    -----------
        n_rows: int
            Returns the number of rows in the stream. This is useful for
            checking if the stream has been downloaded correctly.
    
    Methods:
    --------
        __add__(self, other: AbstractDataMetaData, **kwargs) ->
        AbstractDataMetaData
            Not implemented for stream metadata. Appending stream
            metadata would not make sense. If used with stream metadata
            it will raise a not implemented error.
    """
    data_schema: Dict[AbstractDataSource.StreamType:Tuple[str]]
    feature_schema: Dict[FeatureType, Tuple[bool]]
    resolution: str
    calendar_type: CalendarType

    @property
    def n_rows(self) -> int:
        """
        Just to provide a consistent interface with DatasetMetaData and
        avoid boilerplate code.

        Returns:
        --------
            int: np.inf
        """

        n_rows = np.inf
        return n_rows

    def __add__(self, other: AbstractDataMetaData,
                **kwargs) -> AbstractDataMetaData:
        """
        Not implemented for stream metadata. Appending stream metadata
        would not make sense. If used with stream metadata it will raise
        a not implemented error.

        Args:
        ------
            other (AbstractDataMetaData): another metadata object to be
            appended to the current metadata object.
        Returns:
        --------
            AbstractDataMetaData:
                a new metadata object that is the result of appending
                the current metadata object with the other metadata
                object.
        Raises:
        -------
            NotImplementedError:
                This method is not implemented for stream metadata.
        """
        raise NotImplementedError


@dataclass
class DatasetMetadata(AbstractDataMetaData):
    """
    Subclass of AbstractDataMetaData that provides metadata for static
    datasets. This class is used to represent datasets that are
    downloaded and stored in memory. It provides a consistent interface
    for accessing the data and metadata. It also provides methods for
    joining and appending datasets. This is useful for joining datasets
    that are large to download in one go. Each sub-dataset is downloaded
    for a fixed time interval and each can correponds to differnt data
    sources, feature types and symbols. Joining datasets and validating
    the process is done automatically using this method. This is also
    useful for appending datasets that are large to downolad in one go.
    At each iteration the user can download the data in chunks
    corresponding to a fixed time interval shared between all other
    chunks and automatically validate the process and update the
    metadata.

    Attributes:
    -----------
    data_schema: Dict[AbstractDataSource.DatasetType:Tuple[Asset]]
        A dictionary mapping dataset types to tuples of assets. The data
        schema defines the structure and format of the data for each
        asset in the dataset.
    feature_schema: Dict[FeatureType, Tuple[bool]]
        A dictionary mapping feature types to tuples of booleans, where
        each boolean indicates whether the corresponding feature is
        present in the data.
    resolution: str
        A string representing the resolution of the data, which is the
        fixed length of each time interval in the dataset.
    start: datetime
        A datetime object representing the start time of the dataset.
        Note start and end times are inclusive.
    end: datetime
        A datetime object representing the end time of the dataset. Note
        start and end times are inclusive.
    n_rows: int
        An integer representing the number of rows in the dataset. This
        is useful for checking if the dataset has been downloaded
        correctly and reporting end of dataset to the market
        environment.

    Properties:
    -----------
    stream: StreamMetaData
        Returns a StreamMetaData object that corresponds to the current
        dataset metadata. This is useful for mapping the dataset
        metadata to a stream metadata for live streaming when traders
        want to deploy their trained agents in a live trading
        environment.
    
    Methods:
    --------
    __or__(self, other: AbstractDataMetaData, **kwargs) ->
    AbstractDataMetaData
        Merges two metadata objects. This is useful for joining datasets
        that are large to download in one go. Each sub-dataset is
        downloaded for a fixed time interval and each can correponds to
        differnt data sources, feature types and symbols. Joining
        datasets and validating the process is done automatically using
        this method.
    __add__(self, other: AbstractDataMetaData, **kwargs) ->
    AbstractDataMetaData
        Appends two metadata objects. Downloading large datasets can be
        split across nonoverlapping time intervals and appended to each
        other. This method facilitates updating the metadata object
        automatically and validating the process.
    """
    data_schema: Dict[AbstractDataSource.DatasetType:Tuple[AlpacaAsset]]
    feature_schema: Dict[FeatureType, Tuple[bool]]
    resolution: str
    calendar_type: CalendarType
    start: datetime
    end: datetime

    @property
    def stream(self):
        """
        Returns a StreamMetaData object that corresponds to the current
        dataset metadata. This is useful for mapping the dataset
        metadata to a stream metadata for live streaming when traders
        want to deploy their trained agents in a live trading
        environment.
        """

        data_schema = {
            dataset_type.stream: data_schema[dataset_type]
            for dataset_type in self.data_schema
        }
        stream = StreamMetaData(data_schema=self.data_schema,
                                feature_schema=self.feature_schema,
                                resolution=self.resolution,
                                calendar_type=self.calendar_type)

        return stream

    @property
    def days(self):
        """
        Returns the number of days in the dataset. This is useful for
        checking if the dataset has been downloaded correctly.
        """

        days = len(self.schedule)
        return days

    @property
    def n_rows(self):

        resolution = self.resolution
        resolution_offset = pd.to_timedelta(resolution)
        market_durations = (self.schedule['end'] - self.schedule['start'])

        intervals_per_day = (market_durations / resolution_offset).astype(int)
        n_rows = sum(intervals_per_day)
        return n_rows

    def __or__(self, other: AbstractDataMetaData) -> AbstractDataMetaData:
        """
        This is useful for joining datasets that are large to download
        in one go. Each sub-dataset is downloaded for a fixed time
        interval and each can correponds to differnt data sources,
        feature types and assets. Joining datasets and validating the
        process is done automatically using this method.

        Args:
        ------
            other (AbstractDataMetaData):
                The metadata object to be joined with the current
                metadata object.
        Returns:
        --------
            AbstractDataMetaData:
                a new metadata object that is the result of joining the
                current metadata object with the other metadata object.
        Raises: 
        -------
            ValueError:
                if the start time of the two metadata objects are not
                the same.
            ValueError:
                if the end time of the two metadata objects are not the
                same.
            ValueError:
                if the number of rows of the two metadata objects are
                not the same.
        """

        if self.start != other.start:
            raise ValueError(
                f'Current start time: {self.start}, does not match joined '
                'dataset start time {other.start}.')

        if self.end != other.end:
            raise ValueError(
                f'Current end time: {self.end}, does not match joined '
                'dataset end time {other.end}.')

        if self.n_rows != other.n_rows:
            raise ValueError('Datasets must have the same number of rows.')

        joined_metadata = super().__or__(other,
                                         start=self.start,
                                         end=self.end,
                                         n_rows=self.n_rows)

        return joined_metadata

    def __add__(self, other: AbstractDataMetaData) -> AbstractDataMetaData:
        """
        Appends two metadata objects. Downloading large datasets can be
        split across nonoverlapping time intervals and appended to each
        other. This method facilitates updating the metadata object
        automatically and validating the process.

        Args:
        ------
            other (AbstractDataMetaData):
                The metadata object to be appended to the current
                metadata object.
        Returns:
        --------
            AbstractDataMetaData:
                a new metadata object that is the result of appending
                the current metadata object with the other metadata
                object.
        Raises:
        -------
            ValueError:
                if the start and end time of the two metadata objects
                are not consecutive.
        """

        if not self._check_dates(prev_end=self.end, cur_start=other.start):

            raise ValueError('Non-consecutive market days between end time '
                             f'{self.end} and {other.start}.')

        return super().__add__(other,
                               start=self.start,
                               end=other.end,
                               n_rows=self.n_rows)

    def _check_dates(self, prev_end, cur_start):
        """
        Checks if two dates are consecutive market days. This is used to
        validate the process of appending datasets to ensure temporal
        continuity of the data.

        Args:
        ------
            prev_end (datetime):
                The end date of the previous dataset.
            cur_start (datetime):
                The start date of the current dataset.
        Returns:
        --------
            bool:
                True if the two dates are consecutive market days, False
                otherwise.
        Raises:
        -------
            ValueError:
                if the start date of the current dataset is not greater
                than the end date of the previous dataset.
        Notes:
        ------
            This method is used by the __add__ method to validate the
            process of appending datasets to ensure temporal continuity
            of the data.
        """

        start_date = prev_end.date()
        end_date = cur_start.date()

        if not self.prev_end < cur_start:
            raise ValueError(f'Start date{cur_start} must be greater than '
                             'previous end date {prev_end}.')

        schedule = self.calendar_type.schedule(start_date=start_date,
                                               end_date=end_date)

        conscutive = True if len(schedule) == 2 else False
        return conscutive


class AbstractDataFeeder(ABC):
    """
    Abstract base class for defining a data feeder that is responsible
    for feeding data to a market environment, iteratively. A data feeder
    can feed data in a static or asynchronous manner. A static data
    feeder is responsible for feeding data from a static source, such as
    a HDF5 file, while an asynchronous data feeder is responsible
    
    Methods:
    --------
        get_row_generator(self, *args, **kwargs) -> Iterable[np.ndarray]
            Returns a generator object that can be used to for iterative
            providing data for market simulation.
    """

    @abstractmethod
    def get_features_generator(self, *args, **kwargs):
        """
        Returns a generator object that can be used to for iterative
        providing data for market simulation.
        
        Raises:
            NotImplementedError: This method must be implemented by a
            subclass.
        """

        raise NotImplementedError


class StaticDataFeeder(AbstractDataFeeder):
    """
    Subclass of AbstractStaticDataFeeder that iteratively returns data
    required for the environment from a static source. This is useful
    for loading data from a HDF5 file or a numpy array. This class
    provides a consistent interface for loading data from different
    sources. It also provides methods for splitting the dataset into
    multiple non-overlapping contiguous sub-feeders that span the
    dataset. Common use case is it use in stable baselines vector env to
    parallelize running multiple trading environments, leading to
    significant speedup of training process.

    Attributes:
    -----------
        dataset_metadata (DatasetMetadata):
            Contains metadata for the dataset being loaded.
        datasets (List[h5.Dataset | np.ndarray]):
            Represents the actual dataset(s) to be loaded.
        start_index (int):
            Specifies the starting index to load the data from.
        end_index (int):
            Specifies the ending index to load the data from.
        n_rows (int):
            The number of rows in the dataset.
        n_columns (int):
            The number of columns in the dataset.   
        n_chunks (int):
            Indicates the number of chunks to divide the dataset into
            for loading. Loads one chunk at a time. Useful if datasets
            do not fit in memory or to allocate more memory for the
            training process.
    Methods:
    --------
        get_row_generator(self) -> Iterable[np.ndarray]
            Resets the internal state of the data feeder. Yields:
            Iterable[np.ndarray]: a generator object returning features
            as numpy array.
    """

    def __init__(self,
                 dataset_metadata: DatasetMetadata,
                 datasets: List[h5.Dataset | np.ndarray],
                 start_index: int = 0,
                 end_index: Optional[int] = None,
                 n_chunks: Optional[int] = 1) -> None:
        """
        Initializes a StaticDataFeeder object. 
        
        Args: 
        ------
        dataset_metadata (DatasetMetadata): 
            Contains metadata for the dataset beingloaded. 
        datasets (List[h5.Dataset | np.ndarray]):
            Represents the actual dataset(s) to be loaded. 
        start_index (int, optional): Specifies the starting index to
            load the data from. Default is 0. 
        end_index (int, optional): Specifies the ending index to load
            the data from. If not provided, defaults to the number of
            rows indicated in the metadata object. Default is None. 
        n_chunks (int, optional): 
            Indicates the number of chunks to divide the dataset into
            for loading. Loads one chunk at a time. Useful if datasets
            do not fit in memory or to allocate more memory for the
            training process. Default is 1.
        """

        self.dataset_metadata = dataset_metadata
        self.datasets = datasets
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else \
            self.dataset_metadata.n_rows
        self.n_rows = self.end_index - self.start_index
        self.n_columns = self.dataset_metadata.n_columns
        self.n_chunks = n_chunks

        if not dataset_metadata.valid:
            raise ValueError(
                f'{dataset_metadata} has mismatching number of assets'
                'and asset price mask values.')

        return None

    def get_features_generator(self) -> Iterable[np.ndarray]:
        """
        This method returns a generator object that can be used to for
        iterative providing data for market simulation.

        Returns:
        --------
            Iterable[np.ndarray]: a generator object returning features
            corresponding to each time interval as a numpy array.
        """

        chunk_edge_indices = np.linspace(start=self.start_index,
                                         stop=self.end_index,
                                         num=self.n_chunks + 1,
                                         dtype=int,
                                         endpoint=True)

        for start, end in zip(chunk_edge_indices[:-1], chunk_edge_indices[1:]):

            joined_chunks_in_memory = np.hstack(
                [dataset[start:end, :] for dataset in self.datasets])

            for row in joined_chunks_in_memory:
                yield row

    def split(self, n: int = 1 | float):
        """
        Splits the dataset into multiple non-overlapping contiguous
        sub-feeders that span the dataset. Common use case is it use in
        stable baselines vector env to parallelize running multiple
        trading environments, leading to significant speedup of training
        process. 

        Args:
        ------
            n (int | float): if int, number of sub-feeders to split the
            dataset into. if float (0, 1) yields two sub-feeders
            performing n, 1-n train test split
        Returns:
        --------
            List[StaticDataFeeder]: A list of StaticDataFeeder objects.
        Raises:
        -------
            ValueError: if n is not an int or float in (0, 1]
        """

        if isinstance(n, int):

            if not n > 0:
                raise ValueError("n must be a positive integer")

            edge_indices = np.linspace(start=self.start_index,
                                       stop=self.end_index,
                                       num=self.n + 1,
                                       dtype=int,
                                       endpoint=True)

        elif isinstance(n, float):

            if not 0 < n <= 1:
                raise ValueError("n must be a float in (0, 1]")

            edge_indices = np.array([
                self.start_index,
                int(self.start_index + n *
                    (self.end_index - self.start_index)), self.end_index
            ],
                                    dtype=int)

        static_data_feeders = list()

        for start, end in zip(edge_indices[:-1], edge_indices[1:]):

            static_data_feeder = StaticDataFeeder(
                dataset_metadata=self.dataset_metadata,
                datasets=self.datasets,
                start_index=start,
                end_index=end,
                n_chunks=self.n_chunks)

            static_data_feeders.append(static_data_feeder)

        return static_data_feeders


class AsyncDataFeeder(AbstractDataFeeder):
    """
    A subclass of AbstractDataFeeder that iteratively returns data
    required for the environment from an asynchronous source. This is
    useful for loading data from a live stream.
    """

    def __init__(self, stream_metadata: StreamMetaData,
                 data_client: AbstractDataClient) -> None:
        self.stream_metadata = stream_metadata
        self.data_client = data_client

        return None
