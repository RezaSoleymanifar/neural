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
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (TYPE_CHECKING, Dict, List, Callable, Iterable, Optional)

import dill
import h5py as h5
import numpy as np
import pandas as pd

from neural.client.base import AbstractDataClient
from neural.data.enums import FeatureType, AssetType, CalendarType

if TYPE_CHECKING:
    from neural.utils.time import Resolution


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


class AbstractDataSource(ABC):
    """
    Abstract base class for a data source that standardizes the interface for
    accessing data from different sources. A data source is typically an API or
    a database that provides access to data. This class defines the interface
    for accessing data from a data source. The data source can either provide a
    static data namely dataset that aggreates old data streams useful for
    training. Or it can provide a live stream of data that is used for high
    frequency trading. It also defines a set of nested enums for standardizing
    the available dataset and stream types. When defining a new data source the
    dataset and stream types will inherit from DatasetType and StreamType
    enums.

    Classes:
    -----------
        DatasetType : Enum
            Enumeration class that defines the available historical dataset
            types for the data source. This can include types such as stock
            prices, weather data, social media data, etc. Dataset types are
            used to organize the types of datasets that are available for
            training. Since these types are source dependent, they are defined
            in the data source class. properties and methods:
                - data_source (AbstractDataSource): 
                    A property that returns a pointer to the data source class
                    in the containing scope. This allows the enumeration
                    constants to access the data source's methods and
                    properties.
                - stream (AbstractDataSource.StreamType)): 
                    A property that uses the stream implementation of the data
                    source to map the dataset type to a stream type. This is
                    useful for obtaining a matching stream type for a dataset
                    type.
                - __eq__(self, other) -> bool: 
                    Convenience method to check if other is a dataset type. If
                    the other object inherits from
                    AbstractDataSource.DatasetType then this will return True.


        StreamType : Enum
            Enumeration class that defines the available stream types for the
            data source. This can include types such as price data, volume
            data, order book data, tweets etc. Stream types are used to stream
            live data for high frequency trading. Usually an algorithm is
            trained usinsg a static historical dataset and it's dataset
            metadata is mappped to a stream type for streaming and aggregating
            the type of data that was used to train the agent on. Any dataset
            type should logically have a corresponding stream type, otherwise
            agent will not be deployable in a live trading environment, if
            trained on that dataset type. properties and methods:
                - data_source (AbstractDataSource):
                    A property that returns a pointer to the data source class
                    in the outer scope.
                - __eq__(self, other) -> bool:  
                    Convenience method to check if another object is a stream
                    type.
    
    Methods:
    -----------
        stream(dataset_type: DatasetType) -> StreamType
            Returns the stream type corresponding to the specified dataset
            type. By default maps to corresponding stream type using the value
            of the dataset type enum. This means a dataset type with value
            'BAR' will be mapped to a stream type with value 'BAR'. This
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
        Enumeration class that defines constants for the different types of
        datasets. AbstractDataSource.DatasetType should be subclassed to define
        dataset types for a particular data source.

        Properties:
        -----------
            data_source (AbstractDataSource):
                A property that returns a pointer to the data source class in
                the containing scope. This allows the enumeration constants to
                access the data source's methods and properties.
            stream (AbstractDataSource.StreamType)):
                A property that uses the stream implementation of the data
                source to map the dataset type to a stream type. This is useful
                for obtaining a matching stream type for a dataset type.
        
        Methods:
        --------
            __eq__(self, other) -> bool:
                Convenience method to check if other is a dataset type. If the
                other object inherits from AbstractDataSource.DatasetType then
                this will return True.
        """

        @property
        def data_source(self) -> AbstractDataSource:
            """
            Returns a pointer to the data source class in the outer
            scope. This allows the enumeration constants to access the
            data source's methods and properties.

            Returns:
            --------
                AbstractDataSource: 
                    A pointer to the data source class in the outer
                    scope.
            
            Notes:
            ------
            __qualname__ is a magic attribute that returns the fully
            qualified name of the class. It contains the containing
            modules and classes separated by dots. for example
            'neural.data.base.AbstractDataSource.DatasetType'. This
            attribute is used to access the data source class in the
            outer scope.
            """
            data_source = globals()[self.__class__.__qualname__.split(
                '.', maxsplit=1)[0]]

            return data_source

        @property
        def stream(self) -> AbstractDataSource.StreamType:
            """
            Uses the stream implementation of the data source to map the
            dataset type to a stream type. This is useful for obtaining a
            matching stream of data for a particular dataset type.

            Returns:
            --------
                AbstractDataSource.StreamType:
                    A stream type corresponding to the dataset type.
            """
            return self.data_source.stream(self)

        def __eq__(self, other) -> bool:
            """
            convenience method to check if other is a dataset type.
            This equality is independent of the data source.

            Args:
            -----
                other (object):
                    The object to be compared with the current dataset
                    type.
            
            Returns:
            --------
                bool:
                    True if the other object is a dataset type, False   
                    otherwise.
            """
            return isinstance(other, AbstractDataSource.DatasetType)

    class StreamType(Enum):
        """
        Enumeration class that defines constants for the different types
        of data streams. AbstractDataSource.StreamType should be
        subclassed to define stream types for a particular data source.

        Attributes:
        ----------
            data_source (AbstractDataSource):
                A property that returns a pointer to the data source
                class in the outer scope.
        Methods:
        --------
            __eq__(self, other) -> bool:    
                Convenience method to check if another object is a
                stream type.
        """

        @property
        def data_source(self) -> AbstractDataSource:
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

        def __eq__(self, other) -> bool:
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
        Returns a StreamType enum member corresponding to the given DatasetType
        enum member. If a different behavior is intended subclasses can
        override this method to provide custom mapping between dataset and
        stream types.

        Args:
        ---------
            dataset_type (DatasetType): 
                A member of the DatasetType enum that represents the type of
                dataset.

        Returns:
        --------
            StreamType: 
                A member of the StreamType enum that corresponds to the given
                dataset_type.

        Raises: 
        -------
            ValueError: 
                If dataset_type is not a valid member of the DatasetType enum,
                or if there is no corresponding member of the StreamType enum.

        Notes:
        ------
            The other direction of mapping from stream to dataset type is not
            valid because there can be multiple dataset types that can be
            mapped to the same stream type.
        """
        if not isinstance(dataset_type, AbstractDataSource.DatasetType):
            raise ValueError(f'{dataset_type} must be of type '
                             f'{AbstractDataSource.DatasetType.__name__}')

        try:
            stream_type = cls.StreamType(dataset_type.value)

        except KeyError as key_error:
            raise KeyError(
                f'Corresponding stream type of {dataset_type} is not found.'
            ) from key_error

        return stream_type


class DataSchema:
    """
    A class that represents a data schema. A data schema is internally a
    dictionary that maps data types (dataset type or stream type) to the
    corresponding assets and feature schema. It serves as a nexus to
    bundle data type, feature schema and assets together.
    
    The feature schema is a
    dictionary that maps feature types to boolean masks. The boolean
    masks indicate where the columns of the corresponding feature types
    are located in the data. Lenght of boolean mask is equal to the
    number columns in the data. Lenght of True values in the boolean
    mask is equal to the number of assets in the data schema.

    Notes:
    ------
        Data schemas can be added together to represent a monolithic
        data schema that consists of smaller data schemas. This is
        useful for have a unified interface for joined datasets or
        streams that abstracts away the construction of data.
    
    Attributes:
    -----------
        schema (Dict[AbstractDataSource.DatasetType:Dict[str,
        List[bool] | List[AbstractAssets]]] |
        Dict[AbstractDataSource.StreamType:Dict[str,
        List[bool] | List[AbstractAssets]]]):
            A dictionary that maps data types to the corresponding
            assets and feature schema. The feature schema is a
            dictionary that maps feature types to boolean masks.
    Example:
    --------
    >>> data_schema = DataSchema(
    ...     DatasetType.BAR, (AAPL, MSFT, GOOG), feature_schema)
    >>> data_schema.schema[DatasetType.BAR]['assets']
    (AAPL, MSFT, GOOG)
    >>> data_schema.schema[DatasetType.BAR]['feature_schema']
    {FeatureType.ASSET_CLOSE_PRICE: [True, False, True, False, True, False]}
    """

    def __init__(self, data_type: AbstractDataSource.DatasetType
                 | AbstractDataSource.StreamType, assets: List[AbstractAsset],
                 feature_schema: FeatureSchema) -> None:
        """
        Initializes the data schema using a data type, assets and a feature
        schema. The data type can be a dataset type or a stream type. The
        assets are a list of assets that are associated with the data schema.
        The feature schema is a dictionary that maps feature types to boolean
        masks. The boolean masks indicate where the columns of the
        corresponding feature types are located in the data. Lenght of boolean
        mask is equal to the number columns in the data.

        Args:
        ------
            data_type (AbstractDataSource.DatasetType |
            AbstractDataSource.StreamType): 
                The data type of the data schema. This can be a dataset type or
                a stream type.
            assets (List[AbstractAsset]):
                A list of assets that are associated with the data schema.
            feature_schema (FeatureSchema):
                A dictionary that maps feature types to boolean masks.
        """

        self.schema = OrderedDict()
        self.schema[data_type]['assets'] = assets
        self.schema[data_type]['feature_schema'] = feature_schema

    @property
    def is_dataset(self) -> bool:
        """
        Returns if the data schema is for a dataset or a stream. If
        the data schema is for a dataset then the data type is a
        dataset type, otherwise it is a stream type.

        Returns:
        --------    
            bool:
                True if the data schema is for a dataset, False if the
                data schema is for a stream.
        
        Notes:
        ------
            Useful to ensure that all data types are either datasets or
            streams.
        """
        data_type = self.schema.keys()[0]
        if issubclass(data_type, AbstractDataSource.DatasetType):
            return True
        elif issubclass(data_type, AbstractDataSource.StreamType):
            return False
        
    @property
    def assets(self) -> List[AbstractAsset]:
        """
        Returns a list of assets that have a price mask associated with
        them. This is useful to filter out tradable assets from the
        assets that exist to provide feature information for the
        tradable assets.
        """
        assets = list()
        for data_type in self.schema:
            asset_prices_mask = self.schema[data_type]['feature_schema'][
                FeatureType.ASSET_CLOSE_PRICE]
            data_type_assets = self.schema[data_type]['assets']
            assets.extend([
                asset for asset, mask_value in zip(
                    data_type_assets, asset_prices_mask) if mask_value
            ])
        if len(assets) != len(set(assets)):
            raise ValueError('Duplicate tradable assets in data schema.')

        return assets

    @property
    def n_features(self) -> int:
        """
        Returns the number of columns in the dataset. Can be used to
        compare against the number columns in the underlying data for 
        sanity checks.

        Returns:
        --------
            int:
                The number of columns in the dataset.
        """
        n_features = sum(
            len(self.schema[data_type]['feature_schema'].values()[0])
            for data_type in self.schema)
        return n_features

    @property
    def asset_prices_mask(self) -> List[bool]:
        """
        Returns a mask for the asset close price feature type. This price is
        used by market environments as the point of reference for placing
        orders. when a time interval is over and features are observed the
        closing price of interval is used to place orders. The order of price
        mask matches the order of assets in the data schema.
        """
        return self.get_features_mask(FeatureType.ASSET_CLOSE_PRICE)

    def get_features_mask(self, feature_type: FeatureType):
        """
        Retrievs the boolean mask for a given feature type. This is
        useful for filtering out features of a particular type from the
        data. For example if the user wants to filter out all features
        that are text based, they can use this method to get the mask
        for text features FeatureType.TEXT and filter out the columns
        """
        mask = [
            mask_value for data_type in self.schema
            for mask_value in self.schema[data_type]['feature_schema'][feature_type]
        ]
        return mask

    
    def __add__(self, other) -> DataSchema:
        """
        Adds two data schemas together. This is useful for joining
        datasets or streams. If assets in a common data type overlap
        then an error is raised. This is due to the fact that same
        assets for the same data type gives redundant information.
        """
        if other.is_dataset != self.is_dataset:
            raise ValueError(
                'Only joining all streams or all datasets are accepted.')

        new_data_schema = deepcopy(self)
        schema = new_data_schema.schema
        for data_type in other.schema.keys():
            if data_type in schema:
                assets = schema[data_type]['assets']
                other_assets = other.schema[data_type]['assets']
                if set(assets).intersection(set(other_assets)):
                    raise ValueError(f'Overlap between {assets} and '
                                     f'{other_assets} in data type {data_type}')
                schema[data_type]['assets'] += other.schema[data_type]['assets']
                schema[data_type]['feature_schema'] += other.schema[data_type][
                    'feature_schema']
            else:
                schema[data_type] = other.schema[data_type]
        return new_data_schema


class FeatureSchema:
    """
    A class that represents a feature schema. A feature schema is a
    an object that maps feature types to boolean masks. The boolean
    masks indicate where the columns of the corresponding feature
    types are located in the data. Lenght of boolean mask is equal to
    the number columns in the data. Length of True values in the
    boolean mask is equal to the number of assets in the data schema.
    A mask is created for every feature type in FeatureType enum.

    Feature schema is instantiated using a pandas DataFrame. The columns
    of the dataframe are matched against the string values of the
    feature types in FeatureType enum. If a column name matches the
    string value of a feature type, the corresponding boolean mask is
    set to True. this process is case insensitive.

    Attributes:
    -----------
        schema (Dict[FeatureType, List[bool]]):
            A dictionary that maps feature types to boolean masks.

    Example:
    --------
    >>> feature_schema = FeatureSchema(dataframe)   
    >>> feature_schema.schema[FeatureType.ASSET_CLOSE_PRICE]
    [True, False, True, False, True, False]
    This mask now can be applied to a row to return the close prices.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initializes the feature schema using a pandas DataFrame. The
        columns of the dataframe are matched against the string values
        of the feature types in FeatureType enum. If a column name
        matches the string value of a feature type, the corresponding
        boolean mask is set to True. this process is case insensitive.

        Args:
        ------
            dataframe (pd.DataFrame):
                The input DataFrame for which the feature schema is to
                be created. By defaulat all feature types in FeatureType
                are enumerated and their value is matched against the
                column names of the input DataFrame.
        """
        self.schema = self.create_feature_schema(dataframe)
        return None

    def __repr__(self) -> str:
        """
        Returns a string representation of the feature schema.

        Returns:
        --------
            str:
                A string representation of the feature schema.
                example: {FeatureType.ASSET_CLOSE_PRICE: [True, False]}
        """
        return str(self.schema)

    def __add__(self, other):
        """
        joins feature schemas of two datasets or streams. The boolean
        masks are simply concatenated to indicate the feature type
        locations in the joined dataset/stream.

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
            ValueError: 
                if the feature schemas of the two metadata objects are
                not compatible.
        """
        if set(self.schema.keys()) != set(other.schema.keys()):
            raise ValueError('Feature schemas do not have same feature types.')

        new_feature_schema = deepcopy(self)
        schema = new_feature_schema.schema
        for feature_type in schema:
            schema[feature_type] += other.schema[feature_type]

        return new_feature_schema

    def create_feature_schema(
            self, dataframe: pd.DataFrame) -> Dict[FeatureType, List[bool]]:
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
                matches the string value of a feature type, the
                corresponding boolean mask is set to True. this process
                is case insensitive. For example if dataframe has the
                column name 'AAPL_close_price' the boolean mask for
                FeatureType.ASSET_CLOSE_PRICE will be set to True at the
                position of the column name since string value of
                ASSET_CLOSE_PRICE is 'CLOSE' and it exists as a
                substring in the column name 'AAPL_close_price'.
                Downloaders and streamers should ensure that the column
                names of the data they provide are consistent with this
                procedure.

        Returns:
        --------
            Dict[FeatureType, List[bool]]: 
                A dictionary with FeatureType as keys and boolean masks
                as values.

        Example:
        --------
        Assume dataframe has column names 'AAPL_close_price',
        'AAPL_OPEN_PRICE' then the feature schema will be:
        {FeatureType.ASSET_CLOSE_PRICE: [True, False],
        FeatureType.ASSET_OPEN_PRICE: [False, True]}. Columns are
        matched due to having "close" and "OPEN" substrings in their
        names matching the lower case values of
        FeatureType.ASSET_CLOSE_PRICE and ASSET_OPEN_PRICE respectively.
        This process is case insensitive.
        """
        feature_schema = dict()
        for feature_type in FeatureType:
            feature_type_mask = dataframe.columns.str.lower().str.match(
                '.*' + feature_type.value.lower() + '.*').to_list()
            feature_schema[feature_type] = feature_type_mask
        return feature_schema


@dataclass
class AbstractDataMetaData:
    """
    Data has a universal representation of a two dimensional array of
    objects throughout the framework. Each row corresponds to a time
    interval with a fixed time span called resolution. Each column
    corresponds to a feature of the data for that interval. The boolean
    masks in feature schema indicates where the columns of the
    corresponding feature types are located in the data. Lenght of
    boolean mask is equal to the number columns in the data. Difference
    between dataset and stream is that dataset is static and can be
    loaded in memory, while stream is dynamic and can only be accessed
    in an iterator like fashion, where each iteration returns a new row
    of data and takes time equal to the resolution of the data. The
    metadata allows fusion of data from multiple data sources into a
    coehsive representation. This is useful for market simulations and
    trading abstracting away the construction of data from the
    representation of data. Metadata also offers automatic validation
    and updating of joined or appended data making joining multiple
    sources of data a self-contained process. For trading purposes any
    asset that has FeatureType.ASSET_CLOSE_PRICE will be treated as a 
    tradable asset. However it is possible to provide other non close 
    price data for other assets, in the same metadata object.

    Attributes:
    -----------
    data_schema (DataSchema):
        An instance of the `DataSchema` class that represents the data
        schema of the data. The data schema is internally a dictionary
        that maps data types (dataset type or stream type) to the
        corresponding assets and feature schema. The feature schema is a
        dictionary that maps feature types to boolean masks. The boolean
        masks indicate where the columns of the corresponding feature
        types are located in the data. When loading multiple datasets
        from HDF5 file they are loaded basedon creation order and
        reorded based on data types in the data schema. This way all
        common data types have the same order of appearance in the final
        feature engineering.
    resolution (Resolution):
        An instance of the `Resolution` enum class that represents the
        resolution of the data. The resolution is the time span of each
        row in the data. examples: resolution = Resolution(5,
        Resolution.Unit.MINUTE) shows a resolution of 5 minutes.
    calendar_type (CalendarType):
        An instance of the `CalendarType` enum class that represents the
        type of calendar used to organize the data. The calendar type
        can affect how data is grouped and sorted in the dataset or
        stream.

    Properties:
    -----------
    n_columns: int
        Returns the number of columns in the dataset.
    assets: List[Asset]
        Returns a list of unique assets in the data schema. Order is
        preserved and determined by the order of appearance of assets in
        the data schema.
    asset_prices_mask:
        Returns a mask for the asset close price feature type. This
        price is used by market environments as the point of reference
        for placing orders. When a time interval is over and features
        are observed, the closing price of the interval is used to place
        orders.
    valid: bool
        Ensures that all symbols have a price mask associated with them.
        This property can be violated during merging, since some feature
        types may not have a price mask associated with them, due to not
        being a price-related feature type. However, post-merging, the
        metadata can validate itself using this property. This is used
        by data feeders to validate input before feeding data to the
        market environments.
    schedule:
        Returns a function that returns a DataFrame representing the
        schedule of the dataset according to its calendar type. This is
        used in training to map dataset rows to corresponding dates. In
        trading this is used to check for market open and close times.

    Methods:
    --------
    create_feature_schema(dataframe: pd.DataFrame) -> Dict[FeatureType,
    List[bool]]
        Creates a feature schema dictionary for a given DataFrame, with
        DataType as keys and boolean masks as values. The boolean masks
        indicate where the columns of the corresponding feature types
        are located in the data. By default, downloaders provide
        downloaded data in a pandas DataFrame format. The created
        feature schema can be used to pass to the feature_schema
        attribute.
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
        datasets that are large to download in one go. Each datasets
        can be downloaded for a span of for example two years and later
        appended using this method. StreamMetadata subclass cannot use
        this method. Appending stream metadata would not make sense. If
        used with stream metadata it will raise a not implemented error.
    
    Notes:
    ------
        When joining metadata the data schema updates based on 
    """
    data_schema: DataSchema
    resolution: Resolution
    calendar_type: CalendarType

    @property
    def n_columns(self) -> int:
        """
        Returns the number of columns in the dataset. Can be used to
        compare against the number columns in the underlying data for 
        sanity checks.

        Returns:
        --------
            int:
                The number of columns in the dataset.
        """
        return self.data_schema.n_features

    @property
    def assets(self) -> List[AbstractAsset]:
        """
        Returns a list of assets that have a price mask associated with
        them. This is useful to filter out tradable assets from the
        assets that exist to provide feature information for the
        tradable assets.
        Returns:
        --------
            List[AbsractAsset]: 
                a list of unique assets in the data schema.

        Notes:
        ------
            By default number of True values in the price mask is equal
            to the number of assets in the data schema. This is enforced
            at time of creation of the data schema.
        """
        return self.data_schema.assets

    @property
    def schedule(self) -> Callable[[datetime, datetime], pd.DataFrame]:
        """
        Returns a function that returns a DataFrame representing the
        schedule of the dataset according to its calendar type. This is
        used in training to map dataset rows to corresponding dates. In 
        trading this is used to check for market open and close times.

        Returns:
        --------
            Callable[[datetime, datetime], pd.DataFrame]:
                A function that returns a DataFrame representing the
                schedule of the dataset according to its calendar type.
        
        Example:
        --------
        >>> start_date, end_date = datetime(2022, 1, 3), datetime(2022, 1, 11)
        >>> schedule = metadata.schedule
        >>> schedule(start_date, end_date)
                    start                       end
        2022-01-03  2022-01-03 00:00:00+00:00   2022-01-04 00:00:00+00:00
        2022-01-04  2022-01-04 00:00:00+00:00   2022-01-05 00:00:00+00:00
        2022-01-05  2022-01-05 00:00:00+00:00   2022-01-06 00:00:00+00:00
        2022-01-06  2022-01-06 00:00:00+00:00   2022-01-07 00:00:00+00:00
        2022-01-07  2022-01-07 00:00:00+00:00   2022-01-08 00:00:00+00:00
        2022-01-10  2022-01-10 00:00:00+00:00   2022-01-11 00:00:00+00:00
        """
        return self.calendar_type.schedule

    def __or__(self, other: AbstractDataMetaData,
               **kwargs) -> AbstractDataMetaData:
        """
        This is useful for joining datasets that are large to download in one
        go. Each sub-dataset is downloaded for a fixed span of time and each
        can correpond to different data sources, feature types and assets.
        Joining datasets and validating the process is done automatically using
        this method.
            
        Args:
        ------
            other (AbstractDataMetaData):
                The metadata object to be joined with the current metadata
                object.
        Returns:
        --------
            AbstractDataMetaData: 
                a new metadata object that is the result of joining the current
                metadata object with the other metadata object.
        Raises:
        -------
            ValueError: 
                If feature schema data types of the two metadata objects
                are not compatible. Feature schema keys must be all
                datasets or all streams. Joining datasets and streams
                will raise an error.
            ValueError: 
                if the resolutions of the two metadata objects are not the
                same.
            ValueError: 
                if the calendar types of the two metadata objects are not the
                same.
        """
        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        if not self.calendar_type != other.calendar_type:
            raise ValueError(
                f'Metadata {other} has calendar type {other.calendar_type} '
                'which is not compatible with {self.calendar_type}.')

        data_schema = self.data_schema + other.data_schema

        joined_metadata = self.__class__(data_schema=data_schema,
                                         resolution=self.resolution,
                                         calendar_type=self.calendar_type,
                                         **kwargs)

        return joined_metadata

    def __add__(self, other: AbstractDataMetaData,
                **kwargs) -> AbstractDataMetaData:
        """
        appends two metadata objects. This is useful for appending
        datasets that are large to downolad in one go. At each iteration
        the user can download the data in chunks corresponding to a
        fixed time span shared between all other chunks and
        automatically validate the process and update the metadata. For
        example downloading trade data for S&P500 stocks for a fixed
        time interval can happen by downloading the data for every two
        years and later appending them using this method. StreamMetadata
        subclass cannot use this method. Appending stream metadata would
        not make sense.if used with stream metadata it will raise a not
        implemented error.

        Args:
        ------
            other (AbstractDataMetaData): 
                another metadata object to be appended to the current
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
                if the data schemas of the two metadata objects are not
                identical.
            ValueError:
                if the feature schemas of the two metadata objects are
                not identical.
            ValueError:
                if the resolutions of the two metadata objects are not
                the same.
            ValueError:
                if the calendar types of the two metadata objects are
                not the same.
        """
        if dill.dumps(self.data_schema) != dill.dumps(other.data_schema):
            raise ValueError('Datasets must have identical data schemas.')

        if self.resolution != other.resolution:
            raise ValueError(
                f'Dataset resolutions{self.resolution} and {other.resolution} '
                'are mismatched.')

        if not self.calendar_type != other.calendar_type:
            raise ValueError(
                f'Metadata {other} has calendar type {other.calendar_type} '
                f'which is not compatible with {self.calendar_type}.')

        appended_metadata = self.__class__(data_schema=self.data_schema,
                                           resolution=self.resolution,
                                           calendar_type=self.calendar_type,
                                           **kwargs)

        return appended_metadata


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
    start: datetime
    end: datetime

    @property
    def stream(self):
        """
        Returns a StreamMetaData object that corresponds to the current
        dataset metadata. This is useful for mapping the dataset
        metadata to a stream metadata for live streaming when traders
        want to deploy their trained agents in a live trading
        environment. Simply changes the dataset type to stream type for
        each data type in the data schema.
        """
        data_schema = {
            dataset_type.stream: self.data_schema.schema[dataset_type]
            for dataset_type in self.data_schema.schema
        }
        stream = StreamMetaData(data_schema=data_schema,
                                resolution=self.resolution,
                                calendar_type=self.calendar_type)
        return stream

    @property
    def schedule(self):
        schedule = super().schedule(start_date=self.start.date(),
                                    end_date=self.end.date())
        return schedule

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
        """
        Returns the number of rows in the dataset. Uses the schedule
        and resolution to calculate the number of rows in the dataset.
        When downloading datasets this process is used to
        create rows in the dataset.
        """
        time_delta = self.resolution.pandas_timedelta
        market_durations = (self.schedule['end'] - self.schedule['start'])
        intervals_per_day = (market_durations / time_delta).astype(int)
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

        joined_metadata = super().__or__(other, start=self.start, end=self.end)
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

        return super().__add__(other, start=self.start, end=other.end)

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
        if not prev_end < cur_start:
            raise ValueError(f'Start date{cur_start} must be after '
                             'previous end date {prev_end}.')
        start_date = prev_end.date()
        end_date = cur_start.date()
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
            Returns a generator object that can be used to iteratively
            provide data for market environment.
    """

    @abstractmethod
    def get_features_generator(self, *args, **kwargs):
        """
        Returns a generator object that can be used to iteratively
        provide data for market environment.
        
        Raises:
            NotImplementedError: 
                This method must be implemented by a subclass.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_rows(self):
        """
        Returns the number of rows to be generated by the data feeder. This is
        useful for checking if the data feeder has been exhausted. Asynchronous
        data feeders return this value as np.inf, since they are not limited by
        the number of rows in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_features(self):
        """
        Returns the number of features in the dataset.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def index(self):
        """
        Returns the current index of the data feeder. This is useful to have a
        reference to the current row index being fed to the market environment.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def done(self):
        """
        Returns True if the data feeder has been exhausted, False otherwise.
        Asynchronous data feeders always return False.
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
    dataset. Common use case is to pair with gym AsyncVectorEnv, and
    SyncVectorEnv to parallelize running multiple trading environments,
    leading to significant speedup of training process and improvement
    in generalization.

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
        self.n_chunks = n_chunks

        self._cumulative_daily_rows = self._get_cumulative_daily_rows()
        self._index = None
        return None

    @property
    def n_rows(self):
        return self.end_index - self.start_index

    @property
    def n_features(self):
        return self.dataset_metadata.n_columns

    @property
    def index(self):
        return self._index

    @property
    def done(self):
        return True if self.index == self.end_index - 1 else False

    @property
    def start_date(self):
        """
        Returns the date corresponding to the start index. This is
        useful for mapping the index of the dataset to the date of the
        episode.
        """
        return self.get_date(self.start_index)

    @property
    def end_date(self):
        """
        Returns the date corresponding to the end index. 
        """
        return self.get_date(self.end_index)

    @property
    def date(self):
        """
        Returns the current date of the episode.
        """
        return self.get_date(self.index)

    @property
    def days(self):
        """
        Returns the number of days in the dataset.
        """
        days = (self.end_date - self.start_date).days
        return days

    def _get_cumulative_daily_rows(self) -> int:
        """
        Returns a pandas Series object that contains the cumulative
        number of rows per day. This is useful for mapping the index of
        the dataset to the date of the episode and also adjusting the 
        start and end times of the data feeders to match the start and
        end of days, namely making sure that data feeders work with
        integer number of days.
        """
        if not self._cumulative_daily_rows:
            schedule = self.dataset_metadata.schedule
            resolution = self.dataset_metadata.resolution
            daily_durations = schedule['end'] - schedule['start']
            rows_per_day = (daily_durations /
                            resolution.pandas_timedelta).astype(int)
            self._cumulative_daily_rows = rows_per_day.cumsum().values
        return self._cumulative_daily_rows

    def _validate_indices(self):
        """
        Validates the start and end indices to make sure that they
        correspond to the start and end of days. This is useful for
        making sure that data feeders work with integer number of days.
        """
        valid_indices = np.concatenate([0], self._cumulative_daily_rows)
        if (self.start_index not in valid_indices
                or self.end_index not in valid_indices):
            raise ValueError(
                f'Start index {self.start_index} or end index {self.end_index} '
                'does not match a row index corresponding to the start/end of a day.'
            )

    def get_date(self, index) -> datetime:
        """
        Returns the date corresponding to the current index. This is
        useful for mapping the index of the dataset to the date of the
        episode.

        Returns:
        --------
            date (datetime):
                The date corresponding to the current index.
        """
        day_index = (index < self._cumulative_daily_rows).argmax()
        date = self.dataset_metadata.schedule.loc[day_index, 'start']
        return date

    def get_features_generator(self) -> Iterable[np.ndarray]:
        """
        This method returns a generator object that can be used to for
        iterative providing data for market simulation. Data is loaded in
        chunks and each chunk is loaded into memory and then iteratively
        returned until chunk is exhausted and next chunk is loaded. This is
        useful for loading large datasets that do not fit in memory or to
        allocate more memory for the training.

        Returns:
        --------
            Iterable[np.ndarray]: 
                a generator object returning features corresponding to each
                time interval as a numpy array.
        """
        chunk_edge_indices = np.linspace(start=self.start_index,
                                         stop=self.end_index,
                                         num=self.n_chunks + 1,
                                         dtype=int,
                                         endpoint=True)
        self._index = self.start_index - 1

        for start, end in zip(chunk_edge_indices[:-1], chunk_edge_indices[1:]):
            joined_chunks_in_memory = np.hstack(
                [dataset[start:end, :] for dataset in self.datasets])
            for row in joined_chunks_in_memory:
                self._index += 1
                yield row

    def split(self, n: int = 1 | float):
        """
        Splits the dataset into multiple non-overlapping contiguous
        sub-feeders that span the dataset. Common use case is it use in
        stable baselines vector env to parallelize running multiple
        trading environments, leading to significant speedup of training
        process. Ensures that the sub-feeders are non-overlapping,
        contiguous and match integer number of days in length.

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
                                       num=n + 1,
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

        start, end, middle = edge_indices[0], edge_indices[1], edge_indices[
            1:-1]
        cumulative_closest_indices = np.searchsorted(
            self._cumulative_daily_rows, middle, side='right') - 1
        edge_indices = np.concatenate((start, cumulative_closest_indices, end))

        if len(edge_indices) != len(np.unique(edge_indices)):
            raise ValueError(
                f'Value of n is too large for n_rows: {self.n_rows} '
                f'cannot be split dataset into {n} sub-feeders.')

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

    @property
    def n_rows(self):
        return np.inf

    @property
    def n_features(self):
        return self.stream_metadata.n_columns

    @property
    def index(self):
        return None

    @property
    def done(self):
        return False
