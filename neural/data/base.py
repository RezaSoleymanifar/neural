from functools import reduce
from datetime import datetime
from typing import Dict, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pickle
import pandas as pd

from neural.data.time import Calendar
from neural.data.enums import AbstractDataSource, FeatureType



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


class AbstractDataFeeder(ABC):

    """
    Abstract base class for defining a data feeder that is responsible for feeding
    data to a market environment, iteratively. A data feeder can feed data in a static
    or asynchronous manner. A static data feeder is responsible for feeding data from
    a static source, such as a HDF5 file, while an asynchronous data feeder is responsible
    
    """

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Returns a generator object that can be used to for iterative providing data
        for market simulation.
        
        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        raise NotImplementedError
