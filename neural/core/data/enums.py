from enum import Enum
from datetime import datetime
from typing import (List, Dict, Tuple)
from dataclasses import dataclass
import pickle

from alpaca.trading.enums import AssetClass
from neural.tools.misc import Calendar

class StreamType(Enum):

    """
    Enumeration class that defines constants for the different types of data streams.

    Attributes:
        QUOTE (str): The type of data stream for quotes.
        TRADE (str): The type of data stream for trades.
        ORDER_BOOK (str): The type of data stream for order book data.
    """

    TRADE = 'TRADE'
    QUOTE = 'QUOTE'
    ORDER_BOOK = 'ORDER_BOOK'
    
class DatasetType(Enum):

    """
    Enumeration class that defines constants for the different types of datasets.

    Attributes:
        BAR (str): The type of dataset for aggregated trade stream data.
        QUOTE (str): The type of dataset for quote data.
        ORDER_BOOK (str): The type of dataset for order book data.
    """

    BAR = 'BAR' # bar is aggregated trade stream
    QUOTE = 'QUOTE'
    ORDER_BOOK = 'ORDER_BOOK'

class ColumnType(Enum):

    """
    Enumeration class that defines constants for the different types of columns in datasets.

    Attributes:
        OPEN (str): The type of column for opening price data.
        HIGH (str): The type of column for high price data.
        LOW (str): The type of column for low price data.
        CLOSE (str): The type of column for closing price data.
        BID (str): The type of column for bid price data.
        ASK (str): The type of column for ask price data.
        SENTIMENT (str): The type of column for sentiment data.
        EMBEDDING (str): The type of column for embedding data.
    """

    OPEN = 'OPEN'
    HIGH = 'HIGH'
    LOW = 'LOW'
    CLOSE = 'CLOSE'
    BID = 'BID'
    ASK = 'ASK'
    SENTIMENT = 'SENTIMENT'
    EMBEDDING = 'EMBEDDING'

@dataclass
class StreamMetaData:

    """
    Dataclass that defines the metadata for a data stream.

    Attributes:
        stream_type (List[StreamType]): A list of StreamType objects that describe the type of stream.
        column_schema (Dict[ColumnType, Tuple[bool]]): A dictionary that maps ColumnType objects to tuples of boolean
            values, where the first boolean indicates whether the column is required and the second boolean indicates
            whether the column is nullable.
        asset_class (AssetClass): An AssetClass object that describes the asset class of the data.
        symbols (Tuple[str]): A tuple of strings that lists the symbols for which the data is provided.
        resolution (str): A string that describes the resolution of the data.
        n_columns (int): An integer that describes the number of columns in the data stream.
    """

    stream_type: List[StreamType]
    column_schema: Dict[ColumnType, Tuple[bool]]
    asset_class: AssetClass
    symbols: Tuple[str]
    resolution: str
    n_columns: int

@dataclass
class DatasetMetadata:
    dataset_type: List[DatasetType]
    column_schema: Dict[ColumnType, Tuple[bool]]
    asset_class: AssetClass
    symbols: Tuple[str]
    start: datetime
    end: datetime
    resolution: str
    n_rows: int
    n_columns: int

    """
    A class that defines the metadata for a dataset.

    Attributes:
        dataset_type (List[DatasetType]): A list of DatasetType objects that describe the type of dataset.
        column_schema (Dict[ColumnType, Tuple[bool]]): A dictionary that maps ColumnType objects to tuples of boolean
            values, where the first boolean indicates whether the column is required and the second boolean indicates
            whether the column is nullable.
        asset_class (AssetClass): An AssetClass object that describes the asset class of the data.
        symbols (Tuple[str]): A tuple of strings that lists the symbols for which the data is provided.
        start (datetime): A datetime object that represents the start time of the data.
        end (datetime): A datetime object that represents the end time of the data.
        resolution (str): A string that describes the resolution of the data.
        n_rows (int): An integer that describes the number of rows in the dataset.
        n_columns (int): An integer that describes the number of columns in the dataset.

    Methods:
        __str__():
            Returns a string representation of the object.

        __or__(other):
            Joins datasets by performing type checking and updating metadata automatically.

        __add__(other):
            Appends datasets by performing type checking and updating metadata automatically.

        _join_column_schemas(other):
            Joins the column schemas of two datasets.

        _consecutive_dates(prev_end, cur_start):
            Checks if the dates between two datasets are consecutive.
    """


    def __str__(self):

        """
        Returns a string representation of the object.

        Returns:
            str: A string representation of the object that lists all attributes and their values.
        """

        attributes = [(attr, getattr(self, attr))
             for attr in self.__annotations__]
        
        return '\n'.join([
            f'{attr}: {value}' for attr, value in attributes])
    
    
    def __or__(self, other):

        """
        Joins datasets by performing type checking and updating metadata automatically.
        x | y automatically type checks and updates metadata of joined datasets.

        Args:
            other (DatasetMetadata): A DatasetMetadata object to join with the current object.

        Returns:
            DatasetMetadata: A new DatasetMetadata object that represents the joined datasets.

        Raises:
            ValueError: If the other object is not a DatasetMetadata object.
            ValueError: If the asset classes of the two objects are different.
            ValueError: If the symbols of the two objects are different.
            ValueError: If the start times of the two objects are different.
            ValueError: If the end times of the two objects are different.
            ValueError: If the resolutions of the two objects are different.
            ValueError: If the number of rows of the two objects are different.

        """

        if not isinstance(other, DatasetMetadata):
            raise ValueError('Only DatasetMetadata objects can be joined.')
        
        if self.asset_class != other.asset_class:
            raise ValueError('Datasets must have the same asset classes.')

        if self.symbols != other.symbols:
            raise ValueError('Datasets must have the same symbols.')
        
        if self.start != other.start:
            raise ValueError('Datasets do not have the same start dates.')

        if self.end != other.end:
            raise ValueError('Datasets do not have the same end dates.')

        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        if self.n_rows != other.n_rows:
            raise ValueError('Datasets must have the same number of rows.')

        dataset_type = self.dataset_type + other.dataset_type
        n_columns = self.n_columns + other.n_columns
        column_schema = self._join_column_schemas(other)


        return DatasetMetadata(
            dataset_type=dataset_type,
            column_schema=column_schema,
            asset_class=self.asset_class,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            resolution=self.resolution,
            n_rows=self.n_rows,
            n_columns=n_columns,
        )

    def __add__(self, other):

        """
        Combines two datasets with the same metadata into a single dataset.
        Only datasets with the same `dataset_type`, `column_schema`, `asset_class`,
        `symbols`, `resolution`, and `n_columns` can be combined.

        Args:
            other (DatasetMetadata): The second dataset to be combined with the first.

        Returns:
            DatasetMetadata: A new `DatasetMetadata` object representing the combined
            dataset.

        Raises:
            ValueError: If `other` is not a `DatasetMetadata` object.
            ValueError: If the `dataset_type` of the two datasets are not the same.
            ValueError: If the `column_schema` of the two datasets are not the same.
            ValueError: If the `asset_class` of the two datasets are not the same.
            ValueError: If the `symbols` of the two datasets are not the same.
            ValueError: If the start time of `other` overlaps with the end time of the current dataset.
            ValueError: If there are non-consecutive market hours between the end time of the current dataset
                       and the start time of `other`.
            ValueError: If the `resolution` of the two datasets are not the same.
            ValueError: If the number of columns of the two datasets are not the same.
        """

        if not isinstance(other, DatasetMetadata):
            raise ValueError('Only Dataset objects can be appended.')


        if self.dataset_type != other.dataset_type:
            raise ValueError(
                f'Dataset types {self.dataset_type} and {other.dataset_type} are mismatched.')


        if pickle.dumps(self.column_schema) != pickle.dumps(other.column_schema):
            raise ValueError(f'Datasets must have identical column schema.')


        if self.asset_class != other.asset_class:
            raise ValueError('Datasets must have the same asset classes.')


        if self.symbols != other.symbols:
            raise ValueError('Datasets must have the same symbols.')


        if other.start <= self.end:
            
            raise ValueError(
                f'Current end time: {self.end}, and appended dataset start time {other.start} overlap.')


        if not self._consecutive_dates(prev_end = self.end, cur_start = other.start):

            raise ValueError(
                f'Non-consecutive market hours between end time {self.end} and {self.start}.'
            )

        if self.resolution != other.resolution:
            raise ValueError(
                f'Dataset resolutions{self.resolution} and {other.resolution} are mismatched.')

        if self.n_columns != other.n_columns:
            raise ValueError('Dataset number of columns mismatch.')

        n_rows = self.n_rows + other.n_rows
        column_schema = self._join_column_schemas(other)

        return DatasetMetadata(
            dataset_type=self.dataset_type,
            column_schema=column_schema,
            asset_class=self.asset_class,
            symbols=self.symbols,
            start=self.start,
            end=other.end,
            resolution=self.resolution,
            n_rows=n_rows,
            n_columns=self.n_columns,
        )

    def _join_column_schemas(self, other):

        """Join column schemas of two DatasetMetadata objects.

        The method checks if the column schema keys of both objects match, and if they do, it merges the column schema
        of both objects and returns the result.

        Args:
            other (DatasetMetadata): Another DatasetMetadata object.

        Raises:
            ValueError: If the column schema keys of both objects do not match.

        Returns:
            dict: The merged column schema.
        """

        if set(self.column_schema.keys()) != set(other.column_schema.keys()):

            raise ValueError(
                'Datasets do not have matching column_schema structure.')

        merged_schema = dict()

        for key in self.column_schema.keys():
            
            merged_schema[key] = self.column_schema[
                key] + other.column_schema[key]

        return merged_schema
    
    def _consecutive_dates(self, prev_end, cur_start):

        """
        Determines whether two dates represent consecutive trading hours for the asset class.
        
        Args:
            prev_end (datetime.datetime): End time of the previous dataset.
            cur_start (datetime.datetime): Start time of the current dataset.

        Returns:
            bool: True if the two dates represent consecutive trading hours, False otherwise.
        """
        
        start_date = prev_end.date()
        end_date = cur_start.date()

        calendar = Calendar(self.asset_class)
        schedule = calendar.get_schedule(
            start_date= start_date, end_date= end_date)
        
        return True if len(schedule) == 2 else False

