from enum import Enum
from datetime import datetime
from typing import (List, Dict, Tuple)
from dataclasses import dataclass
import pickle

from alpaca.trading.enums import AssetClass
from neural.tools.misc import Calendar

class StreamType(Enum):
    BAR = 'BAR'
    QUOTE = 'QUOTE'
    TRADE = 'TRADE'
    
class DatasetType(Enum):
    BAR = 'BAR'
    QUOTE = 'QUOTE'
    TRADE = 'TRADE'

class ColumnType(Enum):
    OPEN = 'OPEN'
    HIGH = 'HIGH'
    LOW = 'LOW'
    CLOSE = 'CLOSE'
    BID = 'BID'
    ASK = 'ASK'

@dataclass
class StreamMetaData:
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


    def __str__(self):

        attributes = [(attr, getattr(self, attr))
             for attr in self.__annotations__]
        
        return '\n'.join([
            f'{attr}: {value}' for attr, value in attributes])
    
    
    def __or__(self, other):
        # For automatic type checking and metadata generation when joining datasets
        # x | y automatically type checks and updates metadata of joined datasets.

        # checking compatibility
        if not isinstance(other, DatasetMetadata):
            raise ValueError('Only DatasetMetadata objects can be joined.')

        if other.dataset_type in self.dataset_type:
            raise ValueError(
                f'Dataset types {self.dataset_type} and {other.dataset_type} are mismatched.')
        
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
        # For automatic type checking and metadata generation when appending datasets
        # x + y automatically type checks and updates metadata of appended datasets.

        # checking compatibility
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

        if set(self.column_schema.keys()) != set(other.column_schema.keys()):

            raise ValueError(
                'Datasets do not have matching column_schema structure.')

        merged_schema = dict()

        for key in self.column_schema.keys():
            
            merged_schema[key] = self.column_schema[
                key] + other.column_schema[key]

        return merged_schema
    
    def _consecutive_dates(self, prev_end, cur_start):

        start_date = prev_end.date()
        end_date = cur_start.date()

        calendar = Calendar(self.asset_class)
        schedule = calendar.get_schedule(
            start_date= start_date, end_date= end_date)
        
        return True if len(schedule) == 2 else False

