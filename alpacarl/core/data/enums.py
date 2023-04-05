from enum import Enum
from datetime import datetime
from typing import (List, Dict, Tuple)
from dataclasses import dataclass

from alpaca.trading.enums import AssetClass


class DatasetType(Enum):
    BAR = 'BAR'
    QUOTE = 'QUOTE'
    TRADE = 'TRADE'
    ORDER_BOOK = 'ORDER_BOOK'

class ColumnType(Enum):
    PRICE = 'PRICE'
    OPEN = 'OPEN'
    HIGH = 'HIGH'
    LOW = 'LOW'
    CLOSE = 'CLOSE'
import pandas_market_calendars as market_calendars


@dataclass
class DatasetMetadata:
    dataset_type: List[DatasetType]
    column_schema = Dict[ColumnType, Tuple[bool]]
    asset_class = AssetClass
    symbols = Tuple(str)
    start: datetime
    end: datetime
    resolution: str
    n_rows: int
    n_columns: int

    def __or__(self, other):
        # For automatic type checking and metadata generation when joining datasets
        # x | y automatically type checks and updates metadata of joined datasets.

        # checking compatibility
        if not isinstance(other, DatasetMetadata):
            raise ValueError('Only Dataset objects can be chained.')

        if other.dataset_type in self.dataset_type:
            raise ValueError(
                'Duplicate dataset type is not allowed in horizontal chaining.')

        if self.asset_class != other.asset_class:
            raise ValueError('Datasets must have the same asset classes.')

        if self.symbols != other.symbols:
            raise ValueError('Datasets must have the same symbols.')

        if self.resolution != other.resolution:
            raise ValueError('Datasets must have the same resolution.')

        if self.n_rows != other.n_rows:
            raise ValueError('Datasets must have the same number of rows.')

        if self.start != other.start:
            raise ValueError('Datasets do not have the same start dates.')

        if self.end != other.end:
            raise ValueError('Datasets do not have the same end dates.')

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
            n_columns=n_columns)

    def __and__(self, other):
        # For automatic type checking and metadata generation when appending to datasets
        # x + y automatically type checks and updates metadata of appended datasets.

        # checking compatibility
        if not isinstance(other, DatasetMetadata):
            raise ValueError('Only Dataset objects can be appended.')

        if self.dataset_type != other.dataset_type:
            raise ValueError(
                f'Dataset types {self.dataset_type} and {other.dataset_type} are mismatched.')

        if self.column_schema != other.column_schema:
            raise ValueError(f'Datasets must have identical column schema.')

        if self.asset_class != other.asset_class:
            raise ValueError('Datasets must have the same asset classes.')

        if self.symbols != other.symbols:
            raise ValueError('Datasets must have the same symbols.')

        if other.start <= self.end:
            raise ValueError(
                f'Cannot perform append. End time: {self.end}, and start time: {other.start} overlap.')

        if abs(self.end.date() - other.start.date()).days != 1:
            raise ValueError(
                f'End date {self.end} and start date {other.start}  are not 1 day apart.')

        if self.resolution != other.resolution:
            raise ValueError(
                f'Dataset resolutions{self.resolution} and {other.resolution} are mismatched.')

        if self.n_columns != other.n_columns:
            raise ValueError('Dataset number of columns mismatch.')

        dataset_type = self.dataset_type + other.dataset_type
        n_columns = self.n_columns + other.n_columns
        column_schema = self._join_column_schemas(other)

        return DatasetMetadata(
            dataset_type=dataset_type,
            start=self.start,
            end=self.end,
            symbols=self.symbols,
            resolution=self.resolution,
            n_rows=self.n_rows,
            n_columns=n_columns,
            column_schema=column_schema)

    def _join_column_schemas(self, other):
        if set(self.column_schema.keys()) != set(other.column_schema.keys()):
            raise ValueError(
                'Datasets do not have matching column_schema structure.')

        merged_schema = dict()

        for key in self.column_schema.keys():
            merged_schema[key] = self.column_schema[key] + \
                other.column_schema[key]

        return merged_schema
