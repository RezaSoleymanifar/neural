from functools import reduce
from datetime import datetime
from typing import Dict, Tuple, Iterable, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict

import h5py as h5
import pickle
import pandas as pd
import numpy as np

from neural.utils.time import Calendar
from neural.data.enums import AbstractDataSource, FeatureType
from neural.client.base import AbstractDataClient


class AssetType(str, Enum):

    """
    An enum class representing different categories of financial instruments.

    STOCK: Represents ownership in a publicly traded corporation. Stocks can be traded on stock exchanges, 
    and their value can fluctuate based on various factors such as the company's financial performance and market conditions.

    CURRENCY: Represents a unit of currency, such as the US dollar, Euro, or Japanese yen. Currencies can be traded on the 
    foreign exchange market (Forex), and their value can fluctuate based on various factors such as interest rates, g
    eopolitical events, and market sentiment.

    CRYPTOCURRENCY: Represents a digital or virtual currency that uses cryptography for security and operates 
    independently of a central bank. Cryptocurrencies can be bought and sold on various cryptocurrency exchanges, 
    and their value can fluctuate based on various factors such as supply and demand, adoption rates, and regulatory developments.

    FUTURES: Represents a standardized contract to buy or sell an underlying asset at a predetermined price and date in the future. 
    Futures can be traded on various futures exchanges, and their value can fluctuate based on various factors such as supply and demand, 
    geopolitical events, and market sentiment.

    OPTIONS: Represents a financial contract that gives the buyer the right, but not the obligation, to buy or sell an underlying 
    asset at a specified price on or before a specified date. Options can be traded on various options exchanges, 
    and their value can fluctuate based on various factors such as the price of the underlying asset, 
    the time until expiration, and market volatility.

    BOND: Represents debt issued by a company or government entity. Bonds can be traded on various bond markets, 
    and their value can fluctuate based on various factors such as interest rates, creditworthiness, and market conditions.

    EXCHANGE_TRADED_FUND: Represents a type of investment fund traded on stock exchanges, similar to mutual funds, 
    but with shares that can be bought and sold like individual stocks. ETFs can provide exposure to a wide range of asset classes, 
    such as stocks, bonds, and commodities, and their value can fluctuate based on various factors such as the performance of 
    the underlying assets and market conditions.

    MUTUAL_FUND: Represents a professionally managed pool of money from many investors, used to purchase a diversified mix of stocks, 
    bonds, or other assets. Mutual funds can be bought and sold through fund companies or brokerages, and their value can 
    fluctuate based on various factors such as the performance of the underlying assets and market conditions.

    COMMODITY: Represents a physical or virtual product that can be bought or sold, such as gold, oil, or currencies. 
    Commodities can be traded on various commodity exchanges, and their value can fluctuate based on various factors 
    such as supply and demand, geopolitical events, and market sentiment.
    """

    STOCK = 'STOCK'
    CURRENCY = 'CURRENCY'
    CRYPTOCURRENCY = 'CRYPTOCURRENCY'
    FUTURES = 'FUTURES'
    OPTIONS = 'OPTIONS'
    BOND = 'BOND'
    EXCHANGE_TRADED_FUND = 'ETF'
    MUTUAL_FUND = 'MUTUAL_FUND'
    COMMODITY = 'COMMODITY'



@dataclass
class Asset:

    """
    A dataclass representing a financial asset.

    Attributes:
        symbol: A string representing the symbol or ticker of the asset.
        asset_type: An instance of the `AssetType` enum class representing the type of asset.
        marginable: A boolean indicating whether the asset can be bought on margin (i.e., borrowed funds).
        fractionable: A boolean indicating whether the asset can be traded in fractional shares.
        shortable: A boolean indicating whether the asset can be sold short (i.e., sold before buying to profit from a price decrease).
        initial_margin: A float representing the initial margin required to trade the asset on margin.
        maintenance_margin: A float representing the maintenance margin required to keep the asset in a margin account.
    """

    symbol: str
    asset_type: AssetType
    marginable: bool
    fractionable: bool
    shortable: bool
    initial_margin: float
    maintenance_margin: float

    


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

    data_schema: Dict[AbstractDataSource.DatasetType: Tuple[Asset]] | Dict[AbstractDataSource.StreamType: Tuple[Asset]]
    feature_schema: Dict[FeatureType, Tuple[bool]]
    resolution: str


    @property
    def assets(self) -> List[Asset]:
        # returns a list of unique assets in the data schema. Order is preserved.
        assets = reduce(lambda x, y: x + y, self.data_schema.values())
        symbols = OrderedDict()
        for asset in assets:
            symbols[asset.symbol] = asset
        assets = list(symbols.values())
        return assets
    

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

        valid = True if len(self.assets) == self.asset_prices_mask.count(True) else False
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


    @property
    def n_rows(self) -> int:

        """
        Just to provide a consistent interface with DatasetMetaData and avoid
        boilerplate code.
        """

        n_rows = np.inf
        return n_rows




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
        Returns the number of columns in the dataset. This is useful for
        checking if the dataset has been downloaded correctly.
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
            symbols=self.assets,
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


        schedule = Calendar.schedule(start_date= start_date, end_date= end_date)
        
        return True if len(schedule) == 2 else False
    
    @property
    def stream(self):
        stream = StreamMetaData(
            data_schema=self.data_schema,
            feature_schema=self.feature_schema,
            resolution=self.resolution)
        return stream


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



class StaticDataFeeder(AbstractDataFeeder):

    """
    Subclass of AbstractStaticDataFeeder that iteratively returns data required for 
    the environment from a static source.
    """

    def __init__(
            self,
            dataset_metadata: DatasetMetadata,
            datasets: List[h5.Dataset | np.ndarray],
            start_index: int = 0,
            end_index: Optional[int] = None,
            n_chunks: Optional[int] = 1) -> None:
        
        """
        Initializes a StaticDataFeeder object.
        Args:
        dataset_metadata (DatasetMetadata): Contains metadata for the dataset being loaded.
        datasets (List[h5.Dataset | np.ndarray]): Represents the actual dataset(s) to be loaded.
        start_index (int, optional): Specifies the starting index to load the data from. Default is 0.
        end_index (int, optional): Specifies the ending index to load the data from. If not provided,
        defaults to the number of rows indicated in the metadata object. Default is None.
        n_chunks (int, optional): Indicates the number of chunks to divide the dataset into for
        loading. Loads one chunk at a time. Useful if datasets do not fit in memory or to
        allocate more memory for the training process. Default is 1.
        """

        self.dataset_metadata = dataset_metadata
        self.datasets = datasets
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else self.dataset_metadata.n_rows
        self.n_rows = self.end_index - self.start_index
        self.n_columns = self.dataset_metadata.n_columns
        self.n_chunks = n_chunks

        return None

    def reset(self) -> Iterable[np.ndarray]:
        """
        Resets the internal state of the data feeder.
        Yields:
            Iterable[np.ndarray]: a generator object returning features as numpy array.
        """

        chunk_edge_indices = np.linspace(
            start=self.start_index,
            stop=self.end_index,
            num=self.n_chunks+1,
            dtype=int,
            endpoint=True)

        for start, end in zip(chunk_edge_indices[:-1], chunk_edge_indices[1:]):

            joined_chunks_in_memory = np.hstack([dataset[start:end, :]
                                                 for dataset in self.datasets])

            for row in joined_chunks_in_memory:
                yield row

    def split(self, n: int = 1 | float):
        """
        Splits the dataset into multiple non-overlapping contiguous sub-feeders that span the dataset.
        Common use case is it use in stable baselines vector env to parallelize running multiple
        trading environments, leading to significant speedup of training process.
        Args:
            n (int | float): if int, number of sub-feeders to split the dataset into. if float (0, 1)
            yields two sub-feeders performing n, 1-n train test split
        Returns:
            List[StaticDataFeeder]: A list of StaticDataFeeder objects.
        """
        
        if isinstance(n, int):

            if not n > 0:
                raise ValueError("n must be a positive integer")

            edge_indices = np.linspace(
                start=self.start_index,
                stop=self.end_index,
                num=self.n+1,
                dtype=int,
                endpoint=True)

        elif isinstance(n, float):
            
            if not 0 < n <= 1:
                raise ValueError("n must be a float in (0, 1]")

            edge_indices = np.array([
                self.start_index,
                int(self.start_index + n * (self.end_index - self.start_index)),
                self.end_index
            ], dtype=int)

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
    def __init__(
        self, 
        stream_metadata: StreamMetaData, 
        data_client: AbstractDataClient
        ) -> None:
        super().__init__()