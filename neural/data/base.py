from datetime import datetime
from typing import (List, Optional, Tuple, Iterable)
from functools import reduce
from abc import ABC, abstractmethod
import os

import pandas as pd
import numpy as np
import pickle
import h5py as h5

from neural.common.constants import HDF5_DEFAULT_MAX_ROWS
from neural.common.exceptions import CorruptDataError
from neural.data.enums import DatasetMetadata, StreamMetaData
from neural.client.base import AbstractDataClient
from neural.tools.base import validate_path



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
    


    

class DataProcessor:

    def __init__(self):

        self.dataset_density = None

    def reindex_and_forward_fill(
            self,
            data: pd.DataFrame,
            open: datetime,
            close: datetime,
            resolution: str):
        
        """
        Reindexes and forward-fills missing rows in the given DataFrame in the [open, close) range based on the given
        resolution. Returns the processed DataFrame.

        :param data: The DataFrame to be processed.
        :type data: pd.DataFrame
        :param open: The open time of the market data interval to process.
        :type open: datetime
        :param close: The close time of the market data interval to process.
        :type close: datetime
        :param resolution: The frequency of the time intervals in the processed data.
        :type resolution: str
        :return: The processed DataFrame.
        :rtype: pd.DataFrame
        """

        # resamples and forward fills missing rows in [open, close) range, i.e.
        # time index = open means open <= time < close.
        index = pd.date_range(
            start=open, end=close, freq=resolution, inclusive='left')

        # creates rows for missing intervals
        processed = data.reindex(index)

        # compute fullness of reindexed dataset
        # drop symbols or move date range if density is low
        non_nan_count = processed.notna().sum().sum()
        total_count = processed.size
        density = non_nan_count/total_count

        DataProcessor.dataset_density = (
            DataProcessor.dataset_density + density
        ) / 2 if DataProcessor.dataset_density else density

        # backward fills if first row is nan
        if processed.isna().any().any():
            processed = processed.ffill()

        # backward fills if first row is nan
        if processed.isna().any().any():
            processed = processed.bfill()

        return processed



class DataStreamer:
    # responsible for receiving stream metadata and a set of data clients that 
    # will be used to stream data from the data source.
    def __init__(self, stream_metadata: StreamMetaData, data_clients = List[AbstractDataClient]) -> None:
        pass