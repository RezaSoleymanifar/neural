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
from neural.data.enums import DatasetMetadata
from neural.tools.base import validate_path



class DataFeeder(ABC):

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
    

class AbstractStaticDataFeeder(ABC):

    """
    Abstract base class for defining a static data feeder that is responsible for feeding
    market information to market environment, iteratively.
    """



    @abstractmethod
    def split(self, *args, **kwargs):
        """
        Returns instances of iteself each initialized with a slice of source data.
        
        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        raise NotImplementedError



class AbstractAsyncDataFeeder(ABC):

    """
    Abstract base class for defining an asynchronous data feeder that can be used with async iterators.

    Methods:
        __aiter__(): Abstract asynchronous method that returns an async iterator for the data.
        
    Raises:
        NotImplementedError: If the __aiter__() method is not implemented by a subclass.
    """

    @abstractmethod
    async def __aiter__(self):
        pass


class AsyncDataFeeder(AbstractAsyncDataFeeder):
    """
    Subclass of AbstractAsyncDataFeeder that streams and iteratively aggregates live data.

    Methods:
        __aiter__(): Asynchronous method that returns an async iterator for the live data.

    """
    pass


class StaticDataFeeder(AbstractStaticDataFeeder):

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

    def split(self, n: int | float):
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

            assert n > 0, "n must be a positive integer"

            edge_indices = np.linspace(
                start=self.start_index,
                stop=self.end_index,
                num=self.n+1,
                dtype=int,
                endpoint=True)

        elif isinstance(n, float):

            assert 0 < n < 1, "n must be a float between 0 and 1"

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


class DatasetIO:


    def to_hdf5(
            file_path: str | os.PathLike,
            data_to_write: np.ndarray,
            metadata: DatasetMetadata,
            target_dataset_name: str):
        
        """
        Write data to an HDF5 file and update metadata.

        Args:
            file_path (str | os.PathLike): The file path of the HDF5 file.
            data_to_write (np.ndarray): The data to write to the HDF5 file.
            metadata (DatasetMetadata): The metadata of the dataset being written.
            target_dataset_name (str): The name of the dataset to write to in the HDF5 file.

        Returns:
            None
        """

        validate_path(file_path=file_path)

        with h5.File(file_path, 'a') as hdf5:

            if target_dataset_name not in hdf5:
                # Create a fixed-size dataset with a predefined data type and dimensions
                target_dataset = hdf5.create_dataset(
                    name=target_dataset_name, data=data_to_write,
                    dtype=np.float32, maxshape=(HDF5_DEFAULT_MAX_ROWS,
                                                data_to_write.shape[1]), chunks=True)

                serialized_metadata = pickle.dumps(metadata, protocol=0)
                target_dataset.attrs['metadata'] = serialized_metadata

            else:

                target_dataset_metadata, target_dataset = DatasetIO._extract_hdf5_dataset(
                    hdf5=hdf5, target_dataset_name=target_dataset_name)

                # Append the new data to the dataset and update metadata
                new_metadata = target_dataset_metadata + metadata
                target_dataset.resize(
                    (new_metadata.n_rows, new_metadata.n_columns))

                target_dataset[
                    target_dataset_metadata.n_rows: new_metadata.n_rows, :] = data_to_write
                serialized_new_metadata = pickle.dumps(
                    new_metadata, protocol=0)
                target_dataset.attrs['metadata'] = serialized_new_metadata

        return None


    def _extract_hdf5_dataset(
        hdf5: h5.File,
        target_dataset_name: str
        ) -> Tuple[DatasetMetadata, h5.Dataset]:
        """
        Extracts a target dataset and its metadata from an HDF5 file.

        Args:
            hdf5 (h5.File): The HDF5 file object to extract the dataset from.
            target_dataset_name (str): The name of the target dataset to extract.

        Returns:
            Tuple[DatasetMetadata, h5.Dataset]: A tuple containing the metadata object for the
                target dataset and the target dataset object as a h5py dataset object.

        Raises:
            CorruptDataError: If the number of rows or columns specified in the metadata object
                does not match the actual number of rows or columns in the target dataset.
        """

        target_dataset = hdf5[target_dataset_name]
        serialized_metadata = target_dataset.attrs['metadata']
        metadata = pickle.loads(serialized_metadata.encode())

        # corrupt data check
        if metadata.n_rows != len(target_dataset):
            raise CorruptDataError(
                f'Rows in {target_dataset_name}: {len(target_dataset)}.'
                f'Rows in metadata: {metadata.n_rows}')

        if metadata.n_columns != target_dataset.shape[1]:
            raise CorruptDataError(
                f'Columns in {target_dataset_name}: {target_dataset.shape[1]}.'
                f'Columns in metadata: {metadata.n_columns}')

        return metadata, target_dataset

    def load_from_hdf5(
        file_path: str | os.PathLike,
        target_dataset_name: Optional[str] = None
    ) -> Tuple[DatasetMetadata, List[h5.Dataset]]:
        """
        Loads one or more datasets from an HDF5 file and returns a tuple containing their metadata
        and the datasets themselves as h5py dataset objects.

        Args:
            file_path (str | os.PathLike): The path to the HDF5 file to load the dataset(s) from.
            target_dataset_name (Optional[str]): The name of the target dataset to load. If not
                provided, all datasets in the file will be loaded.

        Returns:
            Tuple[DatasetMetadata, List[h5.Dataset]]: A tuple containing the metadata object for the
                loaded dataset(s) and the loaded dataset(s) as h5py dataset objects.

        Raises:
            CorruptDataError: If the number of rows or columns specified in the metadata object
                does not match the actual number of rows or columns in the target dataset.
            FileNotFoundError: If the file_path does not exist.
            ValueError: If the file_path is not a valid HDF5 file.

        """

        validate_path(file_path=file_path)

        hdf5 = h5.File(file_path, 'r')

        if target_dataset_name is not None:

            metadata, dataset = DatasetIO._extract_hdf5_dataset(
                hdf5=hdf5, target_dataset_name=target_dataset_name)

            return metadata, [dataset]

        dataset_list = list()
        metadata_list = list()

        for dataset_name in hdf5:

            metadata, dataset = DatasetIO._extract_hdf5_dataset(
                hdf5=hdf5, target_dataset_name=dataset_name)

            dataset_list.append(dataset)
            metadata_list.append(metadata)

        joined_metadata = reduce(lambda x, y: x | y, metadata_list)

        return joined_metadata, dataset_list

    

class DataProcessor:

    def __init__(self):
        self.dataset_density = 0

    def resample(
        dataset_type,
        start: datetime,
        end: datetime,
        resolution: str
        ) -> None:

        pass

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


    def create_column_schema(data: pd.DataFrame):
        """
        Creates a column schema dictionary for a given DataFrame, with ColumnType as keys and boolean masks as values.
        Args:
            data (pd.DataFrame): The input DataFrame for which the column schema is to be created.

        Returns:
            Dict[ColumnType, pd.Series]: A dictionary containing ColumnType keys and boolean masks for each column in the input DataFrame.
        """

        column_schema = dict()

        for column_type in ColumnType:

            mask = data.columns.str.match(column_type.value.lower())
            column_schema[column_type] = mask

        return column_schema
