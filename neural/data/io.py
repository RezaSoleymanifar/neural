from typing import List, Optional, Tuple
import os
from functools import reduce

import numpy as np
import pickle
import h5py as h5

from neural.common.constants import HDF5_DEFAULT_MAX_ROWS
from neural.common.exceptions import CorruptDataError
from neural.data.base import DatasetMetadata
from neural.tools.base import validate_path




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
