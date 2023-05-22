from typing import List, Optional, Tuple
import os
from functools import reduce
import tarfile

import numpy as np
import dill
import h5py as h5

import torch
from torch import nn

from neural.common.constants import (HDF5_DEFAULT_MAX_ROWS, GLOBAL_DATA_TYPE)
from neural.common.exceptions import CorruptDataError
from neural.data.base import DatasetMetadata
from neural.meta.pipe import AbstractPipe
from neural.meta.agent import Agent
from neural.utils.base import validate_path


def to_hdf5(file_path: str | os.PathLike, numpy_array: np.ndarray,
            dataset_metadata: DatasetMetadata, dataset_name: str):
    """
    Saves a numpy array to an HDF5 file. If the file does not exist, new
    file will be created. If file exists and dataset already exists, the
    new data will be appended to the existing dataset. If the file
    exists but the dataset does not, a new dataset will be created.
    
    Args:
    -------
        file_path (str | os.PathLike): 
            The path to the HDF5 file to save the dataset to.
        numpy_array (np.ndarray): 
            The numpy array to save. The number of rows must match the
            number of rows
        dataset_metadata (DatasetMetadata):
            The metadata object for the dataset.
        dataset_name (str):
            The name of the dataset to save.
    Raises:
    -------
        ValueError: 
            If the number of rows in the numpy array does not match the
            number of rows in the metadata object.
    """

    validate_path(file_path=file_path)

    if len(numpy_array) != dataset_metadata.n_rows:
        raise ValueError(
            f'Number of rows in numpy array: {len(numpy_array)}.'
            f'Number of rows in metadata: {dataset_metadata.n_rows}')

    with h5.File(file_path, 'a') as hdf5:

        if dataset_name not in hdf5:
            dataset = hdf5.create_dataset(name=dataset_name,
                                          data=numpy_array,
                                          dtype=GLOBAL_DATA_TYPE,
                                          maxshape=(HDF5_DEFAULT_MAX_ROWS,
                                                    numpy_array.shape[1]),
                                          chunks=True)

            serialized_metadata = dill.dumps(dataset_metadata, protocol=0)
            dataset.attrs['metadata'] = serialized_metadata

        else:
            dataset_metadata_, dataset = extract_hdf5_dataset(
                hdf5_file=hdf5, dataset_name=dataset_name)

            new_dataset_metadata = dataset_metadata_ + dataset_metadata
            dataset.resize(
                (new_dataset_metadata.n_rows, new_dataset_metadata.n_columns))

            dataset[dataset_metadata_.n_rows:new_dataset_metadata.
                    n_rows, :] = numpy_array
            serialized_new_dataset_metadata = dill.dumps(new_dataset_metadata,
                                                 protocol=0)

            dataset.attrs['metadata'] = serialized_new_dataset_metadata

    return None


def from_hdf5(
    file_path: str | os.PathLike,
    dataset_name: Optional[str] = None
) -> Tuple[DatasetMetadata, List[h5.Dataset]]:
    """
    
    """

    validate_path(file_path=file_path)

    hdf5_file = h5.File(file_path, 'r')

    if dataset_name is not None:

        dataset_metadata, dataset = extract_hdf5_dataset(
            hdf5_file=hdf5_file, dataset_name=dataset_name)

        return dataset_metadata, [dataset]

    dataset_list = list()
    dataset_metadata_list = list()

    for dataset_name in hdf5_file:

        dataset_metadata, dataset = extract_hdf5_dataset(
            hdf5_file=hdf5_file, dataset_name=dataset_name)

        dataset_list.append(dataset)
        dataset_metadata_list.append(dataset_metadata)

    joined_metadata = reduce(lambda x, y: x | y, dataset_metadata_list)

    return joined_metadata, dataset_list


def extract_hdf5_dataset(
        hdf5_file: h5.File,
        dataset_name: str) -> Tuple[DatasetMetadata, h5.Dataset]:
    """
    
    """

    try:
        dataset = hdf5_file[dataset_name]
    except KeyError:
        raise ValueError(f'Dataset {dataset_name} does not exist in file.')

    serialized_metadata = dataset.attrs['metadata']
    metadata = dill.loads(serialized_metadata.encode())

    # sanity checking considtency between metadata and dataset
    if metadata.n_rows != len(dataset):
        raise CorruptDataError(f'Rows in {dataset_name}: {len(dataset)}.'
                               f'Rows in metadata: {metadata.n_rows}')

    if metadata.n_columns != dataset.shape[1]:
        raise CorruptDataError(f'Columns in {dataset_name}: {dataset.shape[1]}.'
                               f'Columns in metadata: {metadata.n_columns}')

    return metadata, dataset


def get_file_like(object: object,
                  file_name: str) -> Tuple[tarfile.TarInfo, dill.BytesIO]:
    # creates a file-like object from an object and tar info for that
    # object. can be used to add an object to a tarfile as a file.

    if isinstance(object, os.PathLike):
        path = object
        file = open(path, 'rb')
        file_tar_info = tarfile.TarInfo(name=file_name)
        file_tar_info.size = os.path.getsize(path)

    else:
        object_bytes = dill.dumps(object)
        file = dill.BytesIO(object_bytes)
        file_tar_info = tarfile.TarInfo(name='dataset_metadata')
        file_tar_info.size = len(object_bytes)

    return file_tar_info, file


def add_to_tarfile(file_path, file_tar_info, file_like):
    # adds a file to a tarfile. Useful for organizing files that need to
    # be bundled together for storage or transfer.
    validate_path(file_path=file_path)
    with tarfile.open(file_path, 'w') as file:

        file.addfile(tarinfo=file_tar_info, fileobj=file_like)


def save_agent(file_path: str | os.PathLike, agent: Agent):

    model = agent.model
    pipe = agent.pipe
    dataset_metadata = agent.dataset_metadata

    dataset_metadata_tar_info, dataset_metadata_file = get_file_like(
        dataset_metadata, 'dataset_metadata')
    add_to_tarfile(file_path, dataset_metadata_tar_info, dataset_metadata_file)

    pipe_tar_info, pipe_file = get_file_like(dataset_metadata, 'pipe')
    add_to_tarfile(file_path, pipe_tar_info, pipe_file)

    model_file_path = os.path.join(os.path.dirname(file_path), 'model')
    torch.save(model, model_file_path)
    model_tar_info, model_file = get_file_like(model_file_path, 'model')
    add_to_tarfile(file_path, model_tar_info, model_file)

    # a copy model file is now in the agent file_path.
    os.remove(os.path.join(os.path.dirname(file_path), 'model'))


def load_agent(
    file_path: str | os.PathLike,
) -> Tuple[nn.Module, AbstractPipe, DatasetMetadata]:

    with tarfile.open(file_path, 'r') as agent_file:

        # Load dataset metadata
        dataset_metadata_file = agent_file.extractfile('dataset_metadata')
        dataset_metadata_bytes = dataset_metadata_file.read()
        dataset_metadata = dill.loads(dataset_metadata_bytes)

        # Load pipe class definition
        pipe_file = agent_file.extractfile('pipe')
        pipe_bytes = pipe_file.read()
        pipe_class = dill.loads(pipe_bytes)

        # Load model
        with tarfile.open(mode='r', fileobj=agent_file) as inner_tar:
            model_file = inner_tar.extractfile('model')
            model = torch.load(model_file)

    # Initialize pipe
    pipe = pipe_class()

    return Agent(model=model, pipe=pipe, dataset_metadata=dataset_metadata)
