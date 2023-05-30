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

            serialized_dataset_metadata = dill.dumps(dataset_metadata,
                                                     protocol=0)
            dataset.attrs['metadata'] = serialized_dataset_metadata

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
    Loads a dataset from an HDF5 file. If dataset_name is not specified,
    all datasets in the file will be loaded. If dataset_name is
    specified, only the dataset with the specified name will be loaded.
    If all datasets are loaded, the datasets will be returned in a list
    and the metadata will be joined. If only one dataset is loaded, the
    dataset will be returned as a list and the metadata will be returned
    as a single object.

    Args:
    -------
        file_path (str | os.PathLike):
            The path to the HDF5 file to load the dataset from.
        dataset_name (Optional[str]):
            The name of the dataset to load. If None, all datasets in
            the
    Returns:
    --------
        dataset_metadata (DatasetMetadata):
            The metadata of the dataset(s).
        dataset_list (List[h5.Dataset]):
            The dataset(s) loaded from the file.
    
    Examples:
    ---------
        Assume x, y are metadata objects corresponding to two datasets
        in the hdf5 file. Then the loaded metadata will be x | y and the
        loaded datasets will be [dataset_x, dataset_y]. More on joining
        metadata objects: neural/data/base.py
    """

    validate_path(file_path=file_path)

    hdf5_file = h5.File(file_path, 'r')

    if dataset_name is not None:

        dataset_metadata, dataset = extract_hdf5_dataset(
            hdf5_file=hdf5_file, dataset_name=dataset_name)

        return dataset_metadata, [dataset]

    dataset_list = list()
    dataset_metadata_list = list()

    sorted_hdf5_file = sorted(
        hdf5_file, key=lambda dataset: hdf5_file[dataset].id.get_offset())
    for dataset_name in sorted_hdf5_file:

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
    Extracts a dataset from an HDF5 file and returns the dataset and its
    metadata. 

    Args:
    -------
        hdf5_file (h5.File):
            The HDF5 file to extract the dataset from.
        dataset_name (str):
            The name of the dataset to extract.
    Raises:
    ------- 
        ValueError:
            If the dataset does not exist in the file.
        CorruptDataError:   
            If the number of rows or columns in the dataset does not
            match the number of rows or columns in the metadata.    
    """

    try:
        dataset = hdf5_file[dataset_name]
    except KeyError:
        raise ValueError(f'Dataset {dataset_name} does not exist in file.')

    serialized_dataset_metadata = dataset.attrs['metadata']
    dataset_metadata = dill.loads(serialized_dataset_metadata.encode())

    if dataset_metadata.n_rows != len(dataset):
        raise CorruptDataError(f'Rows in {dataset_name}: {len(dataset)}.'
                               f'Rows in metadata: {dataset_metadata.n_rows}')
    if dataset_metadata.n_columns != dataset.shape[1]:
        raise CorruptDataError(
            f'Columns in {dataset_name}: {dataset.shape[1]}.'
            f'Columns in metadata: {dataset_metadata.n_columns}')

    return dataset_metadata, dataset


def get_file_like(object: object,
                  file_name: str) -> Tuple[tarfile.TarInfo, dill.BytesIO]:
    """
    Creates a file-like object and its tar info from an object. This can
    be used to add an object to a tarfile as a file. Similarly if object
    is a path (file path) then it can create a file like object for the
    existing file in path and its tar info.

    Args:
    -------
        object (object | os.PathLike):
            The object to create a file-like object from.
        file_name (str):
            The name of the file to create.
    Returns:
    --------
        file_tar_info (tarfile.TarInfo):
            The tar info of the file.
        file (dill.BytesIO):
            The file-like object.
    """

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
    """
    Adds a file-like object to a tarfile. The file-like object can be
    created using the get_file_like function. Then this function can be
    used to add the file to the tarfile specified by file_path.

    Args:
    -------
        file_path (str | os.PathLike):
            The path to the tarfile to add the file to.
        file_tar_info (tarfile.TarInfo):
            The tar info of the file to add.
        file_like (dill.BytesIO):
            The file-like object to add.
    """

    validate_path(file_path=file_path)
    with tarfile.open(file_path, 'w') as file:

        file.addfile(tarinfo=file_tar_info, fileobj=file_like)

    return None


def save_agent(file_path: str | os.PathLike, agent: Agent):
    """
    A function to save an agent to a tarfile. The agent is saved as a
    tarfile with the following structure:
    agent.tar
    ├── dataset_metadata
    ├── pipe
    └── model.tar
        └── model

    Args:
    -------
        file_path (str | os.PathLike):
            The path to the tarfile to save the agent to.
        agent (Agent):
            The agent to save.
    """
    model = agent.model
    pipe = agent.pipe
    dataset_metadata = agent.dataset_metadata

    dataset_metadata_tar_info, dataset_metadata_file = get_file_like(
        dataset_metadata, 'dataset_metadata')
    add_to_tarfile(file_path, dataset_metadata_tar_info, dataset_metadata_file)

    pipe_tar_info, pipe_file = get_file_like(pipe, 'pipe')
    add_to_tarfile(file_path, pipe_tar_info, pipe_file)

    model_file_path = os.path.join(os.path.dirname(file_path), 'model')
    torch.save(model, model_file_path)
    model_tar_info, model_file = get_file_like(model_file_path, 'model')
    add_to_tarfile(file_path, model_tar_info, model_file)

    os.remove(os.path.join(os.path.dirname(file_path), 'model'))


def load_agent(
    file_path: str | os.PathLike,
) -> Tuple[nn.Module, AbstractPipe, DatasetMetadata]:
    """
    Loads an agent from a tarfile. The agent is saved as a tarfile with
    the following structure:
    agent.tar
    ├── dataset_metadata
    ├── pipe
    └── model.tar
        └── model

    Args:
    -------
        file_path (str | os.PathLike):
            The path to the tarfile to load the agent from.
    Returns:
    --------
        model (nn.Module):
            The model of the agent.
        pipe (AbstractPipe):
            The pipe of the agent.
        dataset_metadata (DatasetMetadata):
            The metadata of the dataset used to train the agent.
    """
    with tarfile.open(file_path, 'r') as agent_file:

        dataset_metadata_file = agent_file.extractfile('dataset_metadata')
        dataset_metadata_bytes = dataset_metadata_file.read()
        dataset_metadata = dill.loads(dataset_metadata_bytes)

        pipe_file = agent_file.extractfile('pipe')
        pipe_bytes = pipe_file.read()
        pipe = dill.loads(pipe_bytes)

        with tarfile.open(mode='r', fileobj=agent_file) as inner_tar:
            model_file = inner_tar.extractfile('model')
            model = torch.load(model_file)

    return Agent(model=model, pipe=pipe, dataset_metadata=dataset_metadata)
