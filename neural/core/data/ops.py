from datetime import datetime
from typing import (List, Optional, Tuple, Any)
from functools import reduce
import os
from abc import ABC, abstractmethod

from alpaca.trading.enums import AssetClass, AssetStatus
import pandas as pd
import numpy as np
import pickle
import h5py as h5

from alpaca.trading.enums import AssetClass, AssetStatus

from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoQuotesRequest,
    CryptoTradesRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest
)

from neural.common import logger
from neural.common.exceptions import CorruptDataError
from neural.core.data.enums import DatasetType, DatasetMetadata
from neural.connect.client import AlpacaMetaClient
from neural.tools.ops import (progress_bar, to_timeframe, 
    create_column_schema, validate_path)
from neural.tools.misc import Calendar
from neural.common.constants import HDF5_DEFAULT_MAX_ROWS
    
class DataFetcher():
    def __init__(
        self,
        client: AlpacaMetaClient
        ) -> None:

        self.client = client

        return None

    def _validate_resolution(self, resolution):

        """
        Validates the resolution of the dataset.

        Parameters:
        resolution (str): The resolution of the dataset.

        Returns:
        None.

        Raises:
        ValueError: If the resolution is not one of the accepted resolutions.
        """

        accepted_resolutions = {'1Min', '5Min', '15Min', '30Min'}

        if resolution not in accepted_resolutions:
            raise ValueError(f'Accepted resolutions: {accepted_resolutions}.')
        
        return
        
        
    def _validate_symbols(self, symbols: List[str]):

        """
        Validates the list of symbols.

        Args:
        - symbols (List[str]): The list of symbols to validate.

        Returns:
        - str: The asset class of the symbols.

        Raises:
        - ValueError: If the symbols argument is an empty sequence.
                      If any symbols have duplicate values.
                      If any symbol is not a known symbol.
                      If any symbol is not a tradable symbol.
                      If any symbol is not an active symbol.
                      (warning only) if any symbol is not a fractionable symbol.
                      (warning only) if any symbol is not easy to borrow (ETB).
                      If the symbols are not of the same asset class.
        """

        if len(symbols) == 0:
            raise ValueError('symbols argument cannot be an empty sequence.')

        duplicate_symbols = [
            symbol for symbol in set(symbols) if symbols.count(symbol) > 1]
        
        if duplicate_symbols:
            raise ValueError(f'Symbols {duplicate_symbols} have duplicate values.')


        for symbol in symbols:

            symbol_data = self.client._AlpacaMetaClient__symbols.get(symbol)

            if symbol_data is None:
                raise ValueError(f'Symbol {symbol} is not a known symbol.')

            if not symbol_data.tradable:
                raise ValueError(f'Symbol {symbol} is not a tradable symbol.')

            if symbol_data.status != AssetStatus.ACTIVE:
                raise ValueError(f'Symbol {symbol} is not an active symbol.')

            if not symbol_data.fractionable:
                logger.warning(f'Symbol {symbol} is not a fractionable symbol.')

            if not symbol_data.easy_to_borrow:
                logger.warning(
                    f'Symbol {symbol} is not easy to borrow (ETB).')

        asset_classes = set(
            self.client._AlpacaMetaClient__symbols.get(
            symbol).asset_class for symbol in symbols)


        # checks if symbols have the same asset class
        if len(asset_classes) != 1:
            raise ValueError('Symbols are not of the same asset class.')
        
        asset_class = asset_classes.pop()

        return asset_class

    def get_downloader_and_request(
        self, 
        dataset_type: DatasetType, 
        asset_class = AssetClass
        ) -> Tuple[Any, Any]:
        

        """
        Returns the appropriate data downloader and request object based on the provided dataset type
        and asset class.

        Parameters:
        -----------
        dataset_type: DatasetType
            The type of dataset being downloaded, one of ['BAR', 'QUOTE', 'TRADE'].
        asset_class: AssetClass, optional
            The asset class being downloaded, defaults to `AssetClass.US_EQUITY`.

        Returns:
        --------
        Tuple[Any, Any]
            A tuple containing the appropriate downloader and request objects.
        """

        client_map = {
            AssetClass.US_EQUITY: self.client.clients['stocks'],
            AssetClass.CRYPTO: self.client.clients['crypto']}

        client = client_map[asset_class]


        downloader_request_map = {
            DatasetType.BAR: {
                AssetClass.US_EQUITY: (client.get_stock_bars, StockBarsRequest),
                AssetClass.CRYPTO: (client.get_crypto_bars, CryptoBarsRequest)},

            DatasetType.QUOTE: {
                AssetClass.US_EQUITY: (client.get_stock_quotes, StockQuotesRequest),
                AssetClass.CRYPTO: (client.get_crypto_quotes, CryptoQuotesRequest)},

            DatasetType.TRADE: {
                AssetClass.US_EQUITY: (client.get_stock_trades, StockTradesRequest),
                AssetClass.CRYPTO: (client.get_crypto_trades, CryptoTradesRequest)}}


        downloader, request = downloader_request_map[dataset_type][asset_class]
                
        return downloader, request


    def download_raw_dataset(
        self,
        dataset_type: DatasetType,
        symbols: List[str],
        asset_class: AssetClass,
        resolution: str,
        start: datetime,
        end: datetime,
        ) -> None:

        """
        Downloads raw dataset from the Alpaca API.

        Args:
            dataset_type (DatasetType): The type of dataset to download (bar, quote, or trade).
            symbols (List[str]): A list of symbols to download.
            asset_class (AssetClass): The asset class to download.
            resolution (str): The resolution of the dataset to download (e.g., "1Min").
            start (datetime): The start date and time of the dataset to download.
            end (datetime): The end date and time of the dataset to download.

        Returns:
            pd.DataFrame: The downloaded dataset as a pandas DataFrame.
        """                

        resolution = to_timeframe(resolution)

        data_fetcher = DataFetcher(self.client)

        downloader, request = data_fetcher.get_downloader_and_request(
            dataset_type=dataset_type, 
            asset_class=asset_class)

        data = downloader(request(
            symbol_or_symbols=symbols,
            timeframe=resolution, 
            start=start, 
            end=end))
        
        try:
            data_df = data.df

        except KeyError:
            raise KeyError(f'No data in requested range {start}-{end}')
        
        return data.df
        
    def download_features_to_hdf5(
        self,
        file_path: str | os.PathLike,
        target_dataset_name: str,
        dataset_type: DatasetType,
        symbols: List[str],
        resolution: str,
        start_date: str | datetime,
        end_date: str | datetime
        ) -> DatasetMetadata:

        """
        Downloads financial features data for the given symbols and saves it in an HDF5 file format.
        
        Args:
            file_path (str | os.PathLike): The file path of the HDF5 file to save the data.
            target_dataset_name (str): The name of the dataset to create in the HDF5 file.
            dataset_type (DatasetType): The type of dataset to download. Either 'BAR', 'TRADE', or 'QUOTE'.
            symbols (List[str]): The list of symbol names to download features data for.
            resolution (str): The frequency at which to sample the data. One of '1Min', '5Min', '15Min', or '30Min'.
            start_date (str | datetime): The start date to download data for, inclusive. If a string, it should be in
                the format 'YYYY-MM-DD'.
            end_date (str | datetime): The end date to download data for, inclusive. If a string, it should be in
                the format 'YYYY-MM-DD'.
                
        Returns:
            metadata (DatasetMetadata): The metadata of the saved dataset.
        """

        validate_path(file_path=file_path)

        asset_class = self._validate_symbols(symbols)
        self._validate_resolution(resolution=resolution)


        calendar = Calendar(asset_class= asset_class)
        schedule = calendar.get_schedule(start_date=start_date, end_date=end_date)

        if len(schedule) == 0:
            raise ValueError(
            'No market hours in date range provided.')


        logger.info(
            f"Downloading dataset for {len(symbols)} symbols | resolution: {resolution} |"
            f" {len(schedule)} market days from {start_date} to {end_date}")

        # shows dataset download progress bar
        progress_bar_ = progress_bar(len(schedule))

        # fetches and saves data on a daily basis
        for market_open, market_close in schedule.values:
            
            raw_dataset = self.download_raw_dataset(
                dataset_type=dataset_type, 
                symbols=symbols, 
                asset_class=asset_class,
                resolution= resolution, 
                start=market_open, 
                end=market_close)


            # check for missing symbols
            dataset_symbols = raw_dataset.index.get_level_values('symbol').unique().tolist()
            missing_symbols = set(dataset_symbols) ^ set(symbols)

            if missing_symbols:
                raise ValueError(
                f'No data for symbols {missing_symbols} in {market_open}, {market_close} time range.')


            # reordering rows to symbols. API does not maintain symbol order.
            raw_dataset = raw_dataset.reindex(
                index=pd.MultiIndex.from_product([
                symbols, raw_dataset.index.levels[1]]))


            # resets multilevel symbol index
            raw_dataset = raw_dataset.reset_index(level=0, names='symbol')      



            processed_groups = list()
            # raw data is processed symbol by symbol
            for symbol, group in raw_dataset.groupby('symbol'):
        
                processed_group = DataProcessor.reindex_and_forward_fill(
                    data=group, open=market_open, 
                    close=market_close, resolution=resolution)
                
                processed_groups.append(processed_group)

            
            features_df = pd.concat(processed_groups, axis= 1)
            features_df = features_df.select_dtypes(include=np.number)

            column_schema = create_column_schema(data=features_df)
            
            features_np = features_df.to_numpy(dtype = np.float32)
            n_rows, n_columns = features_np.shape

            metadata = DatasetMetadata(
                dataset_type= [dataset_type],
                column_schema=column_schema,
                asset_class=asset_class,
                symbols=symbols,
                start= market_open,
                end= market_close,
                resolution = resolution,
                n_rows= n_rows,
                n_columns= n_columns,
            )

            DatasetIO.write_to_hdf5(
                file_path = file_path, 
                data_to_write=features_np, 
                metadata= metadata, 
                target_dataset_name= target_dataset_name)
            
            progress_bar_.set_description(
                f"Density: {DataProcessor.running_dataset_density:.0%}")
            
            progress_bar_.update(1)

        progress_bar_.close()

        return None
    

class DataProcessor:

    running_dataset_density = 0
    
    def resample(
        dataset_type, 
        start: datetime, 
        end: datetime, 
        resolution: str
        ) -> None:

        pass

    def reindex_and_forward_fill(
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

        # resamples and forward fills missing rows in [open, close) range
        index = pd.date_range(
            start=open, end=close, freq=resolution, inclusive='left')

        # creates rows for missing intervals
        processed = data.reindex(index)
        
        # compute fullness of reindexed dataset
        # drop symbols or move date range if density is low
        non_nan_count = processed.notna().sum().sum()
        total_count = processed.size
        density = non_nan_count/total_count

        DataProcessor.running_dataset_density = (
            DataProcessor.running_dataset_density + density
            ) / 2 if DataProcessor.running_dataset_density else density

        # backward fills if first row is nan
        if processed.isna().any().any():
            processed = processed.ffill()

        # backward fills if first row is nan
        if processed.isna().any().any():
            processed = processed.bfill()
        
        return processed

class DatasetIO:

    def write_to_hdf5(
        file_path: str | os.PathLike, 
        data_to_write: np.ndarray, 
        metadata: DatasetMetadata, 
        target_dataset_name: str):

        """Write data to an HDF5 file and update metadata.

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
                    name = target_dataset_name, data=data_to_write,
                    dtype=np.float32, maxshape=(HDF5_DEFAULT_MAX_ROWS, 
                    data_to_write.shape[1]), chunks=True)

                serialized_metadata = pickle.dumps(metadata, protocol=0)
                target_dataset.attrs['metadata'] = serialized_metadata

            else:

                target_dataset_metadata, target_dataset = DatasetIO._extract_target_dataset(
                    hdf5=hdf5, target_dataset_name=target_dataset_name)
                
                # Append the new data to the dataset and update metadata
                new_metadata = target_dataset_metadata + metadata
                target_dataset.resize(
                    (new_metadata.n_rows, new_metadata.n_columns))
                
                target_dataset[
                    target_dataset_metadata.n_rows : new_metadata.n_rows, :] = data_to_write
                serialized_new_metadata = pickle.dumps(new_metadata, protocol=0)
                target_dataset.attrs['metadata'] = serialized_new_metadata
        
        return None

    def _extract_target_dataset(
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

        validate_path(file_path= file_path)
        
        hdf5 =  h5.File(file_path, 'r')


        if target_dataset_name is not None:

            metadata, dataset =  DatasetIO._extract_target_dataset(
                hdf5 = hdf5, target_dataset_name=target_dataset_name)
            
            return metadata, [dataset]
        

        dataset_list = list()
        metadata_list = list()

        for dataset_name in hdf5:

            metadata, dataset = DatasetIO._extract_target_dataset(
                hdf5 = hdf5, target_dataset_name= dataset_name)
            
            dataset_list.append(dataset)
            metadata_list.append(metadata)

        
        joined_metadata = reduce(lambda x, y: x | y, metadata_list)

        return joined_metadata, dataset_list

            
    def merge_datasets(self):

        # merges all datasets into one contiguous dataset
        raise NotImplementedError(
            'method not implemented yet.'
        )


class AbstractStaticDataFeeder(ABC):

    """
    Abstract base class for defining a static data feeder that can reset and split the data.
    """

    @abstractmethod
    def reset(self):

        """
        Resets the internal state of the data feeder.
        
        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        raise NotImplementedError
    
    @abstractmethod
    def split(self):

        """
        Returns the split indices of the data for training and validation.
        
        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """

        raise NotImplementedError

    # @abstractmethod
    # def __delete__(self):
    #     pass
    

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
        end_index : Optional[int] = None,
        n_chunks : Optional[int] = 1) -> None:

        """
        Initializes a StaticDataFeeder object.

        Args:
            dataset_metadata (DatasetMetadata): Metadata for the dataset being loaded.
            datasets (List[h5.Dataset | np.ndarray]): The actual dataset(s) being loaded.
            start_index (int): The starting index to load the data from. Defaults to 0.
            end_index (Optional[int]): The ending index to load the data from. If not 
                provided, defaults to the number of rows specified in the metadata object.
            n_chunks (Optional[int]): The number of chunks to break the dataset into for 
                efficient loading. Defaults to 1.
        """

        self.dataset_metadata = dataset_metadata
        self.datasets = datasets
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else self.dataset_metadata.n_rows
        self.n_rows = self.end_index - self.start_index
        self.n_columns = self.dataset_metadata.n_columns
        self.n_chunks = n_chunks

        return None

    def reset(self):

        """
        Resets the internal state of the data feeder.

        Yields:
            numpy.ndarray: The data row by row from the specified dataset chunk(s).
        """        

        chunk_edge_indices =  np.linspace(
            start = self.start_index, 
            stop = self.end_index,
            num = self.n_chunks+1, 
            dtype= int, 
            endpoint=True)


        for start, end in zip(chunk_edge_indices[:-1], chunk_edge_indices[1:]):

            joined_chunks_in_memory = np.hstack([dataset[start:end, :] 
                for dataset in self.datasets])
            
            for row in joined_chunks_in_memory:
                yield row

    
    def split(self, n: int):
        
        """
        Splits the dataset into multiple non-overlapping contiguous sub-feeders that span the dataset.

        Args:
            n (int): The number of sub-feeders to split the dataset into.

        Returns:
            List[StaticDataFeeder]: A list of StaticDataFeeder objects.
        """

        assert n > 0, "n must be a positive integer"
        static_data_feeders = list()


        edge_indices = np.linspace(
            start=self.start_index,
            stop=self.end_index,
            num=self.n+1,
            dtype=int,
            endpoint=True)


        for start, end in zip(edge_indices[:-1], edge_indices[1:]):

            static_data_feeder = StaticDataFeeder(
                dataset_metadata=self.dataset_metadata,
                datasets=self.datasets,
                start_index=start,
                end_index=end,
                n_chunks=self.n_chunks)
            
            static_data_feeders.append(static_data_feeder)

        return static_data_feeders

    # def __del__(self):
    #     for dataset in self.datasets:
    #         if dataset:
    #             dataset.file.close()
