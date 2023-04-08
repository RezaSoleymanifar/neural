from datetime import datetime
from typing import (List, Optional, Tuple, Any)
from functools import reduce
import os

from alpaca.trading.enums import AssetClass, AssetStatus
import pandas as pd
import numpy as np
import pickle, h5py, json

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
from neural.tools.ops import (progress_bar, Calendar,
    to_timeframe, create_column_schema, validate_path)
from neural.tools.enums import CalendarType
from neural.common.constants HDF5_DEFAULT_MAX_ROWS

    
class DataFetcher():
    def __init__(
        self,
        client: AlpacaMetaClient
        ) -> None:

        self.client = client

        return None

    def _validate_resolution(self, resolution):
        accepted_resolutions = {'1Min', '5Min', '15Min', '30Min'}

        if resolution not in accepted_resolutions:
            raise ValueError(f'Accepted resolutions: {accepted_resolutions}.')
        
        
    def _validate_symbols(self, symbols: List[str]):

        if len(symbols) == 0:
            raise ValueError('symbols argument cannot be an empty sequence.')


        duplicate_symbols = list(set([
            symbol for symbol in symbols if symbols.count(symbol) > 1]))
        
        if duplicate_symbols:
            raise ValueError(
                f'Symbols {duplicate_symbols} have duplicate values.')



        for symbol in symbols:

            if symbol not in self.client._AlpacaMetaClient__symbols:
                raise ValueError(
                    f'Symbol {symbol} is not a known symbol.')

            if not self.client._AlpacaMetaClient__symbols[symbol].tradable:
                raise ValueError(
                    f'Symbol {symbol} is not a tradable symbol.')
        
            if not self.client._AlpacaMetaClient__symbols[
                symbol].status == AssetStatus.ACTIVE:
                raise ValueError(
                    f'Symbol {symbol} is not an active symbol.')

        asset_classes = set(self.client._AlpacaMetaClient__symbols[
            symbol].asset_class for symbol in symbols)


        # checks if symbols have the same asset class
        if len(asset_classes) != 1:
            raise ValueError(
                'Symbols are not of the same asset class.')
        asset_class = asset_classes.pop()

        return asset_class

    def get_downloader_and_request(
        self, 
        dataset_type: DatasetType, 
        asset_class = AssetClass
        ) -> Tuple[Any, Any]:
        

        if asset_class == AssetClass.US_EQUITY:
            client = self.client.clients['stocks']
            
        elif asset_class == AssetClass.CRYPTO:
            client = self.client.clients['crypto']



        if dataset_type == DatasetType.BAR:

            if asset_class == AssetClass.US_EQUITY:

                downloader = client.get_stock_bars
                request = StockBarsRequest
                
            elif asset_class == AssetClass.CRYPTO:

                downloader = client.get_crypto_bars
                request = CryptoBarsRequest



        elif dataset_type == DatasetType.QUOTE:

            if asset_class == AssetClass.US_EQUITY:
                downloader = client.get_stock_quotes
                request = StockQuotesRequest
            
            elif asset_class == AssetClass.CRYPTO:
                downloader = client.get_crypto_quotes
                request = CryptoQuotesRequest



        elif dataset_type == DatasetType.TRADE:

            if asset_class == AssetClass.US_EQUITY:
                downloader = client.get_stock_trades
                request = StockTradesRequest
            
            elif asset_class == AssetClass.CRYPTO:
                downloader = client.get_crypto_trades
                request = CryptoTradesRequest
                
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


        validate_path(file_path=file_path)

        asset_class = self._validate_symbols(symbols)
        self._validate_resolution(resolution=resolution)


        # getting market schedule
        if asset_class == AssetClass.US_EQUITY:
            calendar = Calendar(calendar_type= CalendarType.NYSE)
        
        elif asset_class == AssetClass.CRYPTO:
            calendar = Calendar(calendar_type = CalendarType.ALWAYS_OPEN)

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

            if len(missing_symbols) != 0:
                raise ValueError(
                f'No data for symbols {missing_symbols} in {market_open}, {market_close} time range.')


            # reordering rows to symbols. API does not maintain symbol order.
            raw_dataset = raw_dataset.reindex(
                index=pd.MultiIndex.from_product([
                symbols, raw_dataset.index.levels[1]]))


            # resets multilevel symbol index
            raw_dataset = raw_dataset.reset_index(level=0, names='symbol')      


            # add resample method here for trades and quotes

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

        validate_path(file_path=file_path)
        
        with h5py.File(file_path, 'a') as hdf5:

            if target_dataset_name not in hdf5:

                # Create a fixed-size dataset with a predefined data type and dimensions
                target_dataset = hdf5.create_dataset(
                    name = target_dataset_name, shape=data_to_write.shape,
                    dtype=np.float32, chunks = True)
                
                serialized_metadata = pickle.dumps(metadata, protocol=0)
                target_dataset.attrs['metadata'] = serialized_metadata

            else:

                target_dataset_metadata, target_dataset = DatasetIO._extract_target_dataset(
                    hdf5=hdf5, maxshape=(HDF5_DEFAULT_MAX_ROWS, metadata.n_columns), target_dataset_name=target_dataset_name)
                                
                new_metadata = target_dataset_metadata + metadata

                target_dataset.resize(
                    (new_metadata.n_rows, new_metadata.n_columns))

                # Append the new data to the dataset and update metadata
                target_dataset[metadata.n_rows:new_metadata.n_rows, :] = data_to_write
                target_dataset.attrs['metadata'] = new_metadata
        
        return None

    def _extract_target_dataset(
            hdf5: h5py.File,
            target_dataset_name: str
            ) -> Tuple[DatasetMetadata, h5py.Dataset]:

        target_dataset = hdf5[target_dataset_name]
        serialized_metadata = target_dataset.attrs['metadata']
        metadata = pickle.loads(serialized_metadata.encode())

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
        ) -> Tuple[DatasetMetadata, List[h5py.Dataset]]:

        validate_path(file_path= file_path)
        
        with h5py.File(file_path, 'r') as hdf5:

            if target_dataset_name is None:

                dataset_list = list()
                metadata_list = list()

                for dataset_name in hdf5:

                    metadata, dataset = DatasetIO._extract_target_dataset(
                        hdf5 = hdf5, target_dataset_name= dataset_name)
                    
                    dataset_list.append(dataset)
                    metadata_list.append(metadata)

                
                joined_metadata = reduce(lambda x, y: x | y, metadata_list)

                return joined_metadata, dataset_list
            
            else:

                metadata, dataset =  DatasetIO._extract_target_dataset(
                    hdf5 = hdf5, target_dataset_name=target_dataset_name)
                
                return metadata, [dataset]


class RowGenerator:
    # to iteratively return info required for environments from a dataset.
    def __init__(
        self, 
        dataset_metadata: DatasetMetadata, 
        datasets: List[h5py.Dataset],
        start_index: int = 0,
        end_index : Optional[int] = None,
        n_chunks : Optional[int] = 1) -> None:

        self.dataset_metadata = dataset_metadata
        self.datasets = datasets
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else self.dataset_metadata.n_rows
        self.n_rows = self.end_index - self.start_index
        self.n_columns = self.dataset_metadata.n_columns
        self.n_chunks = n_chunks

    # end_index = n_rows thus it's a dummy index.
    def __iter__(self):
        
        start = 0

        for end in np.linspace(
            self.start_index, 
            self.end_index,
            self.n_chunks, 
            dtype= int, 
            endpoint=True):
            
            rows_in_memory = np.hstack([dataset[start:end, :] 
                for dataset in self.datasets])
            
            for row in rows_in_memory:
                yield row

            start = end

    def reset(self):

        return  RowGenerator(
            dataset_metadata=self.dataset_metadata,
            datasets=self.datasets,
            start_index=self.start_index,
            end_index=self.end_index, # exclusive
            n_chunks = self.n_chunks)
    
    def reproduce(self, n: int):

        assert n > 0, "n must be a positive integer"
        generators = list()

        start_index = 0

        for end_index in np.linspace(
            self.start_index, 
            self.end_index, 
            self.n_chunks, 
            dtype=int, 
            endpoint=True):

            generator = RowGenerator(
                dataset_metadata=self.dataset_metadata,
                datasets=self.datasets,
                start_index=start_index,
                end_index=end_index,
                n_chunks=self.n_chunks)
            
            generators.append(generator)

            start_index = end_index

        return generators
    