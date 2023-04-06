from datetime import datetime
from typing import (List, Optional, Tuple)
from functools import reduce
import os

from alpaca.trading.enums import AssetClass, AssetStatus
import pandas as pd
import numpy as np
import pickle, h5py

from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical.stock import StockBarsRequest
from alpaca.data.historical.crypto import CryptoBarsRequest

from alpacarl.common import logger
from alpacarl.common.exceptions import CorruptDataError
from alpacarl.core.data.enums import DatasetType, DatasetMetadata
from alpacarl.connect.client import AlpacaMetaClient
from alpacarl.tools.ops import (progress_bar, Calendar,
    to_timeframe, create_column_schema, validate_path)

    
class DatasetDownloader():
    def __init__(
        self,
        client: AlpacaMetaClient
        ) -> None:

        self.client = client

        return None

    def _validate_symbols(self, symbols: List[str]):

        if len(symbols) == 0:
            raise ValueError('symbols argument cannot be an empty sequence.')
        
        duplicate_symbols = list(set([
            symbol for symbol in symbols if symbols.count(symbol) > 1]))
        
        if duplicate_symbols:
            raise ValueError(
                f'Symbols {duplicate_symbols} have duplicate values.')
        
        # checks if symbols name is valid
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
        asset_class = AssetClass):
        
        if dataset_type == DatasetType.BAR:
            # choose relevant client
            if asset_class == AssetClass.US_EQUITY:

                client = self.client.clients['stocks']
                downloader = client.get_stocks_bars
                request = StockBarsRequest
                
            elif asset_class == AssetClass.CRYPTO:

                client = self.client.clients['crypto']
                downloader = client.get_crypto_bars
                request = CryptoBarsRequest
                
            return downloader, request
        
    def download_dataset_to_hdf5(
        self,
        file_path: str | os.PathLike,
        target_dataset_name: str,
        dataset_type: DatasetType,
        symbols: List[str],
        resolution: str,
        start_date: datetime.date,
        end_date: datetime.date,
        ) -> None:

        validate_path(file_path=file_path)
        
        # converts to expected input formats
        if end_date == datetime.today():
            raise ValueError(
            'Today\'s data is only available through streaming.')
        

        if not os.path.exists(file_path):
            raise ValueError(f'Path {file_path} does not exist.')
        
    
        resolution = to_timeframe(resolution)

        # API produces results in sorted order of symbols
        symbols = sorted(symbols)

        asset_class = self._validate_symbols(symbols)

        downloader, request = DatasetDownloader(
            client = self, dataset_type = dataset_type, asset_class = asset_class)

        if asset_class == AssetClass.US_EQUITY:
            calendar = Calendar(calendar_type= Calendar.NYSE)
        
        elif asset_class == AssetClass.CRYPTO:
            calendar = Calendar(calendar_type = Calendar.ALWAYS_OPEN)

        schedule = calendar.get_schedule(start_date=start_date, end_date=end_date)
        time_zone = calendar.get_time_zone()

        # shows dataset download progress bar
        progress_bar_ = progress_bar(total=len(schedule))

        logger.info(
            f"Downloading dataset for {len(self.symbols)} symbols | resolution: {resolution} |"
            f" {len(schedule)} working days from {start_date} to {end_date}"
        )

        # cache and download data day by day
        for _, day in schedule:
            
            market_open = day['market_open']
            market_close = day['market_close']


            bars = downloader(
                request(symbol_or_symbols=symbols, timeframe=resolution,
                start=market_open, end=market_close))

            bars = bars.tz_convert(time_zone)
            
            if bars['symbol'].nunique() != symbols:
                pass


            features_df = pd.concat(
                [DataProcessor.resample_and_forward_fill(data = group[1],
                open = market_open, close = market_close, resolution = resolution)
                for group in bars.groupby('symbol')],
                axis=1)


            symbols_in_df = features_df['symbols'].tolist()
            missing_symbols = set(symbols_in_df) ^ set(symbols)

            if len(missing_symbols) != 0:
                raise ValueError(
                f'No data for symbols {missing_symbols} on {start_date}.')

            features_df = features_df.select_dtypes(include=np.number)
            column_schema = create_column_schema(
                data=features_df,
                dataset_type=dataset_type)
            
            features_np = features_df.to_numpy(dtype = np.float32)
            n_rows, n_columns = features_np.shape

            
            metadata = DatasetMetadata(
                dataset_type= [dataset_type],
                asset_class=asset_class,
                column_schema = column_schema,
                symbols=symbols,
                start= start_date,
                end= end_date,
                resolution = resolution,
                n_rows= n_rows,
                n_columns= n_columns,
            )

            DatasetIO.write_to_hdf5(
                file_path = file_path, 
                data_to_write=features_np, 
                metadata= metadata, 
                target_dataset_name= target_dataset_name)

            progress_bar_.update(1)
        progress_bar_.close()

        return None
    

class DataProcessor:
    def __init__(self) -> None:
        pass 

    def resample_and_forward_fill(
            data, 
            open: datetime, 
            close: datetime, 
            resolution: str):
        
        # resamples and forward fills missing rows in [open, close] range
        index = pd.date_range(
            start=open, end=close, freq=resolution)

        # creates rows for missing intervals
        resampled = data.reindex(index, method='ffill')

        if resampled.isna().all().all():

            logger.exception(
                'Data does not have entries in market hours.')
            raise ValueError

        # backward fills if first row is nan
        if resampled.isna().any().any():
            resampled = resampled.bfill()

        # Prefix column names with symbol
        symbol = resampled['symbol'][0]

        resampled.columns = [
            f'{symbol}_{col}' for col in data.columns]
        
        return resampled

class DatasetIO:

    def write_to_hdf5(
        file_path: str | os.PathLike, 
        data_to_write: np.ndarray, 
        metadata: DatasetMetadata, 
        target_dataset_name: str):

        validate_path(file_path=file_path)

        if not os.path.exists(file_path):
            raise ValueError(f'Path {file_path} does not exist.')
        
        with h5py.File(file_path, 'w') as hdf5:

            if target_dataset_name not in hdf5:
                # Create a fixed-size dataset with a predefined data type and dimensions
                target_dataset = hdf5.create_dataset(
                    name = target_dataset_name, shape=data_to_write.shape,
                    dtype=np.float32, chunks = True)
                
                serialized_metadata = pickle.dumps(metadata)
                target_dataset.attrs['metadata'] = serialized_metadata

            else:

                target_dataset_metadata, target_dataset = DatasetIO.load_from_hdf5(
                    hdf5, target_dataset_name= target_dataset_name)

                new_metadata = target_dataset_metadata + metadata

                target_dataset.resize(
                    (new_metadata.n_rows, new_metadata.n_columns))

                # Append the new data to the dataset and update metadata
                target_dataset[metadata.n_rows:new_metadata.n_rows, :] = data_to_write
                target_dataset.attrs['metadata'] = new_metadata
        
        return None

    def _extract_dataset(
            hdf5: h5py.File,
            target_dataset_name: str
            ) -> Tuple[DatasetMetadata, h5py.Dataset]:

        target_dataset = hdf5[target_dataset_name]
        serialized_metadata = target_dataset.attrs['metadata']
        metadata = pickle.loads(serialized_metadata)

        if metadata.n_rows != len(target_dataset):
            raise CorruptDataError(
                f'Rows in {target_dataset_name}: {len(target_dataset)}.' 
                f'Rows in metadata: {metadata.n_rows}')
        
        if metadata.n_columns != target_dataset.shape[1]:
            raise CorruptDataError(
                f'Columns in {target_dataset_name}: {target_dataset.shape[1]}.'
                f'Rows in metadata: {metadata.n_columns}')

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

                    metadata, dataset = DatasetIO._extract_dataset(
                        hdf5 = hdf5, target_dataset_name= dataset_name)
                    
                    dataset_list.append(dataset)
                    metadata_list.append(metadata)

                
                joined_metadata = reduce(lambda x, y: x | y, metadata_list)

                return joined_metadata, dataset_list
            
            else:

                metadata, dataset =  DatasetIO._extract_dataset(
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
    