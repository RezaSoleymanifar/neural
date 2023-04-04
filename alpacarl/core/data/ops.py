from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import AssetClass
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpacarl.core.client import AlpacaMetaClient

from alpacarl.meta import log
from alpacarl.aux.tools import progress_bar

import os, re
from datetime import datetime
from typing import (List, Optional, Tuple)
from functools import reduce

import pandas as pd
import numpy as np
import pickle, h5py

    
class DatasetDownloader():
    def __init__(
        self,
        meta_client: AlpacaMetaClient
        ) -> None:

        self.meta_client = meta_client

        return None

    def validate_symbols(self, symbols: List[str]):

        duplicate_symbols = list(set([x for x in symbols if symbols.count(x) > 1]))
        if duplicate_symbols is not None:
            raise ValueError(f'Symbols {duplicate_symbols} have duplicate values.')
        

        # checks if symbols name is valid
        for symbol in symbols:
            if symbol not in self.symbols:
                raise ValueError(f'Symbol {symbol} is not a supported symbol.')

        asset_classes = set(
            self.symbols[symbol]['asset_class'] for symbol in symbols)

        # checks if symbols have the same asset class
        if len(asset_classes) != 1:
            raise ValueError('Symbols are not of the same asset class.')

        asset_class = asset_classes.pop()

        return asset_class
    
    def download_and_write_dataset(self,
        path: str,
        target_dataset_name: str,
        dataset_type: DatasetType,
        symbols: List[str],
        resolution: str,
        start_date: str,
        end_date: str,
        ) -> None:
        
        if not os.path.exists(path):
            raise ValueError(f'Path {path} does not exist.')
        
        
        # converts to expected input formats
        start_date, end_date = to_datetime(start_date), to_datetime(end_date)
        if start_date = datetime.today

        resolution = to_timeframe(resolution)
        # API produces results in sorted order of symbols
        symbols = sorted(symbols)
        
        asset_class = self.validate_symbols(symbols)

        downloader, request = DatasetDownloader(
            meta_client = self, dataset_type = dataset_type, asset_class = asset_class)

        if asset_class == AssetClass.US_EQUITY:
            calendar = Calendar(calendar_type= Calendar.NYSE)
        
        elif asset_class == AssetClass.CRYPTO:
            calendar = Calendar(calendar_type = Calendar.ALWAYS_OPEN)

        schedule = calendar.get_schedule(start_date=start_date, end_date=end_date)
        time_zone = calendar.get_time_zone()

        # shows dataset download progress bar
        progress_bar_ = progress_bar(total=len(schedule))

        log.logger.info(
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


            features_df = pd.concat(
                [DataProcessor.resample_and_ffil(group[1], day, resolution
                ) for group in bars.groupby('symbol')], axis=1)

            features_df = features_df.select_dtypes(include=np.number)
            features_np = features_df.to_numpy(dtype = np.float32)
            n_rows, n_columns = features_np.shape


            column_schema = DatasetMetadata.create_column_schema(
                dataset_type, features_df)
            
            metadata = DatasetMetadata(
                dataset_type= dataset_type,
                column_schema = column_schema,
                asset_class=asset_class,
                symbols=symbols,
                start= start_date,
                end= end_date,
                resolution = resolution,
                n_rows= n_rows,
                n_columns= n_columns,
            )

            DatasetIO.write_to_hdf5(
                path = path, 
                data_to_write=features_np, 
                metadata= metadata, 
                target_dataset_name= target_dataset_name)

            progress_bar_.update(1)
        progress_bar_.close()

        return None
    
    def get_downloader_and_request(
        self, 
        dataset_type: DatasetType, 
        asset_class = AssetClass):
        
        if dataset_type == DatasetType.BAR:
            # choose relevant client
            if asset_class == AssetClass.US_EQUITY:
                client = self.meta_client.clients['stocks']
                downloader = client.get_stocks_bars
                request = StockBarsRequest
                
            elif asset_class == AssetClass.CRYPTO:
                client = self.meta_client.clients['crypto']
                downloader = client.get_crypto_bars
                request = CryptoBarsRequest
                
            return downloader, request
    

class DataProcessor:
    def __init__(self) -> None:
        pass 

    def resample_and_ffil(data, open: datetime, close: datetime, interval: str):
        # resamples and forward fills missing rows in [open, close] range

        index = pd.date_range(start=open, end=close, freq=interval)

        # creates rows for missing intervals
        resampled = data.reindex(index, method='ffill')
        if resampled.isna().all().all():
            log.logger.exception(
                'Data does not have entries in NYSE market hours.')
            raise ValueError

        # backward fills if first row is nan
        if resampled.isna().any().any():
            resampled = resampled.bfill()

        # Prefix column names with symbol
        symbol = resampled['symbol'][0]
        resampled.columns = [f'{symbol}_{col}' for col in data.columns]
        return resampled

class DatasetIO:

    def write_to_hdf5(
        path: str | os.PathLike, 
        data_to_write: np.ndarray, 
        metadata: DatasetMetadata, 
        target_dataset_name: str):

        if os.path.exists(path):

            with h5py.File(path, 'w') as hdf5:

                if target_dataset_name not in hdf5:
                    # Create a fixed-size dataset with a predefined data type and dimensions
                    target_dataset = hdf5.create_dataset(
                        name = target_dataset_name, shape=data_to_write.shape, dtype=np.float32, chunks = True)
                    
                    serialized_metadata = pickle.dumps(metadata)
                    target_dataset.attrs['metadata'] = serialized_metadata

                else:

                    target_dataset_metadata, target_dataset = DatasetIO.load_from_hdf5(
                        hdf5, target_dataset_name= target_dataset_name)

                    new_metadata = target_dataset_metadata + metadata

                    target_dataset.resize((new_metadata.n_rows, new_metadata.n_columns))

                    # Append the new data to the dataset and update metadata
                    target_dataset[metadata.n_rows:new_metadata.n_rows, :] = data_to_write
                    target_dataset.attrs['metadata'] = new_metadata

        else:

            raise ValueError(f'Path {path} does not exist.')
        
        return None

    def extract_dataset(
            hdf5: h5py.File,
            target_dataset_name: str
            ) -> Tuple[DatasetMetadata, h5py.Dataset]:

        target_dataset = hdf5[target_dataset_name]
        serialized_metadata = target_dataset.attrs['metadata']
        metadata = pickle.loads(serialized_metadata)

        return metadata, target_dataset            

    def load_from_hdf5(
        path: str | os.PathLike, 
        target_dataset_name: Optional[str] = None
        ) -> Tuple[DatasetMetadata, List[h5py.Dataset]]:

        if os.path.exists(path):

            with h5py.File(path, 'r') as hdf5:

                if target_dataset_name is None:
                    dataset_list = list()
                    metadata_list = list()
                    for dataset_name in hdf5:

                        metadata, dataset = DatasetIO.extract_dataset(
                            hdf5 = hdf5, target_dataset_name= dataset_name)
                        dataset_list.append(dataset)
                        metadata_list.append(metadata)
                        joined_metadata = reduce(lambda x, y: x | y, metadata_list)

                    return joined_metadata, dataset_list
                
                else:
                    metadata, dataset =  DatasetIO.extract_dataset(
                        hdf5 = hdf5, target_dataset_name=target_dataset_name)
                    return metadata, [dataset]
        else:
            raise ValueError(f'Path {path} does not exist.')

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
        for end in np.linspace(self.start_index, self.end_index, self.n_chunks, dtype= int, endpoint=True):
            rows_in_memory = np.hstack([dataset[start:end, :] for dataset in self.datasets])
            for row in rows_in_memory:
                yield row
            start = end

    def reset(self):
        return  RowGenerator(
            dataset_metadata=self.dataset_metadata,
            datasets=self.datasets,
            start_index=self.start_index,
            end_index=self.end_index, # exclusive
            n_rows_per_read=self.n_rows_per_read)
    
    def reproduce(self, n: int):

        assert n > 0, "n must be a positive integer"
        generators = list()

        start_index = 0

        for end_index in np.linspace(self.start_index, self.end_index, self.n_chunks, dtype=int, endpoint=True):
            generator = RowGenerator(
                dataset_metadata=self.dataset_metadata,
                datasets=self.datasets,
                start_index=start_index,
                end_index=end_index,
                n_chunks=self.n_chunks)
            
            generators.append(generator)
            start_index = end_index
        return generators
    
# def to_datetime(date: str):
#     try:
#         date_format = "%Y-%m-%d"
#         dt = datetime.strptime(date, date_format)
#     except:
#         ValueError('Invalid date. Valid examples: 2022-03-20, 2015-01-01')
#     return dt

def to_timeframe(time_frame: str):

    match = re.search(r'(\d+)(\w+)', time_frame)

    if match:

        amount = int(match.group(1))
        unit = match.group(2)

        map = {
            'Min': TimeFrameUnit.Minute,
            'Hour': TimeFrameUnit.Hour,
            'Day': TimeFrameUnit.Day,
            'Week': TimeFrameUnit.Week,
            'Month': TimeFrameUnit.Month}

        return TimeFrame(amount, map[unit])
    else:
        raise ValueError(
            "Invalid timeframe. Valid examples: 59Min, 23Hour, 1Day, 1Week, 12Month")