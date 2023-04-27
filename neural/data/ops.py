from datetime import datetime
from typing import (List, Optional, Tuple, Any, Iterable)
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
from neural.data.enums import DatasetType, DatasetMetadata
from neural.client.alpaca import AlpacaClient
from neural.tools.base import (progress_bar, to_timeframe, 
    create_column_schema, validate_path)
from neural.tools.misc import Calendar
from neural.common.constants import HDF5_DEFAULT_MAX_ROWS
    


