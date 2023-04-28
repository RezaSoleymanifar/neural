import os
import numpy as np
import logging

from neu


# =====================================CONNECTION=========================================

# Load the API key and secret from environment variables
API_KEY = os.environ.get('API_KEY', None)
API_SECRET = os.environ.get('API_SECRET', None)


# =====================================CALENDAR=========================================

CALENDAR = 
# =====================================LOG=========================================

# if not set no log files will be created. If set, the log files will be created
# at specified path.
LOG_PATH = None
# maximum size of log file in bytes. If the size of log file exceeds this value,
# a backup will be created.
MAX_LOG_SIZE = 10_000_000  # 10 MB
# maximum number of log files to keep. If the number of log files exceeds this
# value, the oldest log file will be deleted.
LOG_BACKUP_COUNT = 10

# Set the logging level for the logger/file/console handler
# lower level = more messages, i.e. messages at lower
# levels are filtered out.
# CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET

LOG_LEVEL = logging.INFO


# =====================================DATA=========================================


ACCEPTED_DOWNLOAD_RESOLUTIONS = {'1Min', '5Min', '15Min', '30Min'}
# Set the default maximum number of rows for HDF5 storage.
# Note hdf5 files occupy a contiguous block of memory and
# they have size of HDF5_DEFAULT_MAX_ROWS even when empty.
HDF5_DEFAULT_MAX_ROWS = 5_000_000


# dictionary observations will be checked to be dict of numpy arrays.
ACCEPTED_OBSERVATION_TYPES = {np.ndarray, dict}


# dictionary actions will be checked to be dict of numpy arrays.
ACCEPTED_ACTION_TYPES = {np.ndarray, dict}


# sets global numerical precision. This is used for all computations
# that involve numpy arrays. Note that default data type of tensors in
# PyTorch is torch.float32.
GLOBAL_DATA_TYPE = np.float32




#=====================================TRADE=========================================

# Set the minimum net worth for a pattern day trader
PATTERN_DAY_TRADER_MINIMUM_NET_WORTH = 25_000


# Dow Jones Industrial Average symbols
DOW_JONES_SYMBOLS = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX',
    'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
    'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
                     'PFE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT']


# a list of crypto symbols with high market capitalization
CRYPTO_SYMBOLS = ['ADA', 'BCH', 'BTC', 'DOGE',
                  'DOT', 'ETH', 'LINK', 'LTC', 'UNI', 'XRP']


# a custom list of stock symbols
CUSTOM_SYMBOLS = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN',
    'BA', 'BAC', 'BLK', 'BMY', 'CMCSA', 'COST', 'CRM', 'CSCO',
    'CVX', 'DIS', 'FB', 'FIS', 'GOOGL', 'HD', 'HON', 'IBM', 'INTC',
    'JNJ', 'JPM', 'KO', 'LOW', 'MA', 'MAA', 'MDT', 'MRK', 'MSFT',
    'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PYPL',
    'SBUX', 'T', 'TSLA', 'UNH', 'UPS', 'V', 'VZ', 'WMT', 'XOM']
