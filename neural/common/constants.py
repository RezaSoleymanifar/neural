"""
constants.py

This module contains constants for the neural package.

Constants:
----------
    API_KEY (str): 
        The default API key for the client API. Tries to load from
        environment variable API_KEY. If not found, then API_KEY is set
        to None.
    API_SECRET (str):
        The default API secret for the client API. Tries to load from
        environment variable API_SECRET. If not found, then API_SECRET
        is set to None.
    CALENDAR (Calendar):
        if not set, the default calendar from neural.utils.time will be
        used. If set, the calendar will be used for all time related
        computations.
    LOG_PATH (str):
        if not set no log files will be created. If set, the log files
        will be created at specified path.
    MAX_LOG_SIZE (int):
        maximum size of log file in bytes. If the size of log file
        exceeds this value, a backup will be created.
    LOG_BACKUP_COUNT (int):
        The number of backup log files to keep. If the log file exceeds
        MAX_LOG_SIZE, then it will be truncated and renamed to
        <LOG_PATH>.1. If <LOG_PATH>.1 exists, then it will be renamed to
        <LOG_PATH>.2, and so on. If LOG_BACKUP_COUNT is 0, then no
        backup log files will be kept.
    LOG_LEVEL (int):
        The logging level for the logger. The logger will log messages
        with a level greater than or equal to LOG_LEVEL. The available
        logging levels are: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
        lower level = more messages, i.e. messages at lower levels are filtered out.
CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
    ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS (set):
        The accepted download resolutions for Alpaca API.
    HDF5_DEFAULT_MAX_ROWS (int):
        The default maximum number of rows for HDF5 storage. Note hdf5
        files occupy a contiguous block of memory and they have size of
        HDF5_DEFAULT_MAX_ROWS even when empty.
    ACCEPTED_OBSERVATION_TYPES (set):
        The accepted observation types for neural environments.
    ACCEPTED_ACTION_TYPES (set):
        The accepted action types for neural environments.
    GLOBAL_DATA_TYPE (type):
        The global numerical precision. This is used for all
        computations that involve numpy arrays. Note that default data
        type of tensors in PyTorch is torch.float32.
    ALPACA_MINIMUM_SHORT_MARRGIN_EQUITY (int):
        The minimum net worth required for short and margin trading in
        Alpaca API.
    PATTERN_DAY_TRADER_MINIMUM_NET_WORTH (int):
        The minimum net worth required for a pattern day trader to
        continue doing day trading.
    DOW_JONES_SYMBOLS (list):
        A list of Dow Jones Industrial Average symbols.
    CRYPTO_SYMBOLS (list):
        A list of crypto symbols with high market capitalization.
    CUSTOM_SYMBOLS (list):
        A custom list of stock symbols.
"""

import logging

import os
import numpy as np

from neural.utils.time import Calendar

# =====================================CONNECTION==============================

API_KEY = os.environ.get('API_KEY', None)
API_SECRET = os.environ.get('API_SECRET', None)

# =====================================CALENDAR================================

CALENDAR = Calendar

# =====================================LOG=====================================

LOG_PATH = None
MAX_LOG_SIZE = 10_000_000  # 10 MB
LOG_BACKUP_COUNT = 10

# Set the logging level for the logger/file/console handler 

LOG_LEVEL = logging.INFO

# =====================================DATA====================================

ALPACA_ACCEPTED_DOWNLOAD_RESOLUTIONS = {'1Min', '5Min', '15Min', '30Min'}
# Set the default maximum number of rows for HDF5 storage. Note hdf5
# files occupy a contiguous block of memory and they have size of
# HDF5_DEFAULT_MAX_ROWS even when empty.
HDF5_DEFAULT_MAX_ROWS = 5_000_000

# dictionary observations will be checked to be dict of numpy arrays.
ACCEPTED_OBSERVATION_TYPES = {np.ndarray, dict}

# dictionary actions will be checked to be dict of numpy arrays.
ACCEPTED_ACTION_TYPES = {np.ndarray, dict}

# sets global numerical precision. This is used for all computations
# that involve numpy arrays. Note that default data type of tensors in
# PyTorch is torch.float32.
GLOBAL_DATA_TYPE = np.float32

#=====================================TRADE====================================

# Set the minimum net worth required for short and margin trading in
# Alpaca API.
ALPACA_MINIMUM_SHORT_MARRGIN_EQUITY = 2_000

# Set the minimum net worth for a pattern day trader
PATTERN_DAY_TRADER_MINIMUM_EQUITY = 25_000

# Dow Jones Industrial Average symbols
DOW_JONES_SYMBOLS = [
    'AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'GS', 'HD',
    'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT'
]

# a list of crypto symbols with high market capitalization
CRYPTO_SYMBOLS = [
    'ADA', 'BCH', 'BTC', 'DOGE', 'DOT', 'ETH', 'LINK', 'LTC', 'UNI', 'XRP'
]

# a custom list of stock symbols
CUSTOM_SYMBOLS = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN', 'BA', 'BAC', 'BLK', 'BMY',
    'CMCSA', 'COST', 'CRM', 'CSCO', 'CVX', 'DIS', 'FB', 'FIS', 'GOOGL', 'HD',
    'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'LOW', 'MA', 'MAA', 'MDT', 'MRK',
    'MSFT', 'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PYPL',
    'SBUX', 'T', 'TSLA', 'UNH', 'UPS', 'V', 'VZ', 'WMT', 'XOM'
]
