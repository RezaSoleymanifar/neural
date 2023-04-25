import os
import numpy as np


# Load the API key and secret from environment variables
API_KEY = os.environ.get('API_KEY', None)
API_SECRET = os.environ.get('API_SECRET', None)


# Set the default maximum number of rows for HDF5 storage
HDF5_DEFAULT_MAX_ROWS = 2_000_000


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


# dictionary observations are expected to be dict of numpy arrays.
ACCEPTED_OBSERVATION_TYPES = {np.ndarray, dict}

# dictionary actions are expected to dict of be numpy arrays.
ACCEPTED_ACTION_TYPES = {np.ndarray, dict}

# sets global numerical precision
GLOBAL_DATA_TYPE = np.float32