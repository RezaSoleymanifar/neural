import os

# Load the API key and secret from environment variables
API_KEY = os.environ.get('API_KEY', None)
API_SECRET = os.environ.get('API_SECRET', None)

# Set the default maximum number of rows for HDF5 storage
HDF5_DEFAULT_MAX_ROWS = 2_000_000


# Set the minimum net worth for a pattern day trader
PATTERN_DAY_TRADER_MINIMUM_NET_WORTH = 25_000

DOW_JONES_SYMBOLS = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX',
    'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
    'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
                     'PFE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WMT']


CRYPTO_SYMBOLS = ['ADA', 'BCH', 'BTC', 'DOGE',
                  'DOT', 'ETH', 'LINK', 'LTC', 'UNI', 'XRP']


CUSTOM_SYMBOLS = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN',
    'BA', 'BAC', 'BLK', 'BMY', 'CMCSA', 'COST', 'CRM', 'CSCO',
    'CVX', 'DIS', 'FB', 'FIS', 'GOOGL', 'HD', 'HON', 'IBM', 'INTC',
    'JNJ', 'JPM', 'KO', 'LOW', 'MA', 'MAA', 'MDT', 'MRK', 'MSFT',
    'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PYPL',
    'SBUX', 'T', 'TSLA', 'UNH', 'UPS', 'V', 'VZ', 'WMT', 'XOM']

