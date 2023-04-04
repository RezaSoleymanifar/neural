import os

ALPACA_API_ENDPOINT = 'https://api.alpaca.markets'
ALPACA_API_ENDPOINT_PAPER = 'https://paper-api.alpaca.markets'

ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', None)
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET', None)


DOW_JONES_SYMBOLS = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX',
    'DD', 'DIS', 'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ',
    'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
    'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']


CRYPTO_SYMBOLS = ['ADA', 'BCH', 'BTC', 'DOGE',
                  'DOT', 'ETH', 'LINK', 'LTC', 'UNI', 'XRP']


CUSTOM_SYMBOLS = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMZN',
    'BA', 'BAC', 'BLK', 'BMY', 'CMCSA', 'COST', 'CRM', 'CSCO',
    'CVX', 'DIS', 'FB', 'FIS', 'GOOGL', 'HD', 'HON', 'IBM', 'INTC',
    'JNJ', 'JPM', 'KO', 'LOW', 'MA', 'MAA', 'MDT', 'MRK', 'MSFT',
    'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PYPL',
    'SBUX', 'T', 'TSLA', 'UNH', 'UPS', 'V', 'VZ', 'WMT', 'XOM']
