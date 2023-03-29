import os 

ALPACA_API_ENDPOINT = 'https://api.alpaca.markets'
ALPACA_API_ENDPOINT_PAPER = 'https://paper-api.alpaca.markets'

ALPACA_API_KEY = os.environ.get('ALPACA_API_KEY', None)
ALPACA_API_SECRET = os.environ.get('ALPACA_API_SECRET', None)

# DOW_JONES = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DWDP', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'UTX', 'V', 'VZ', 'WMT', 'XOM']
