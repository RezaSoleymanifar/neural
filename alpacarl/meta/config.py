import os 
from alpacarl.meta import log

ALPACA_API_BASE_URL = 'https://api.alpaca.markets'
ALPACA_API_PAPER_URL = 'https://paper-api.alpaca.markets'

try:
    ALPACA_API_KEY = os.environ['ALPACA_API_KEY']
    ALPACA_API_SECRET = os.environ['ALPACA_API_SECRET']
except:
    log.logger.warning('Alpaca API secret and key not set up in OS environment.')