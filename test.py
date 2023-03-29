from alpaca_trade_api.rest import REST
from alpacarl.meta.config import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_PAPER_URL

interval = '1Min' # "1Min", "5Min", "15Min", "1H"
start = "2018-01-02"  # start time, min="2017-01-01"
end = "2018-01-04"  # End time, max=today
symbol = ['DOW']

api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_PAPER_URL, api_version='v2')

df = api.get_bars(['DOW'], interval, start, end, adjustment='raw', limit=None).df
print(df)