from alpacarl.meta.data import DataHandler
from alpacarl.env.base import BaseEnv

interval = '15Min' # "1Min", "5Min", "15Min", "1H", "1D", "1W"
start = "2020-01-01"  # start time, min="2015-01-01"
end = "2020-06-08"  # End time, max=today's date

dh = DataHandler()
dh.set_symbols('Dow Jones')
prices = dh.prices(start, end, interval)
print(prices)