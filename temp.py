from alpacarl.meta.config import ALPACA_API_PAPER_URL, ALPACA_API_BASE_URL
from alpacarl.core.data import DataHandler
from alpacarl.env.base import BaseEnv


interval = '1Min' # "1Min", "5Min", "15Min", "1H"
start = "2017-01-01"  # start time, min="2017-01-01"
end = "2017-02-01"  # End time, max=today

dh = DataHandler()
dh.connect(ALPACA_API_PAPER_URL)
dh.symbols = 'Dow Jones'
dh.download(start, end, interval, dir = './assets/data')

env = BaseEnv(features)

# obs = env.reset()
# while True:
#     action = env.action_space.sample()  # choose a random action
#     obs, reward, done, history = env.step(action)
#     if done:
#         obs = env.reset()


# def self.download(start, end, interval, quotes = True, trades = False, market_times = True):
#     generators = dict{source: generators}
#     working_days = get_working_days()
#     for day in working days:
#         for source in sources:
#             process(generator)
        