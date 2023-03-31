from alpacarl.meta.config import ALPACA_API_PAPER_URL, ALPACA_API_BASE_URL
from alpacarl.core.data import AlpacaMetaClient, RowGenerator
from alpacarl.env.base import BaseEnv, FractionalActionWrapper
from alpacarl.core.data import DatasetType

interval = '1Min'  # "1Min", "5Min", "15Min", "1H"
start = "2018-01-02"  # start time, min="2017-01-01"
end_date = "2018-01-04"  # End time, max=today

meta_client = AlpacaMetaClient(sandbox=True)
meta_client.setup_clients_and_account()

dataset_type = DatasetType.BAR

# dh.download(start, end, interval, dir = './assets/data')
# data = RowGenerator(dir='assets/data/data.csv', chunk = 100_000)


# env = BaseEnv(data)
# env = FractionalActionWrapper(env)

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
