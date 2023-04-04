from alpacarl.meta.constants import ALPACA_API_PAPER_URL, ALPACA_API_BASE_URL, CUSTOM_SYMBOLS
from alpacarl.core.data.ops import RowGenerator, DatasetDownloader
from alpacarl.env.base import BaseEnv, FractionalActionWrapper
from alpacarl.core.data.ops import DatasetType, DatasetIO
from alpacarl.core.client import AlpacaMetaClient


path = ''
symbols = ['AAPL']
start_date = "2018-01-02"  # start time, min="2017-01-01"
end_date = "2018-01-04"  # End time, max=today
resolution = '1Min'  # "1Min", "5Min", "15Min", "1H"


metadata, _ = DatasetIO.load_from_hdf5(path = path)
client = AlpacaMetaClient(sandbox=True)
client.setup_clients_and_account()
data_downloader = DatasetDownloader(client)
DatasetDownloader.download_and_write_dataset(
    path, DatasetType.BAR, symbols, resolution, start_date, end_date)

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
