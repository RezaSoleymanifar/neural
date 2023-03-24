from alpacarl.meta.config import ALPACA_API_PAPER_URL, ALPACA_API_BASE_URL
from alpacarl.core.data import DataHandler
from alpacarl.env.base import TrainEnv
import pickle


interval = '15Min' # "1Min", "5Min", "15Min", "1H"
start = "2020-01-01"  # start time, min="2015-01-01"
end = "2020-02-08"  # End time, max=today's date

dh = DataHandler()
dh.connect(ALPACA_API_PAPER_URL)
dh.symbols = 'Dow Jones'
prices, features, scaler = dh.featurize(start, end, interval)

with open('./assets/scalers/scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

with open('./assets/scalers/scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

env = TrainEnv(prices, features)

obs = env.reset()
while True:
    action = env.action_space.sample()  # choose a random action
    obs, reward, done, history = env.step(action)
    if done:
        obs = env.reset()

