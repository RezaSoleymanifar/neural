from alpacarl.core.data import DataHandler
from alpacarl.env.base import BaseEnv

interval = '15Min' # "1Min", "5Min", "15Min", "1H", "1D", "1W"
start = "2020-01-01"  # start time, min="2015-01-01"
end = "2020-06-08"  # End time, max=today's date

dh = DataHandler()
dh.connect()
dh.symbols = 'Dow Jones'
print(dh.symbols)


# env = CustomEnv()
# env.connect()
# observation = env.reset()
# for t in range(100):
#     action = env.action_space.sample()  # choose a random action
#     observation, reward, done, info = env.step(action)
#     if done:
#         print(f"Episode finished after {t+1} timesteps")
#         break
# env.render()