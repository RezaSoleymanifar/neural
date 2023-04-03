<div align="center">
<img align="center" src=figs/alpacarl.png width="100%"/>
</div>

[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)

# AlpacaRL
AlpacaRL is a deep reinforcement learning python package that is specifically designed for high-frequency stocks and crypto trading with Alpaca API.

The Alpaca API provides low-latency, commission-free trading of stocks and crypto assets, making it an ideal platform for high-frequency trading strategies that can benefit from the speed and efficiency of deep reinforcement learning algorithms.
 
Our library provides support for all stages of development to deployment processes, from data collection to feature engineering, model design, training, back-testing and final implementation in market environment.

AlpacaRL supports an array of state of the art DRL algorithms utilizing stable-baselines3 and PyTorch. AlpacaRL emphasizes a minimal and pythonic interface that eliminates continuity concerns or boilerplate code, by encapsulating logical compoents of high frequency trading process. Data retrieval, model training and inference operations are highly optimized thanks to HDF5 file format, GPU processing support, and C-level Cython implementation of bottlenecks.