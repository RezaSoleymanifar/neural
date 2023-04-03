<div align="center">
<img align="center" src=figs/alpacarl.png width="100%"/>
</div>

[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)

<h1></h1><h2></h2>
<h1>AlpacaRL: <span style="font-size: 80%;">Deep reinforcement learning for high frequency trading</span></h1>


AlpacaRL is a pythonic libarary that is written and optimized for high-frequency stocks and crypto trading. AlpacaRL offers seamless integration with Alpaca API allowing end-to-end support from development to deployment of DRL algorithms.

The Alpaca API provides low-latency, commission-free trading of stocks and crypto assets, making it an ideal platform for high-frequency trading strategies that can benefit from the speed and efficiency of deep reinforcement learning algorithms.
 
Our library provides end-to-end support from data collection to feature engineering, model design, training, back-testing and final implementation in market environment. AlpacaRL supports a host of state of the art DRL algorithms implemented in stable-baselines3 and PyTorch. Data retrieval, model training and inference operations are highly optimized thanks to HDF5 file format, GPU processing support, and C-level implementation in training bottlenecks, leading to drastic speed-up of model prototyping.