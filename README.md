<div align="center">
<img align="center" src=assets/neural.gif width="100%"/>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)

<h1></h1><h2></h2>
<h1>neuralHFT: <span style="font-size: 70%;">deep reinforcement learning for high frequency trading.</span></h1>


neuralHFT is a pythonic libarary that is written and optimized for high-frequency stocks and crypto trading. neuralHFT offers seamless integration with [Alpaca API](https://alpaca.markets/) allowing end-to-end support from development to deployment of DRL algorithms.

The Alpaca API provides low-latency, commission-free trading of stocks and crypto assets, making it an ideal platform for high-frequency trading strategies. Our library provides end-to-end support from data collection to feature engineering, model design, training, back-testing and final implementation in market environment. 

neuralHFT supports a host of state of the art DRL algorithms implemented in [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) and [PyTorch](https://pytorch.org/). Data retrieval, model training and inference operations are highly optimized thanks to [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file format, GPU processing support, and C-level implementation in training bottlenecks, leading to drastic speed-up of model prototyping.