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
<h1>neuralHFT: <span style="font-size: 80%;">Deep reinforcement learning for high frequency trading.</span></h1>


neuralHFT is a pythonic libarary that is written and optimized for high-frequency stocks and crypto trading. neuralHFT is compatible with state of the art DRL algorithms implemented with [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/), and [PyTorch](https://pytorch.org/) out of the box and with no boilerplate code required. Seamless integration with [Alpaca API](https://alpaca.markets/) is offered by default allowing standalone and end-to-end support from training to launching HFT algorithms in your live/paper trading account. The Alpaca API provides low-latency, commission-free trading of stocks and crypto assets, making it an ideal platform for high-frequency trading strategies.

The standalone design enables users with granular control over dataset creation, feature engineering, model architecture, market simulation, reward shaping and low level control over training in a clean and pythonic fashion. This enables users to shift focus from time-consuming initial setup to more rapid prottyping, and model discovery.