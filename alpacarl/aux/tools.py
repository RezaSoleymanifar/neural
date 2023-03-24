import pandas as pd

def sharpe(assets_hist, base=0):
    hist = pd.Series(assets_hist)
    returns = hist.pct_change().dropna()
    val = (returns.mean()-base)/returns.std()
    return val

class RewardShaper:
    def __init__(self) -> None:
        pass
    def discount(self):
        pass
    def standardize(self):
        pass