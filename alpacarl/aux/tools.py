import pandas as pd
from typing import List
import tableprint

def sharpe(assets_hist, base=0):
    hist = pd.Series(assets_hist)
    returns = hist.pct_change().dropna()
    val = (returns.mean()-base)/returns.std()
    return val

def print_(entries: List, style='banner', align='left', width = 15, header = False) -> None:
    # helper method to tabulate performance metrics.
    if header:
        row = tableprint.header(entries, style=style, align=align, width=width)
    else:
        row = tableprint.row(entries, style=style, align=align, width=width)
    print(row)
    return None

class RewardShaper:
    def __init__(self) -> None:
        pass
    def discount(self):
        pass
    def standardize(self):
        pass