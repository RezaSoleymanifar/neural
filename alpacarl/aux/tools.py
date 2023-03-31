import pandas as pd
from typing import List, Iterable
import tableprint
import tqdm

def sharpe(assets_hist: List[float], base=0):
    hist = pd.Series(assets_hist)
    returns = hist.pct_change().dropna()
    val = (returns.mean()-base)/returns.std()
    return val

def tabular_print(entries: List, style='banner', align='left', width = 15, header = False) -> None:
    # helper method to tabulate performance metrics.
    if header:
        row = tableprint.header(entries, style=style, align=align, width=width)
    else:
        row = tableprint.row(entries, style=style, align=align, width=width)
    print(row)
    return None

def progress_bar(iterable: Iterable):
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}'
    bar = tqdm(total = iterable, bar_format = bar_format)
    return bar
