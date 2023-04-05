from enums import CalendarType
from typing import List, Iterable

import pandas_market_calendars as market_calendars
import pandas as pd
import tableprint
import tqdm


class Calendar:

    def __init__(self, calendar_type=CalendarType) -> None:
        self.calendar_type = calendar_type
        self.calendar = None

    def get_calendar(self):

        calendar = market_calendars.get_calendar(self.calendar_type.value)

        return calendar

    # get core hours of calendar
    def get_schedule(self, start_date, end_date):

        self.calendar = self.get_calendar()

        schedule = self.calendar.schedule(
            start_date=start_date, end_date=end_date)

        return schedule

    def get_time_zone(self) -> str:

        if self.calendar_type == Calendar.ALWAYS_OPEN:
            time_zone = 'UTC'

        elif self.calendar_type == Calendar.NYSE:
            time_zone = 'America/New_York'

        return time_zone
    

def sharpe(assets_hist: List[float], base=0):

    hist = pd.Series(assets_hist)
    returns = hist.pct_change().dropna()
    val = (returns.mean()-base)/returns.std()

    return val

def tabular_print(
        entries: List, style='banner',
        align='left', width = 15, header = False) -> None:
    
    # helper method to tabulate performance metrics.
    if header:
        row = tableprint.header(
            entries, style=style, align=align, width=width)

    else:
        row = tableprint.row(
            entries, style=style, align=align, width=width)

    return row

def progress_bar(iterable: Iterable):
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}'
    bar = tqdm(total = iterable, bar_format = bar_format)
    return bar
