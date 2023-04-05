from enum import Enum

class CalendarType(Enum):
    NYSE = 'NYSE'
    ALWAYS_OPEN = '24/7'
import pandas_market_calendars as market_calendars
