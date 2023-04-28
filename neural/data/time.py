from abc import abstractmethod, ABC
from typing import Any
from enum import Enum

import pandas as pd
import pandas_market_calendars as market_calendars

from neural.data.enums import CalendarType



class CalendarType(Enum):

    """

    If an asset does not fall under these three categories it can be handled by user speciying the
    CalendarType.MY_CALENDAR_TYPE = 'VALID_PANDAS_CALENDAR' and providing a valid pandas market calendar.
    More information here: https://pandas-market-calendars.readthedocs.io/en/latest/modules.html.

    """

    NEW_YORK_STOCK_EXCHANGE = 'NYSE'
    TWENTY_FOUR_SEVEN = '24/7'
    TWENTY_FOUR_FIVE = '24/5'
    CHICAGO_MERCANTILE_EXCHANGE = 'CME'
    INTERCONTINENTAL_EXCHANGE = 'ICE'
    LONDON_STOCK_EXCHANGE = 'LSE'
    TOKYO_STOCK_EXCHANGE = 'TSE'
    SINGAPORE_EXCHANGE = 'SGX'
    AUSTRALIAN_SECURITIES_EXCHANGE = 'ASX'
    MOSCOW_EXCHANGE = 'MOEX'
    BME_SPANISH_EXCHANGES = 'BM'
    BOVESPA = 'FBOVESPA'
    JOHANNESBURG_STOCK_EXCHANGE = 'JSE'
    SHANGHAI_STOCK_EXCHANGE = 'SSE'
    SHENZHEN_STOCK_EXCHANGE = 'SZSE'
    HONG_KONG_EXCHANGES_AND_CLEARING = 'HKEX'
    NATIONAL_STOCK_EXCHANGE_OF_INDIA = 'NSE'
    BOMBAY_STOCK_EXCHANGE = 'BSE'
    KOREA_EXCHANGE = 'KRX'
    TAIWAN_STOCK_EXCHANGE = 'TWSE'


    @property
    def schedule(self):
        return Calendar(self.value).schedule


class AbstractCalendar(ABC):

    """
    An abstract base class representing a trading calendar. This class should be inherited 
    by custom calendars that are not supported by pandas_market_calendars.
    Link: https://pandas-market-calendars.readthedocs.io/en/latest/usage.html.
    """

    @abstractmethod
    def schedule(self, caelndar_type: CalendarType) -> pd.DataFrame:
        """
        An abstract method that returns a DataFrame of market_open and market_close date times
        corresponding to a working day on this calendar. The DataFrame should have columns 'market_open'
        and 'market_close' and index as a DatetimeIndex of dates. The times reported should be time zone
        naive and in UTC.

        Example:
        ---------
        	        market_open	                market_close
        2022-01-03	2022-01-03 00:00:00+00:00	2022-01-04 00:00:00+00:00
        2022-01-04	2022-01-04 00:00:00+00:00	2022-01-05 00:00:00+00:00
        2022-01-05	2022-01-05 00:00:00+00:00	2022-01-06 00:00:00+00:00
        2022-01-06	2022-01-06 00:00:00+00:00	2022-01-07 00:00:00+00:00
        2022-01-07	2022-01-07 00:00:00+00:00	2022-01-08 00:00:00+00:00
        2022-01-10	2022-01-10 00:00:00+00:00	2022-01-11 00:00:00+00:00
        """
        raise NotImplementedError

class Calendar:

    """
    A class representing a trading calendar for different asset classes.

    Attributes:
    ---------
        asset_class (AssetClass): The asset class for which the calendar is created.
        calendar (market_calendars.MarketCalendar): The trading calendar object for the specified asset class.

    Methods:
    ---------
        _get_calendar() -> market_calendars.MarketCalendar: Returns a trading calendar object based on the 
        asset class. 
        get_schedule(start_date, end_date) -> pd.DataFrame: Returns a schedule dataframe with
        trading dates and times.
        get_local_time_zone() -> str: Returns the local time zone for the specified calendar type.
    """

    def __init__(self, calendar_type: CalendarType) -> None:
        
        """
        Initializes a new instance of the Calendar class.

        Args:
            asset_class (AssetClass): The asset class for which the calendar is created.
        """
        
        self.calendar_type = calendar_type
        self.calendar = None

        return None


    def schedule(
        self,
        start_date: Any,
        end_date: Any
        ) -> pd.DataFrame:
        
        """
        Returns a schedule dataframe with core trading open and close times
        per day.

        Args:
        ---------
            start_date: The start date for the trading schedule.
            end_date: The end date for the trading schedule.

        Returns:
        ---------
            pd.DataFrame: A dataframe with trading dates and times.
        """

        calendar = market_calendars.get_calendar(self.calendar_type.value)

        # Time returned is always UTC
        schedule_dataframe = calendar.schedule(start_date=start_date, end_date=end_date)

        return schedule_dataframe


