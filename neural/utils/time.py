"""
time.py

A module for working with time and dates. This module contains a class
for creating trading calendars for different asset classes. The trading
calendars are based on pandas_market_calendars. The module also contains
a class for creating custom calendars that are not supported by
pandas_market_calendars.
"""

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any, TYPE_CHECKING

import pandas as pd
import pandas_market_calendars as market_calendars

if TYPE_CHECKING:
    from neural.data.enums import CalendarType


class AbstractCalendar(ABC):
    """
    An abstract base class representing a trading calendar. This class
    should be inherited by custom calendars that are not supported by
    pandas_market_calendars. Link:
    https://pandas-market-calendars.readthedocs.io/en/latest/usage.html.

    Methods:
    ---------
        schedule(calendar_type, start_date, end_date) ->
        pd.DataFrame:
            Returns a schedule dataframe with trading dates and times.
    """

    @staticmethod
    @abstractmethod
    def schedule(
        calendar_type: CalendarType,
        start_date: Any,
        end_date: Any,
    ) -> pd.DataFrame:
        """
        An abstract method that returns a DataFrame of start and
        end date times corresponding to a working day on this
        calendar. The DataFrame should have columns 'start' and
        'end' and index as a DatetimeIndex of dates. The times
        reported should be time zone naive and in UTC.

        Args:
        ---------
            calendar_type (CalendarType):
                The type of calendar to use.
            start_date (Any):
                The start date for the trading schedule.
            end_date (Any):
                The end date for the trading schedule.

        Example:
        ---------
                    start	                    end
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
    This is a class for creating trading calendars for different
    exchanges. The default trading calendars are based on
    pandas_market_calendars. More information can be found here:
    https://pandas-market-calendars.readthedocs.io/en/latest/usage.html.

    Attributes:
    ---------
        calendar_names: list
            A list of supported calendar names.

    Methods:
    ---------
        schedule(calendar_type, start_date, end_date) -> pd.DataFrame:
            Returns a schedule dataframe with trading dates and times.

    Example:
    ---------
    Option 1, using default calendar:
        >>> from neural.data.enums import CalendarType
        >>> from neural.utils.time import Calendar

        >>> calendar = Calendar()
        >>> schedule = calendar.schedule(
        ... calendar_type=CalendarType.TWENTY_FOUR_SEVEN,
        ... start_date='2022-01-01',
        ... end_date='2022-01-10')

        >>> schedule
                    start                       end
        2022-01-03  2022-01-03 00:00:00+00:00   2022-01-04 00:00:00+00:00
        2022-01-04  2022-01-04 00:00:00+00:00   2022-01-05 00:00:00+00:00
        2022-01-05  2022-01-05 00:00:00+00:00   2022-01-06 00:00:00+00:00
        2022-01-06  2022-01-06 00:00:00+00:00   2022-01-07 00:00:00+00:00
        2022-01-07  2022-01-07 00:00:00+00:00   2022-01-08 00:00:00+00:00
        2022-01-10  2022-01-10 00:00:00+00:00   2022-01-11 00:00:00+00:00

    Option 2, using custom calendar:

        >>> from neural.data.enums import CalendarType

        >>> calendar = CustomCalendar()
        >>> CalendarType.CUSTOM_CALENDAR_TYPE = 'CUSTOM_CALENDAR_TYPE'
        >>> schedule = calendar.schedule(
        ... calendar_type=CalendarType.CUSTOM_CALENDAR_TYPE,
        ... start_date='2022-01-01',
        ... end_date='2022-01-10')
    """

    @property
    def calendar_names(self) -> list:
        """
        Returns a list of supported calendar names.

        Returns:
        ---------
            list: A list of supported calendar names.
        """
        calendar_names = market_calendars.get_calendar_names()
        return calendar_names

    @staticmethod
    def schedule(calendar_type: CalendarType, start_date: Any,
                 end_date: Any) -> pd.DataFrame:
        """
        Returns a schedule dataframe with core trading open and close
        times per day.

        Args:
        ---------
            calendar_type (CalendarType):
                The type of calendar to use.
            start_date (Any):
                The start date for the trading schedule. The date
                can be anything that can be read by pandas to_datetime.
                examples include pd.Timestamp, datetime.datetime, str.
        Returns:
        ---------
            pd.DataFrame: A dataframe with trading dates and times.
        """

        calendar = market_calendars.get_calendar(calendar_type.value)

        # Time returned is always UTC
        schedule_dataframe = calendar.schedule(start_date=start_date,
                                               end_date=end_date)
        schedule_dataframe = schedule_dataframe.rename(columns={
            "market_open": "start",
            "market_close": "end"
        })
        
        return schedule_dataframe
