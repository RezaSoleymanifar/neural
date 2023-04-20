from collections import deque

import pandas_market_calendars as market_calendars
import pandas as pd

from alpaca.trading.enums import AssetClass
from neural.tools.enums import CalendarType


class Calendar:

    """
    A class representing a trading calendar for different asset classes.
    Attributes:
        asset_class (AssetClass): The asset class for which the calendar is created.
        calendar (market_calendars.MarketCalendar): The trading calendar object for the specified asset class.

    Methods:
        _get_calendar() -> market_calendars.MarketCalendar: Returns a trading calendar object based on the asset class.
        get_schedule(start_date, end_date) -> pd.DataFrame: Returns a schedule dataframe with trading dates and times.
        get_local_time_zone() -> str: Returns the local time zone for the specified calendar type.
    """

    def __init__(
        self, 
        asset_class: AssetClass
        ) -> None:

        """
        Initializes a new instance of the Calendar class.

        Args:
            asset_class (AssetClass): The asset class for which the calendar is created.
        """

        self.asset_class = asset_class
        self.calendar = None


    def _get_calendar(
        self
        ) -> market_calendars.MarketCalendar:

        """
        Returns a trading calendar object based on the asset class.

        Returns:
            market_calendars.MarketCalendar: The trading calendar object for the specified asset class.
        """

        asset_class_to_calendar_type = {
            AssetClass.US_EQUITY: CalendarType.US_EQUITY,
            AssetClass.CRYPTO: CalendarType.CRYPTO}

        calendar_type = asset_class_to_calendar_type.get(self.asset_class)
        calendar = market_calendars.get_calendar(calendar_type.value)

        return calendar


    def get_schedule(
            self, 
            start_date, 
            end_date
            ) -> pd.DataFrame:

        """
        Returns a schedule dataframe with trading dates and times.

        Args:
            start_date: The start date for the trading schedule.
            end_date: The end date for the trading schedule.

        Returns:
            pd.DataFrame: A dataframe with trading dates and times.
        """

        calendar = self._get_calendar()
        
        # Time returned is always UTC
        schedule = calendar.schedule(
            start_date=start_date, end_date=end_date)

        return schedule


    def get_local_time_zone(self) -> str:
        
        """
        Returns the local time zone for the specified calendar type.

        Returns:
            str: The local time zone for the calendar type.
        """

        calendar_type_to_time_zone = {
            CalendarType.CRYPTO: 'UTC',
            CalendarType.US_EQUITY: 'America/New_York'}

        time_zone = calendar_type_to_time_zone.get(self.calendar_type)
        return time_zone



class FillDeque:
    """
    A custom deque implementation that fills itself with the first item it receives 
    when it's empty until it reaches the specified buffer size. After that, it behaves 
    like a regular deque with a fixed maximum size.
    """

    def __init__(self, buffer_size):
        """
        Initializes the FillDeque instance with the specified buffer size.

        Args:
            buffer_size (int): The maximum size of the deque.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def append(self, item):
        """
        Appends the item to the deque. If the deque is empty, it fills the deque with
        the first item received until it reaches the maximum buffer size.

        Args:
            item: The item to append to the deque.
        """

        if not self.buffer:
            for _ in range(self.buffer_size):
                self.buffer.append(item)
        else:
            self.buffer.append(item)

    def to_list(self):
        """
        Returns the current deque buffer as a list.

        Returns:
            list: The current buffer as a list.
        """
        return list(self.buffer)



class RunningMeanSTD:
    pass