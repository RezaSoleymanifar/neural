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
from enum import Enum
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
    This is a class for creating trading calendars for different exchanges. The
    default trading calendars are based on pandas_market_calendars. More
    information can be found here:
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


class Resolution:
    """
    Class for representing the resolution of a dataset or stream. The
    resolution is the fixed length of time interval in the dataset or
    stream. The resolution is represented as a string with the quantity
    and unit of time.

    Attributes:
    -----------
        quantity (int):
            An integer representing the quantity of the resolution.
        unit (Resolution.Unit):
            An instance of the `Resolution.Unit` enum class representing
            the unit of time of the resolution.

    Methods:
    --------
        validate_resolution(quantity: int, unit: Resolution.Unit) -> None:
            Validates the resolution.

    Raises:
    -------
        TypeError:
            If quantity is not an integer or unit is not an instance of
            Resolution.Unit.
        ValueError:
            If quantity is not a positive integer or if unit is not
            compatible with the quantity.

    Example:
    --------
        >>> resolution = Resolution(43, Resolution.Unit.NANO_SECOND)
        >>> print(resolution)
        43Unit.NANO_SECOND
    """
    def __init__(self, quantity: int, unit: Unit):
        """
        Initializes a new instance of the Resolution class.

        Args:
        -----
            quantity (int):
                An integer representing the quantity of the resolution unit.
            unit (Resolution.Unit):
                An instance of the `Resolution.Unit` enum class
                representing the unit of time of the resolution.
        """
        self.validate_resolution(quantity, unit)
        self.quantity = quantity
        self.unit = unit

    @property
    def pandas_timedelta(self):
        """
        Returns a pandas Timedelta object representing the resolution.

        Returns:
        --------
            pd.Timedelta:
                A pandas Timedelta object representing the resolution.
        """
        pandas_unit = {
            Resolution.Unit.NANO_SECOND: 'ns',
            Resolution.Unit.MICRO_SECOND: 'us',
            Resolution.Unit.MILLI_SECOND: 'ms',
            Resolution.Unit.SECOND: 'S',
            Resolution.Unit.MINUTE: 'T',
            Resolution.Unit.HOUR: 'H',
            Resolution.Unit.DAY: 'D'
        }
        unit_str = pandas_unit[self.unit]
        timedelta_str = f"{self.quantity}{unit_str}"
        return pd.Timedelta(timedelta_str)

    def validate_resolution(self, quantity: int, unit: Unit):
        """
        Validates the resolution.

        Args:
        -----
            quantity (int):
                An integer representing the quantity of the resolution
                unit.
            unit (Resolution.Unit):
                An instance of the `Resolution.Unit` enum class
                representing the unit of time of the resolution.

        Raises:
        -------
            TypeError:
                If quantity is not an integer or unit is not an instance
                of Resolution.Unit.
            ValueError:
                If quantity is not a positive integer or if unit is not
                compatible with the quantity.
        """
        if not isinstance(quantity, int):
            raise TypeError("Quantity must be an integer value.")
        if not isinstance(unit, Resolution.Unit):
            raise TypeError(
                f"Unit must be an instance of {Resolution.Unit.__name__}.")

        if quantity <= 0:
            raise ValueError("Quantity must be a positive integer value.")
        if unit == Resolution.Unit.NANO_SECOND and quantity > 999:
            raise ValueError(
                "Nanosecond units can only be used with quantities between 1-999."
            )
        if unit == Resolution.Unit.MICRO_SECOND and quantity > 999:
            raise ValueError(
                "Microsecond units can only be used with quantities between 1-999."
            )
        if unit == Resolution.Unit.MILLI_SECOND and quantity > 999:
            raise ValueError(
                "Millisecond units can only be used with quantities between 1-999."
            )
        if unit == Resolution.Unit.SECOND and quantity > 59:
            raise ValueError(
                "Second units can only be used with quantities between 1-59.")
        if unit == Resolution.Unit.MINUTE and quantity > 59:
            raise ValueError(
                "Minute units can only be used with quantities between 1-59.")
        if unit == Resolution.Unit.HOUR and quantity > 23:
            raise ValueError(
                "Hour units can only be used with quantities between 1-23.")
        if unit == Resolution.Unit.DAY and quantity != 1:
            raise ValueError(
                "Day units can only be used with quantities equal to 1.")

        return None

    def __eq__(self, other):
        """
        Checks if two resolutions are equal. Two resolutions are equal if
        their quantities and units are equal.
        """
        if isinstance(other, Resolution):
            return self.quantity == other.quantity and self.unit == other.unit
        return False

    def __str__(self):
        """
        String representation of the resolution. This is useful for
        printing the resolution.

        Returns:
        --------
            str:
                A string representation of the resolution.
        
        Example:
        --------
            >>> resolution = Resolution(43, Resolution.Unit.NANO_SECOND)
            >>> print(resolution)
            43Unit.NANO_SECOND
        """
        return f"{self.quantity}{self.unit}"

    class Unit(Enum):
        """
        Enumeration class that defines the available units of time for
        representing the resolution of a dataset or stream. The
        resolution is the fixed length of time interval in the dataset
        or stream. The resolution is represented as a string with the
        quantity and unit of time.

        Attributes:
        -----------
            NANO_SECOND (str):
                A string representing the nanosecond unit of time.
            MICRO_SECOND (str):
                A string representing the microsecond unit of time.
            MILLI_SECOND (str):
                A string representing the millisecond unit of time.
            SECOND (str):
                A string representing the second unit of time.
            MINUTE (str):
                A string representing the minute unit of time.
            HOUR (str):
                A string representing the hour unit of time.
            DAY (str):
                A string representing the day unit of time.
        """
        NANO_SECOND = 'NANO_SECOND'
        MICRO_SECOND = 'MICRO_SECOND'
        MILLI_SECOND = 'MILLI_SECOND'
        SECOND = 'SECOND'
        MINUTE = 'MINUTE'
        HOUR = 'HOUR'
        DAY = 'DAY'
