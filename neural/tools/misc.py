from collections import deque

import pandas_market_calendars as market_calendars
import pandas as pd
import numpy as np

from alpaca.trading.enums import AssetClass
from neural.tools.enums import CalendarType
from neural.common.constants import GLOBAL_DATA_TYPE


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



class FillDeque(deque):

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
        self.buffer = deque(maxlen=self.buffer_size)

        return None


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
        
        return None


    def __iter__(self):

        return iter(self.buffer)


    def __getitem__(self, index):

        """
        Returns a slice of the buffer as a list.

        Args:
            index (int, slice): The index or slice to retrieve.

        Returns:
            list: A list of items from the buffer.
        """

        if isinstance(index, int):
            return self.buffer[index]
        
        elif isinstance(index, slice):
            return list(self.buffer)[index]
        
        else:
            raise TypeError("Invalid argument type")
        
    
    def clear(self):

        """
        Removes all elements from the deque buffer.
        """
        self.buffer.clear()

        return None



class RunningMeanStandardDeviation:

    """
    A class for computing the running mean and standard deviation of a series of data.
    Can be used to normalize data to a mean of 0 and standard deviation of 1 in an online
    fashion.

    Implements the Welford online algorithm for computing the standard deviation.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    Usage:
        rms = RunningMeanStandardDeviation()
        rms.update(array)
        mean = rms.mean
        std = rms.std
        normalized_array = rms.normalize(array)

    Args:
        epsilon (float): A small constant to avoid divide-by-zero errors when normalizing data.
        clip (float): A value to clip normalized data to, to prevent outliers from dominating the statistics.
    """

    def __init__(self, epsilon=1e-8, clip_threshold: float = np.inf):

        """
        Initializes the RunningMeanStandardDeviation object.
        """

        assert clip_threshold > 0, "Clip threshold must be greater than 0"
        assert epsilon > 0, "Epsilon value must be greater than 0"

        self.epsilon = epsilon
        self.clip = clip_threshold

        self.shape = None
        self._minimum = None
        self._maximum = None
        self._mean = None
        self._std = None
        self.M2 = None
        self.count = None

        return None
    

    @property
    def minimum(self):
            
        """
        Returns the minimum value of the data stored in the RunningMeanStandardDeviation object.
        """

        assert self.count, "Must have at least one data point to compute minimum"

        return self._minimum


    @property
    def maximum(self):
            
        """
        Returns the max value of the data stored in the RunningMeanStandardDeviation object.
        """
        assert self.count, "Must have at least one data point to compute maximum"

        return self._minimum
    

    @property
    def mean(self):

        """
        Computes and returns the mean of the data stored in the RunningMeanStandardDeviation object.
        """

        assert self.count, "Must have at least one data point to compute mean"

        return self._mean


    @property
    def std(self):

        """
        Computes and returns the standard deviation of the data stored in the RunningMeanStandardDeviation object.
        """

        assert self.count, "Must have at least one data point to compute standard deviation"

        variance = self.M2 / (self.count - 1) if self.count > 1 else np.zeros_like(self.M2)
        self._std = np.sqrt(variance)

        return self._std


    def initialize_rms(self, array: np.ndarray):

        """
        Initializes the RunningMeanStandardDeviation object with data.

        Args:
            x (np.ndarray): The data to initialize the RunningMeanStandardDeviation object with.
        """

        self.shape = array.shape
        self._mean = np.zeros(self.shape)
        self.M2 = np.zeros(self.shape)
        self.count = 0

        self.minimum = np.inf
        self.maximum = -np.inf

        return None


    def update(self, array: np.ndarray):

        """
        Updates the RunningMeanStandardDeviation object with new data.

        Args:
            x (np.ndarray): The new data to be added to the RunningMeanStandardDeviation object.
        """

        if self.shape is None:
            self.initialize_rms(array)

        assert self.shape == array.shape, "Shape of data has changed during update."

        self.count += 1
        delta = array - self._mean
        self._mean += delta / self.count
        delta2 = array - self._mean
        self.M2 += delta * delta2

        self.minimum = np.minimum(self.minimum, array)
        self.maximum = np.maximum(self.maximum, array)

        return None
    
    def normalize(self, array: np.ndarray):

        # Normalize the array using the running mean and standard deviation
        normalized_array = np.clip(
            (array - self.mean) / (self.std + self.epsilon), -self.clip, self.clip)
        
        return normalized_array
