from alpaca.trading.enums import AssetClass
from neural.tools.enums import CalendarType
import pandas_market_calendars as market_calendars
import pandas as pd

class Calendar:

    def __init__(
        self, 
        asset_class: AssetClass
        ) -> None:
        
        self.asset_class = asset_class
        self.calendar = None

    def _get_calendar(
        self
        ) -> market_calendars.MarketCalendar:

        if self.asset_class == AssetClass.US_EQUITY:

            calendar_type = CalendarType.US_EQUITY
        
        elif self.asset_type == AssetClass.CRYPTO:

            calendar_type = CalendarType.CRYPTO

        calendar = market_calendars.get_calendar(calendar_type.value)

        return calendar

    # get core hours of calendar
    def get_schedule(
            self, 
            start_date, 
            end_date
            ) -> pd.DataFrame:

        calendar = self._get_calendar()
        
        # Time returned is always UTC
        schedule = calendar.schedule(
            start_date=start_date, end_date=end_date)

        return schedule

    def get_local_time_zone(self) -> str:

        if self.calendar_type == CalendarType.CRYPTO:
            time_zone = 'UTC'

        elif self.calendar_type == CalendarType.US_EQUITY:
            time_zone = 'America/New_York'

        return time_zone
    