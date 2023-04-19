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

        calendar = self._get_calendar()
        
        # Time returned is always UTC
        schedule = calendar.schedule(
            start_date=start_date, end_date=end_date)

        return schedule


    def get_local_time_zone(self) -> str:

        calendar_type_to_time_zone = {
            CalendarType.CRYPTO: 'UTC',
            CalendarType.US_EQUITY: 'America/New_York'}

        time_zone = calendar_type_to_time_zone.get(self.calendar_type)
        return time_zone
    

class RunningMeanSTD:
    pass