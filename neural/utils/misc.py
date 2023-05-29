from typing import Iterable, Dict
from enum import Enum

import pandas as pd
import re

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit



def objects_to_dataframe(
    object_collection: Iterable[Dict[str, str]]
    ) -> pd.DataFrame:
    """
    Converts a collection of enum objects into a pandas DataFrame.
    Used in the AlpacaClient class to convert the assets property into a DataFrame.

    Args:
    ------
        object_collection (Iterable[Dict[str, str]]): A collection of enum objects, 
        where each object is a dictionary containing key-value pairs.

    Returns:
        pd.DataFrame: A DataFrame containing the converted objects.
    """

    objects_collection_ = object_collection.copy()
    for index, object in enumerate(objects_collection_):
        object_dict = dict(object)

        for key, val in object_dict.items():
            object_dict[key] = val.value if isinstance(val, Enum) else val
            objects_collection_[index] = object_dict

    df = pd.DataFrame(objects_collection_)
    return df



def to_timeframe(time_frame: str):

    """
    Parses a string representation of a time frame into a TimeFrame object.

    Args:
    ------
        time_frame (str): A string representing a time frame, in the format "<amount><unit>",
            where <amount> is an integer and <unit> is one of 'Min', 'Hour', 'Day', 'Week', or 'Month'.

    Returns:
        TimeFrame: A TimeFrame object representing the parsed time frame.

    Raises:
        ValueError: If the input string is not a valid time frame. Valid time frame examples include
            '59Min', '23Hour', '1Day', '1Week', and '12Month'.
    """
    

    match = re.search(r'(\d+)(\w+)', time_frame)

    if not match:
        raise ValueError(
            "Invalid timeframe. Valid examples: 59Min, 23Hour, 1Day, 1Week, 12Month")

    amount = int(match.group(1))
    unit = match.group(2)

    map = {
        'Min': TimeFrameUnit.Minute,
        'Hour': TimeFrameUnit.Hour,
        'Day': TimeFrameUnit.Day,
        'Week': TimeFrameUnit.Week,
        'Month': TimeFrameUnit.Month}

    return TimeFrame(amount, map[unit])

