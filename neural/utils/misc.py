"""
misc.py 

Miscellaneous utility functions. These are less generic more use-case
specific facilities.

Methods:
--------
    objects_list_to_dataframe(objects_list: List[object]) ->
    pd.DataFrame:
        Converts a collection of objects into a pandas DataFrame. Uses
        dictionary representation of the objects to create the
        dataframe. Used in the AlpacaClient class to convert the assets
        objects into a DataFrame, or convert positions objects into a
        dataframe. If a value is Enum, its string representation is used
        in dataframe.
    
    to_timeframe(timeframe: str) -> TimeFrame:
        Parses a string representation of a timeframe into a TimeFrame
        object.
"""
from typing import List
from enum import Enum
import re

import pandas as pd

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from neural.data.base import Resolution

def objects_list_to_dataframe(
    objects_list: List[object]
    ) -> pd.DataFrame:
    """
    Converts a collection of objects into a pandas DataFrame. Uses
    dictionary representation of the objects to create the dataframe.
    Used in the AlpacaClient class to convert the assets objects into a
    DataFrame, or convert positions objects into a dataframe. If a value
    is Enum, its string representation is used in dataframe.

    Args:
    ------
        objects_list List[object]:
            A list of objects, where each object's dictionary
            representation is used to create the dataframe.

    Returns:
    ---------
        pd.DataFrame:
            A DataFrame containing the information on objects.
    """
    objects_list_ = objects_list.copy()
    for index, object_ in enumerate(objects_list_):
        object_dict = dict(object_)
        for attribute_name, attribute in object_dict.items():
            object_dict[attribute_name] = (
                attribute.value if isinstance(attribute, Enum) else attribute)
            objects_list_[index] = object_dict
    return pd.DataFrame(objects_list_)


def resolution_to_timeframe(resolution: Resolution) -> TimeFrame:

    amount = resolution.quantity
    unit = match.group(2)

    resolution_to_timeframe_map= {
        Resolution.Unit.MINUTE: TimeFrameUnit.Minute,
        Resolution.Unit.HOUR: TimeFrameUnit.Hour,
        Resolution.Unit.DAY: TimeFrameUnit.Day,
    }

    return TimeFrame(amount, timeframe_map[unit])
