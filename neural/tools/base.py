from datetime import datetime
from typing import List, Iterable, Dict
from enum import Enum

import pandas as pd
import tableprint, re, os
from tqdm import tqdm

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from neural.data.enums import ColumnType


def validate_path(
    file_path: str | os.PathLike
    ) -> None:

    """
    Validates a file path by checking if it is a directory and if the parent directory exists.
    Args:
        file_path (str | os.PathLike): The file path to be validated.

    Raises:
        ValueError: If the specified path is a directory or if the parent directory does not exist.

    Returns:
        None
    """

    if os.path.isdir(file_path):
        raise ValueError(
            f"The specified path {file_path} is a directory, not a file.")
    
    else:
        dir_path = os.path.dirname(file_path)

        if not os.path.isdir(dir_path):
            raise ValueError(
                f"Directory {dir_path} leading to the specified file does not exist.")
        
    return None



def create_column_schema(data: pd.DataFrame):

    """
    Creates a column schema dictionary for a given DataFrame, with ColumnType as keys and boolean masks as values.
    Args:
        data (pd.DataFrame): The input DataFrame for which the column schema is to be created.

    Returns:
        Dict[ColumnType, pd.Series]: A dictionary containing ColumnType keys and boolean masks for each column in the input DataFrame.
    """

    column_schema = dict()

    for column_type in ColumnType:

        mask = data.columns.str.match(column_type.value.lower())
        column_schema[column_type] = mask

    return column_schema



def to_datetime(date: str):

    try:
        date_format = "%d/%m/%Y"
        date_time_ = datetime.strptime(date, date_format)

    except:
        raise ValueError(
            'Invalid date. Valid examples: 20/03/2018, 01/01/2015'
            )

    return date_time_



def to_timeframe(time_frame: str):

    """
    Converts a string representation of a date to a datetime object.

    Args:
        date (str): The input date string to be converted to a datetime object. Must be in the format "dd/mm/yyyy".

    Raises:
        ValueError: If the input date string is not in the correct format.

    Returns:
        datetime: A datetime object corresponding to the input date string.
    """

    match = re.search(r'(\d+)(\w+)', time_frame)

    if match:

        amount = int(match.group(1))
        unit = match.group(2)

        map = {
            'Min': TimeFrameUnit.Minute,
            'Hour': TimeFrameUnit.Hour,
            'Day': TimeFrameUnit.Day,
            'Week': TimeFrameUnit.Week,
            'Month': TimeFrameUnit.Month}

        return TimeFrame(amount, map[unit])
    
    else:
        raise ValueError(
            "Invalid timeframe. Valid examples: 59Min, 23Hour, 1Day, 1Week, 12Month")



def tabular_print(
    entries: List, style='banner',
    align='left', width = 15, header = False) -> None:

    """
    Prints a list of entries in a tabular format.

    Args:
        entries (List): The list of entries to be printed in a tabular format.
        style (str, optional): The style of the table border. Defaults to 'banner'.
        align (str, optional): The alignment of the text in the table cells. Defaults to 'left'.
        width (int, optional): The width of each cell in the table. Defaults to 15.
        header (bool, optional): Whether the current row should be formatted as a header row. Defaults to False.

    Returns:
        None
    """

    # helper method to tabulate performance metrics.
    if header:
        row = tableprint.header(
            entries, style=style, align=align, width=width)

    else:
        row = tableprint.row(
            entries, style=style, align=align, width=width)

    return row



def progress_bar(total: Iterable):

    """
    Creates a progress bar using the tqdm library.
    Args:
        total (Iterable): The total number of iterations for the progress bar.

    Returns:
        tqdm: The progress bar object.
    """

    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed}<{remaining}'
    bar = tqdm(total = total, bar_format = bar_format)
    return bar



def get_sharpe_ratio(net_worth_hist: List[float], base=0):

    """
    Calculates the Sharpe ratio of a given net worth history list.

    Args:
        net_worth_hist (List[float]): A list of net worth values.
        base (int, optional): The risk-free rate. Defaults to 0.

    Returns:
        float: The calculated Sharpe ratio value.
    """

    hist = pd.Series(net_worth_hist)
    returns = hist.pct_change().dropna()
    val = (returns.mean()-base)/returns.std()

    return val



def objects_to_df(
    object_collection: Iterable[Dict[str, str]]
    ) -> pd.DataFrame:

    """
    Converts a collection of enum objects into a pandas DataFrame.

    Args:
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
