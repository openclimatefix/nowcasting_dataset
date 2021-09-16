""" Function to split datasets up """

import pandas as pd
from typing import List, Tuple, Union

import logging

logger = logging.getLogger(__name__)


def split(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    method: str,
    train_test_validation_split: Tuple[int] = (3, 1, 1),
):
    """
    Split the date using various different methods

    Args:
        datetimes: The datetimes to be split
        method: the method to be used
        train_test_validation_split:

    Returns:

    """

    logger.info(f"Splitting data with method {method}")

    if method == "same":
        train_datetimes = datetimes
        validation_datetimes = datetimes
        test_datetimes = datetimes
    else:
        raise

    return train_datetimes, validation_datetimes, test_datetimes
