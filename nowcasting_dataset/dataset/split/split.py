""" Function to split datasets up """

import pandas as pd
from typing import List, Tuple, Union

import logging

from nowcasting_dataset.dataset.split.day import split_day, split_day_random

logger = logging.getLogger(__name__)


def split(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    method: str,
    train_test_validation_split: Tuple[int] = (3, 1, 1),
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the date using various different methods

    Args:
        datetimes: The datetimes to be split
        method: the method to be used
        train_test_validation_split:

    Returns: train, validation and test dataset

    """

    logger.info(f"Splitting data with method {method}")

    if method == "same":
        train_datetimes = datetimes
        validation_datetimes = datetimes
        test_datetimes = datetimes
    elif method == "day":
        train_datetimes, validation_datetimes, test_datetimes = split_day(
            datetimes=datetimes, train_test_validation_split=train_test_validation_split
        )
    elif method == "day_random":
        train_datetimes, validation_datetimes, test_datetimes = split_day_random(
            datetimes=datetimes, train_test_validation_split=train_test_validation_split
        )
    else:
        raise

    return train_datetimes, validation_datetimes, test_datetimes
