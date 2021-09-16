""" Function to split datasets up """

import pandas as pd
import numpy as np
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
    elif method == "day":
        train_datetimes, validation_datetimes, test_datetimes = _split_day(
            datetimes=datetimes, train_test_validation_split=train_test_validation_split
        )
    else:
        raise

    return train_datetimes, validation_datetimes, test_datetimes


def _split_day(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    train_test_validation_split: Tuple[int] = (3, 1, 1),
):
    """

    Args:
        datetimes:
        train_test_validation_split:

    Returns:

    """

    total_weights = sum(train_test_validation_split)
    cum_weights = np.cumsum(train_test_validation_split)

    # make which day indexes go i.e if the split is [3,1,1] then the
    # - train_ indexes = [0,1,2]
    # - validation_indexes = [3]
    # - test_indexes = [4]
    train_indexes = [i for i in range(total_weights) if i < cum_weights[0]]
    validation_indexes = [
        i for i in range(total_weights) if (i >= cum_weights[0]) & (i < cum_weights[1])
    ]
    test_indexes = [i for i in range(total_weights) if (i >= cum_weights[1]) & (i < cum_weights[2])]

    # select the datetimes
    train = [
        datetime for datetime in datetimes if datetime.dayofyear % total_weights in train_indexes
    ]
    validation = [
        datetime
        for datetime in datetimes
        if datetime.dayofyear % total_weights in validation_indexes
    ]
    test = [
        datetime for datetime in datetimes if datetime.dayofyear % total_weights in test_indexes
    ]

    return train, validation, test
