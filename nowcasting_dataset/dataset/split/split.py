""" Function to split datasets up """

import logging
from enum import Enum
from typing import List, Tuple, Union

import pandas as pd

from nowcasting_dataset.dataset.split.method import split_method
from nowcasting_dataset.dataset.split.year import (
    split_year,
    TrainValidationTestYear,
    default_train_test_validation_year,
)

logger = logging.getLogger(__name__)


class SplitMethod(Enum):
    DAY = "day"
    DAY_RANDOM = "day_random"
    WEEK = "week"
    WEEK_RANDOM = "week_random"
    YEAR = "year"
    SAME = "same"


def split_data(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    method: SplitMethod,
    train_test_validation_split: Tuple[int] = (3, 1, 1),
    train_test_validation_year: TrainValidationTestYear = default_train_test_validation_year,
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the date using various different methods

    Args:
        datetimes: The datetimes to be split
        method: the method to be used
        train_test_validation_split: ratios of how the split is made
        train_test_validation_year: pydantic class of which years below to which dataset

    Returns: train, validation and test dataset

    """

    logger.info(f"Splitting data with method {method}")

    datetimes = pd.DatetimeIndex(datetimes)

    if method == SplitMethod.SAME:
        train_datetimes = datetimes
        validation_datetimes = datetimes
        test_datetimes = datetimes
    elif method == SplitMethod.DAY:
        train_datetimes, validation_datetimes, test_datetimes = split_method(
            datetimes=datetimes,
            train_test_validation_split=train_test_validation_split,
            method="modulo",
        )
    elif method == SplitMethod.DAY_RANDOM:
        train_datetimes, validation_datetimes, test_datetimes = split_method(
            datetimes=datetimes,
            train_test_validation_split=train_test_validation_split,
            method="random",
        )
    elif method == SplitMethod.WEEK:
        train_datetimes, validation_datetimes, test_datetimes = split_method(
            datetimes=datetimes,
            train_test_validation_split=train_test_validation_split,
            method="modulo",
            freq="W",
        )
    elif method == SplitMethod.WEEK_RANDOM:
        train_datetimes, validation_datetimes, test_datetimes = split_method(
            datetimes=datetimes,
            train_test_validation_split=train_test_validation_split,
            method="random",
            freq="W",
        )
    elif method == SplitMethod.YEAR:
        train_datetimes, validation_datetimes, test_datetimes = split_year(
            datetimes=datetimes, train_test_validation_year=train_test_validation_year
        )
    else:
        raise ValueError(f"{method} for splitting day is not implemented")

    return train_datetimes, validation_datetimes, test_datetimes
