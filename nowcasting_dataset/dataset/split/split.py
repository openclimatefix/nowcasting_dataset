""" Function to split datasets up """

import logging
from enum import Enum
from typing import List, Tuple, Union

import pandas as pd

from nowcasting_dataset.dataset.split.method import split_method
from nowcasting_dataset.dataset.split.model import (
    TrainValidationTestSpecific,
    default_train_test_validation_specific,
)

logger = logging.getLogger(__name__)


class SplitMethod(Enum):
    DAY = "day"
    DAY_RANDOM = "day_random"
    WEEK = "week"
    WEEK_RANDOM = "week_random"
    YEAR_SPECIFIC = "year_specific"
    SAME = "same"


def split_data(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    method: SplitMethod,
    train_test_validation_split: Tuple[int] = (3, 1, 1),
    train_test_validation_specific: TrainValidationTestSpecific = default_train_test_validation_specific,
    seed: int = 1234,
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the date using various different methods

    Args:
        datetimes: The datetimes to be split
        method: the method to be used
        train_test_validation_split: ratios of how the split is made
        seed: random seed used to permutate the data for the 'random' method
        train_test_validation_specific: pydandic class of 'train', 'validation' and 'test'. These specifies
            which data goes into which datasets


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
            seed=seed,
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
            seed=seed,
        )
    elif method == SplitMethod.YEAR_SPECIFIC:
        train_datetimes, validation_datetimes, test_datetimes = split_method(
            datetimes=datetimes,
            train_test_validation_split=train_test_validation_split,
            method="specific",
            freq="Y",
            seed=seed,
            train_test_validation_specific=train_test_validation_specific,
        )
    else:
        raise ValueError(f"{method} for splitting day is not implemented")

    return train_datetimes, validation_datetimes, test_datetimes
