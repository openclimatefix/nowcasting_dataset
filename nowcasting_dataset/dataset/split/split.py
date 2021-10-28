""" Function to split datasets up """

import logging
from collections import namedtuple
from enum import Enum
from typing import List, Optional, Tuple, Union

import pandas as pd

from nowcasting_dataset.dataset.split.method import split_by_dates, split_method
from nowcasting_dataset.dataset.split.model import (
    TrainValidationTestSpecific,
    default_train_test_validation_specific,
)

logger = logging.getLogger(__name__)


class SplitMethod(Enum):
    """Different split methods"""

    DATE = "date"
    DAY = "day"
    DAY_RANDOM = "day_random"
    DAY_SPECIFIC = "day_specific"
    WEEK = "week"
    WEEK_RANDOM = "week_random"
    YEAR_SPECIFIC = "year_specific"
    SAME = "same"
    DAY_RANDOM_TEST_YEAR = "day_random_test_year"
    DAY_RANDOM_TEST_DATE = "day_random_test_date"


class SplitName(Enum):
    """The name for each data split."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


SplitData = namedtuple(
    typename="SplitData",
    field_names=[SplitName.TRAIN.value, SplitName.VALIDATION.value, SplitName.TEST.value],
)


def split_data(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    method: SplitMethod,
    train_test_validation_split: Tuple[int] = (3, 1, 1),
    train_test_validation_specific: TrainValidationTestSpecific = (
        default_train_test_validation_specific
    ),
    train_validation_test_datetime_split: Optional[List[pd.Timestamp]] = None,
    seed: int = 1234,
) -> SplitData:
    """
    Split the date using various different methods

    Args:
        datetimes: The datetimes to be split
        method: the method to be used
        train_test_validation_split: ratios of how the split is made
        seed: random seed used to permutate the data for the 'random' method
        train_test_validation_specific: pydandic class of 'train', 'validation' and 'test'.
            These specify which data goes into which dataset.
        train_validation_test_datetime_split: split train, validation based on specific dates.

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
    elif method == SplitMethod.DAY_SPECIFIC:
        train_datetimes, validation_datetimes, test_datetimes = split_method(
            datetimes=datetimes,
            train_test_validation_split=train_test_validation_split,
            train_test_validation_specific=train_test_validation_specific,
            method="specific",
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

    elif method == SplitMethod.DATE:
        train_datetimes, validation_datetimes, test_datetimes = split_by_dates(
            datetimes=datetimes,
            train_validation_datetime_split=train_validation_test_datetime_split[0],
            validation_test_datetime_split=train_validation_test_datetime_split[1],
        )

    elif method in [SplitMethod.DAY_RANDOM_TEST_YEAR, SplitMethod.DAY_RANDOM_TEST_DATE]:
        if method == SplitMethod.DAY_RANDOM_TEST_YEAR:
            # This method splits
            # 1. test set to be in one year, using 'train_test_validation_specific'
            # 2. train and validation by random day, using 'train_test_validation_split' on ratio how to split it
            #
            # This allows us to create a test set for 2021, and train and validation for random days not in 2021

            # create test set
            train_datetimes, validation_datetimes, test_datetimes = split_method(
                datetimes=datetimes,
                train_test_validation_split=train_test_validation_split,
                method="specific",
                freq="Y",
                seed=seed,
                train_test_validation_specific=train_test_validation_specific,
            )
        elif method == SplitMethod.DAY_RANDOM_TEST_DATE:
            # This method splits
            # 1. test set from one date onwards
            # 2. train and validation by random day, using 'train_test_validation_split' on ratio how to split it
            #
            # This allows us to create a test set from a specfic date e.g. 2020-07-01, and train and validation
            # for random days before that date

            # create test set
            train_datetimes, validation_datetimes, test_datetimes = split_by_dates(
                datetimes=datetimes,
                train_validation_datetime_split=train_validation_test_datetime_split[0],
                validation_test_datetime_split=train_validation_test_datetime_split[1],
            )

        # join train and validation together, so they can then be split by random day.
        train_and_validation_datetimes = train_datetimes.append(validation_datetimes)

        # set split ratio to only be on train and validation
        train_validation_split = list(train_test_validation_split)
        train_validation_split[2] = 0
        train_validation_split = tuple(train_validation_split)

        # get train and validation methods
        train_datetimes, validation_datetimes, _ = split_method(
            datetimes=train_and_validation_datetimes,
            train_test_validation_split=train_validation_split,
            method="random",
            seed=seed,
        )

    else:
        raise ValueError(f"{method} for splitting day is not implemented")

    logger.debug(
        f"Split data done, train has {len(train_datetimes):,d}, "
        f"validation has {len(validation_datetimes):,d}, "
        f"test has {len(test_datetimes):,d} t0 datetimes."
    )

    return SplitData(train=train_datetimes, validation=validation_datetimes, test=test_datetimes)
