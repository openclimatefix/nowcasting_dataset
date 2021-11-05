""" Methods for splitting data into train, validation and test """
from typing import List, Tuple

import numpy as np
import pandas as pd

from nowcasting_dataset.dataset.split.model import (
    TrainValidationTestSpecific,
    default_train_test_validation_specific,
)


def split_method(
    datetimes: pd.DatetimeIndex,
    train_test_validation_split: Tuple[int] = (3, 1, 1),
    train_test_validation_specific: TrainValidationTestSpecific = (
        default_train_test_validation_specific
    ),
    method: str = "modulo",
    freq: str = "D",
    seed: int = 1234,
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the data into train, test and (optionally) validation sets.

    method: modulo
    If the split is (3,1,1) then, taking all the days in the dataset:
    - train data will have all days that are modulo 5 remainder 0,1 or 2,
    i.e 1st, 2nd, 3rd, 6th, 7th, 8th, 11th .... of the whole dataset
    - validation data will have all days that are modulo 5 remainder 3, i.e 4th, 9th, ...
    - test data will have all days that are modulo 5 remainder 4, i.e 5th, 10th ,  ...

    method: random
    If the split is (3,1,1) then
    - train data will have 60% of the data
    - validation data will have 20% of the data
    - test data will have have 20% of the data

    Args:
        datetimes: list of datetimes
        train_test_validation_split: how the split is made
        method: which method to use. Can be modulo or random
        freq: This can be D=day, W=week, M=month and Y=year. This means the data is divided up by
            different periods
        seed: random seed used to permutate the data for the 'random' method
        train_test_validation_specific: pydandic class of 'train', 'validation' and 'test'.
            These specify which data goes into which datasets

    Returns: train, validation and test datetimes

    """
    # find all the unique periods (dates, weeks, e.t.c)
    datetimes_period = pd.to_datetime(datetimes.to_period(freq).to_timestamp())
    unique_periods_in_dataset = datetimes_period.unique()

    # find total weights, and cumulative weights
    total_weights = sum(train_test_validation_split)
    cum_weights = np.cumsum(train_test_validation_split)

    if method == "modulo":
        # Method to split by module.
        # I.e 1st, 2nd, 3rd periods goes to train, 4th goes to validation, 5th goes to test and
        # repeat.

        # make which day indexes go i.e if the split is [3,1,1] then the
        # - train_ indexes = [0,1,2]
        # - validation_indexes = [3]
        # - test_indexes = [4]
        train_indexes = [i for i in range(total_weights) if i < cum_weights[0]]
        validation_indexes = [
            i for i in range(total_weights) if (i >= cum_weights[0]) & (i < cum_weights[1])
        ]
        test_indexes = [
            i for i in range(total_weights) if (i >= cum_weights[1]) & (i < cum_weights[2])
        ]

        # find all the unique periods (dates, weeks, e.t.c)
        unique_periods = pd.DataFrame(unique_periods_in_dataset, columns=["period"])
        unique_periods["modulo"] = unique_periods.index % total_weights

        train_periods = unique_periods[unique_periods["modulo"].isin(train_indexes)]["period"]
        validation_periods = unique_periods[unique_periods["modulo"].isin(validation_indexes)][
            "period"
        ]
        test_periods = unique_periods[unique_periods["modulo"].isin(test_indexes)]["period"]

    elif method == "random":

        # randomly sort indexes
        rng = np.random.default_rng(seed)
        unique_periods_in_dataset = rng.permutation(unique_periods_in_dataset)

        # find the train, validation, test indexes.
        #
        # For one year of data, for days, if the split is [3,1,1] for one year then the
        # - train_indexes - first 220 random dates
        # - validation_indexes - next 72 random dates
        # - test_indexes - next 72 random dates
        train_validation_split = int(
            cum_weights[0] / total_weights * len(unique_periods_in_dataset)
        )
        validation_test_split = int(cum_weights[1] / total_weights * len(unique_periods_in_dataset))

        train_periods = pd.to_datetime(unique_periods_in_dataset[0:train_validation_split])
        validation_periods = pd.to_datetime(
            unique_periods_in_dataset[train_validation_split:validation_test_split]
        )
        test_periods = pd.to_datetime(unique_periods_in_dataset[validation_test_split:])

    elif method == "specific":

        train_periods = unique_periods_in_dataset[
            unique_periods_in_dataset.isin(train_test_validation_specific.train)
        ]
        validation_periods = unique_periods_in_dataset[
            unique_periods_in_dataset.isin(train_test_validation_specific.validation)
        ]
        test_periods = unique_periods_in_dataset[
            unique_periods_in_dataset.isin(train_test_validation_specific.test)
        ]

    else:
        raise Exception(f'method ({method}) must be in ["random", "modulo"]')

    train = datetimes[datetimes_period.isin(train_periods)]
    validation = datetimes[datetimes_period.isin(validation_periods)]
    test = datetimes[datetimes_period.isin(test_periods)]

    return train, validation, test


def split_by_dates(
    datetimes: pd.DatetimeIndex,
    train_validation_datetime_split: pd.Timestamp,
    validation_test_datetime_split: pd.Timestamp,
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split datetimes into train, validation and test by two specific datetime splits

    Note that the 'train_validation_datetime_split' should be less than the
    'validation_test_datetime_split'

    Args:
        datetimes: list of datetimes
        train_validation_datetime_split: the datetime which will split the train and validation
            datetimes.
        For example if this is '2021-01-01' then the train datetimes will end by '2021-01-01' and
            the validation datetimes will start at '2021-01-01'.
        validation_test_datetime_split: the datetime which will split the validation and
            test datetimes

    Returns: train, validation and test datetimes

    """
    assert train_validation_datetime_split <= validation_test_datetime_split

    train = datetimes[datetimes < train_validation_datetime_split]
    validation = datetimes[
        (datetimes >= train_validation_datetime_split)
        & (datetimes < validation_test_datetime_split)
    ]
    test = datetimes[datetimes >= validation_test_datetime_split]

    return train, validation, test
