from typing import Union, List, Tuple

import random
import numpy as np
import pandas as pd


def split_day(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    train_test_validation_split: Tuple[int] = (3, 1, 1),
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the data by the day.

    If the split is (3,1,1) then
    train data will have all days of the year that are module 5 remainder 0,1 or 2
    validation data will have all days of the year that are module 5 remainder 3
    test data will have all days of the year that are module 5 remainder 4

    Args:
        datetimes: list of datetimes
        train_test_validation_split: how the split is made

    Returns: train, validation and test datetimes

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


def split_day_random(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    train_test_validation_split: Tuple[int] = (3, 1, 1),
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the data by the day. Take random days

    If the split is (3,1,1) then
    train data will have 60% of the days
    validation data will have 20% of the days
    test data will have have 20% of the days

    Args:
        datetimes: list of datetimes
        train_test_validation_split: how the split is made

    Returns: train, validation and test datetimes

    """

    total_weights = sum(train_test_validation_split)
    cum_weights = np.cumsum(train_test_validation_split)

    if type(datetimes) is pd.DatetimeIndex:
        datetimes = datetimes.to_list()

    # create list of days, that are in the dataset
    days_in_dataset = pd.DatetimeIndex(datetimes).dayofyear.unique().to_list()

    # randomly sort indexes
    random.shuffle(days_in_dataset)

    # make which day indexes go i.e if the split is [3,1,1] then the
    # - train_indexes = [0,1,2 ... 220,]
    # - validation_indexes = [221, 222, ... 293]
    # - test_indexes = [294, ...., 366]
    train_validation_split = int(cum_weights[0] / total_weights * 365)
    validation_test_split = int(cum_weights[1] / total_weights * 365)

    train_indexes = days_in_dataset[0:train_validation_split]
    validation_indexes = days_in_dataset[train_validation_split:validation_test_split]
    test_indexes = days_in_dataset[validation_test_split:]

    # select the datetimes
    train = [datetime for datetime in datetimes if datetime.dayofyear in train_indexes]
    validation = [datetime for datetime in datetimes if datetime.dayofyear in validation_indexes]
    test = [datetime for datetime in datetimes if datetime.dayofyear in test_indexes]

    return train, validation, test
