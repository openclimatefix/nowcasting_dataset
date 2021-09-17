from typing import Union, List, Tuple

import random
import numpy as np
import pandas as pd


def split_week(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    train_test_validation_split: Tuple[int] = (3, 1, 1),
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the data by the week.

    If the split is (3,1,1) then
    train data will have all weeks of the year that are module 5 remainder 0,1 or 2
    validation data will have all weeks of the year that are module 5 remainder 3
    test data will have all weeks of the year that are module 5 remainder 4

    Args:
        datetimes: list of datetimes
        train_test_validation_split: how the split is made

    Returns: train, validation and test datetimes

    """

    total_weights = sum(train_test_validation_split)
    cum_weights = np.cumsum(train_test_validation_split)

    # make which week indexes go i.e if the split is [3,1,1] then the
    # - train_indexes = [0,1,2]
    # - validation_indexes = [3]
    # - test_indexes = [4]
    train_indexes = [i for i in range(total_weights) if i < cum_weights[0]]
    validation_indexes = [
        i for i in range(total_weights) if (i >= cum_weights[0]) & (i < cum_weights[1])
    ]
    test_indexes = [i for i in range(total_weights) if (i >= cum_weights[1]) & (i < cum_weights[2])]

    # select the datetimes
    train = [
        datetime for datetime in datetimes if datetime.weekofyear % total_weights in train_indexes
    ]
    validation = [
        datetime
        for datetime in datetimes
        if datetime.weekofyear % total_weights in validation_indexes
    ]
    test = [
        datetime for datetime in datetimes if datetime.weekofyear % total_weights in test_indexes
    ]

    return train, validation, test


def split_week_random(
    datetimes: Union[List[pd.Timestamp], pd.DatetimeIndex],
    train_test_validation_split: Tuple[int] = (3, 1, 1),
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the data by the week. Take random weeks

    If the split is (3,1,1) then
    train data will have 60% of the weeks
    validation data will have 20% of the weeks
    test data will have have 20% of the weeks

    Args:
        datetimes: list of datetimes
        train_test_validation_split: how the split is made

    Returns: train, validation and test datetimes

    """

    total_weights = sum(train_test_validation_split)
    cum_weights = np.cumsum(train_test_validation_split)

    if type(datetimes) is pd.DatetimeIndex:
        datetimes = datetimes.to_list()

    # get weeks of the year that are in the dataset
    weeks_in_dataset = pd.DatetimeIndex(datetimes).isocalendar().week.unique()
    maximum_week = max(weeks_in_dataset)

    # randomly sort indexes
    random.shuffle(weeks_in_dataset)

    # make which week indexes go i.e if the split is [3,1,1] then the
    # - train_indexes = [0,1,2 ... 31,]
    # - validation_indexes = [32, 33, ... 42]
    # - test_indexes = [43, ...., 52]
    train_validation_split = int(cum_weights[0] / total_weights * maximum_week)
    validation_test_split = int(cum_weights[1] / total_weights * maximum_week)

    train_indexes = weeks_in_dataset[0:train_validation_split]
    validation_indexes = weeks_in_dataset[train_validation_split:validation_test_split]
    test_indexes = weeks_in_dataset[validation_test_split:]

    # select the datetimes
    train = [datetime for datetime in datetimes if datetime.isocalendar().week in train_indexes]

    validation = [
        datetime for datetime in datetimes if datetime.isocalendar().week in validation_indexes
    ]
    test = [datetime for datetime in datetimes if datetime.isocalendar().week in test_indexes]

    return train, validation, test
