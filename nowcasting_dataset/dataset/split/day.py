from typing import Union, List, Tuple

import random
import numpy as np
import pandas as pd


def split_day(
    datetimes: pd.DatetimeIndex,
    train_test_validation_split: Tuple[int] = (3, 1, 1),
    method: str = "modulo",
) -> (List[pd.Timestamp], List[pd.Timestamp], List[pd.Timestamp]):
    """
    Split the data by the day into train, test and validation.

    method: modulo
    If the split is (3,1,1) then, taking all the days in the dataset:
    - train data will have all days that are modulo 5 remainder 0,1 or 2,
    i.e 1st, 2nd, 3rd, 6th, 7th, 8th, 11th .... of the whole dataset
    - validation data will have all days that are modulo 5 remainder 3, i.e 4th, 9th, ...
    - test data will have all days that are modulo 5 remainder 4, i.e 5th, 10th ,  ...

    method: random
    If the split is (3,1,1) then
    - train data will have 60% of the days
    - validation data will have 20% of the days
    - test data will have have 20% of the days

    Args:
        datetimes: list of datetimes
        train_test_validation_split: how the split is made
        method: which method to use. Can be modulo or random

    Returns: train, validation and test datetimes

    """

    # find all the unique dates
    datetimes_date = pd.to_datetime(datetimes.date)
    days_in_dataset = datetimes_date.unique()

    total_weights = sum(train_test_validation_split)
    cum_weights = np.cumsum(train_test_validation_split)

    if method == "modulo":
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

        # find all the unique dates
        unique_dates = pd.DataFrame(days_in_dataset, columns=["date"])
        unique_dates["modulo"] = unique_dates.index % total_weights

        train_dates = unique_dates[unique_dates["modulo"].isin(train_indexes)]["date"]
        validation_dates = unique_dates[unique_dates["modulo"].isin(validation_indexes)]["date"]
        test_dates = unique_dates[unique_dates["modulo"].isin(test_indexes)]["date"]

    elif method == "random":

        # randomly sort indexes
        days_in_dataset = np.random.permutation(days_in_dataset)

        # make which day indexes go i.e if the split is [3,1,1] for one year then the
        # - train_indexes - first 220 random dates
        # - validation_indexes - next 72 random dates
        # - test_indexes - next 72 random dates
        train_validation_split = int(cum_weights[0] / total_weights * 365)
        validation_test_split = int(cum_weights[1] / total_weights * 365)

        train_dates = pd.to_datetime(days_in_dataset[0:train_validation_split])
        validation_dates = pd.to_datetime(
            days_in_dataset[train_validation_split:validation_test_split]
        )
        test_dates = pd.to_datetime(days_in_dataset[validation_test_split:])
    else:
        raise Exception(f'method ({method}) must be in ["random", "modulo"]')

    train = datetimes[datetimes_date.isin(train_dates)]
    validation = datetimes[datetimes_date.isin(validation_dates)]
    test = datetimes[datetimes_date.isin(test_dates)]

    return train, validation, test


def split_day_random(
    datetimes: pd.DatetimeIndex,
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

    # make dates of the all the datetimes
    datetimes_date = pd.to_datetime(datetimes.date)

    # calculate total weights and cumulative weights, useful for splitting
    total_weights = sum(train_test_validation_split)
    cum_weights = np.cumsum(train_test_validation_split)

    # find all the unique dates
    days_in_dataset = datetimes_date.unique()

    # randomly sort indexes
    days_in_dataset = np.random.permutation(days_in_dataset)

    # make which day indexes go i.e if the split is [3,1,1] for one year then the
    # - train_indexes - first 220 random dates
    # - validation_indexes - next 72 random dates
    # - test_indexes - next 72 random dates
    train_validation_split = int(cum_weights[0] / total_weights * 365)
    validation_test_split = int(cum_weights[1] / total_weights * 365)

    train_dates = pd.to_datetime(days_in_dataset[0:train_validation_split])
    validation_dates = pd.to_datetime(days_in_dataset[train_validation_split:validation_test_split])
    test_dates = pd.to_datetime(days_in_dataset[validation_test_split:])

    # select the datetimes
    train = datetimes[datetimes_date.isin(train_dates)]
    validation = datetimes[datetimes_date.isin(validation_dates)]
    test = datetimes[datetimes_date.isin(test_dates)]

    return train, validation, test
