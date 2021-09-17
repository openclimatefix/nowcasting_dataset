import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset.dataset.split.split import split_data, SplitMethod
from nowcasting_dataset.dataset.split.year import TrainValidationTestYear


def test_split_same():

    datetimes = pd.date_range("2021-01-01", "2021-01-02", freq="5T")

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.SAME)

    assert (train == datetimes).all()
    assert (validation == datetimes).all()
    assert (test == datetimes).all()


def test_split_day():

    datetimes = pd.date_range("2021-01-01", "2021-02-01", freq="5T")

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.DAY)

    unique_dates = pd.Series(np.unique(datetimes.date))

    train_dates = unique_dates[unique_dates.isin(np.unique(train.date))]
    assert (np.unique(train_dates.index % 5) == [0, 1, 2]).all()

    validation_dates = unique_dates[unique_dates.isin(np.unique(validation.date))]
    assert (np.unique(validation_dates.index % 5) == [3]).all()

    test_dates = unique_dates[unique_dates.isin(np.unique(test.date))]
    assert (np.unique(test_dates.index % 5) == [4]).all()

    # check all first 288 datetimes are in the same day
    day = train[0].dayofyear
    for t in train[0 : 12 * 24]:
        assert t.dayofyear == day
    assert train[12 * 24 + 1] != day


def test_split_day_every_5():

    datetimes = pd.date_range("2021-01-01", "2021-01-02", freq="5T")
    datetimes = datetimes.append(pd.date_range("2021-01-06", "2021-01-07", freq="5T"))
    datetimes = datetimes.append(pd.date_range("2021-01-11", "2021-01-12", freq="5T"))
    datetimes = datetimes.append(pd.date_range("2021-01-16", "2021-01-17", freq="5T"))
    datetimes = datetimes.append(pd.date_range("2021-01-21", "2021-01-22", freq="5T"))

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.DAY)

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0

    unique_dates = pd.Series(np.unique(datetimes.date))

    train_dates = unique_dates[unique_dates.isin(np.unique(train.date))]
    assert (np.unique(train_dates.index % 5) == [0, 1, 2]).all()

    validation_dates = unique_dates[unique_dates.isin(np.unique(validation.date))]
    assert (np.unique(validation_dates.index % 5) == [3]).all()

    test_dates = unique_dates[unique_dates.isin(np.unique(test.date))]
    assert (np.unique(test_dates.index % 5) == [4]).all()

    # check all first 288 datetimes are in the same day
    day = train[0].dayofyear
    for t in train[0 : 12 * 24]:
        assert t.dayofyear == day
    assert train[12 * 24 + 1] != day


def test_split_day_random():

    datetimes = pd.date_range("2021-01-01", "2021-12-31 23:59:00", freq="5T")

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.DAY_RANDOM)

    assert len(train) == int((365 * 0.6)) * 24 * 12  # 60% of days
    assert len(validation) == int((365 * 0.2)) * 24 * 12  # 20% of days
    assert len(test) == int((365 * 0.2)) * 24 * 12  # 20% of days

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = [t for t in train_df if t in validation_df]
    train_test_overlap = [t for t in train_df if t in test_df]
    validation_test_overlap = [t for t in validation_df if t in test_df]

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check all first 288 datetimes are in the same day
    day = train[0].dayofyear
    for t in train[0 : 12 * 24]:
        assert t.dayofyear == day
    assert train[12 * 24 + 1] != day


def test_split_year():

    datetimes = pd.date_range("2014-01-01", "2021-01-01", freq="MS")

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.YEAR)

    assert len(train) == 12 * 5  # 2015, 2016, 2017, 2018, 2019 months
    assert len(validation) == 12  # months in 2020
    assert len(test) == 1  # january 2021

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = [t for t in train_df if t in validation_df]
    train_test_overlap = [t for t in train_df if t in test_df]
    validation_test_overlap = [t for t in validation_df if t in test_df]

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check all first 288 datetimes are in the same day
    year = train[0].year
    for t in train[0:12]:
        assert t.year == year


def test_split_year_error():

    with pytest.raises(Exception):
        TrainValidationTestYear(train=[2015, 2016], validation=[2016], test=[2017])

    with pytest.raises(Exception):
        TrainValidationTestYear(train=[2015], validation=[2016], test=[2016, 2017])

    with pytest.raises(Exception):
        TrainValidationTestYear(train=[2015], validation=[2016], test=[2015, 2017])


def test_split_week():

    datetimes = pd.date_range("2021-01-01", "2021-06-01", freq="30T")

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.WEEK)

    unique_weeks = pd.Series(datetimes.to_period("W").to_timestamp().unique())
    train_weeks = pd.to_datetime(train.to_period("W").to_timestamp())
    validation_weeks = pd.to_datetime(validation.to_period("W").to_timestamp())
    test_weeks = pd.to_datetime(test.to_period("W").to_timestamp())

    train_dates = unique_weeks[unique_weeks.isin(train_weeks)]
    assert (np.unique(train_dates.index % 5) == [0, 1, 2]).all()

    validation_dates = unique_weeks[unique_weeks.isin(validation_weeks)]
    assert (np.unique(validation_dates.index % 5) == [3]).all()

    test_dates = unique_weeks[unique_weeks.isin(test_weeks)]
    assert (np.unique(test_dates.index % 5) == [4]).all()

    # check all first day datetimes are in the same week
    week = train[0].week
    for t in train[0:48]:
        assert t.week == week


def test_split_week_random():

    datetimes = pd.date_range("2021-01-04", "2022-01-02", freq="1D")

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.WEEK_RANDOM)

    assert len(train) == 217  # 60% of weeks
    assert len(validation) == 70  # 20% of weeks
    assert len(test) == 77  # 20% of weeks (funny rounding due to taking int(52*0.8) - int(52*0.6))

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = [t for t in train_df if t in validation_df]
    train_test_overlap = [t for t in train_df if t in test_df]
    validation_test_overlap = [t for t in validation_df if t in test_df]

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check all first 3 days datetimes are in the same week
    week = train[0].isocalendar().week
    for t in train[0:3]:
        assert t.isocalendar().week == week
