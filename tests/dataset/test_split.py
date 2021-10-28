import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset.dataset.split.model import TrainValidationTestSpecific
from nowcasting_dataset.dataset.split.split import SplitMethod, split_data


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

    train_validation_overlap = train_df.join(validation_df, how="inner")
    train_test_overlap = train_df.join(test_df, how="inner")
    validation_test_overlap = validation_df.join(test_df, how="inner")

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

    train, validation, test = split_data(datetimes=datetimes, method=SplitMethod.YEAR_SPECIFIC)

    assert len(train) == 12 * 5  # 2015, 2016, 2017, 2018, 2019 months
    assert len(validation) == 12  # months in 2020
    assert len(test) == 1  # january 2021

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = train_df.join(validation_df, how="inner")
    train_test_overlap = train_df.join(test_df, how="inner")
    validation_test_overlap = validation_df.join(test_df, how="inner")

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check all first 288 datetimes are in the same day
    year = train[0].year
    for t in train[0:12]:
        assert t.year == year


def test_split_day_specific():

    datetimes = pd.date_range("2021-01-01", "2021-01-10", freq="D")

    train_test_validation_specific = TrainValidationTestSpecific(
        train=["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
        validation=["2021-01-05", "2021-01-06", "2021-01-07"],
        test=["2021-01-08", "2021-01-09", "2021-01-10"],
    )

    train, validation, test = split_data(
        datetimes=datetimes,
        method=SplitMethod.DAY_SPECIFIC,
        train_test_validation_specific=train_test_validation_specific,
    )

    assert len(train) == 4
    assert len(validation) == 3
    assert len(test) == 3

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = train_df.join(validation_df, how="inner")
    train_test_overlap = train_df.join(test_df, how="inner")
    validation_test_overlap = validation_df.join(test_df, how="inner")

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0


def test_split_year_error():

    with pytest.raises(Exception):
        TrainValidationTestSpecific(train=[2015, 2016], validation=[2016], test=[2017])

    with pytest.raises(Exception):
        TrainValidationTestSpecific(train=[2015], validation=[2016], test=[2016, 2017])

    with pytest.raises(Exception):
        TrainValidationTestSpecific(train=[2015], validation=[2016], test=[2015, 2017])


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

    train_validation_overlap = train_df.join(validation_df, how="inner")
    train_test_overlap = train_df.join(test_df, how="inner")
    validation_test_overlap = validation_df.join(test_df, how="inner")

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check all first 3 days datetimes are in the same week
    # train is a pd.DatetimeIndex object.
    week = train[0].week
    for t in train[0:3]:
        assert t.week == week


def test_split_random_day_test_specific():

    datetimes = pd.date_range("2020-01-01", "2022-01-01", freq="1D")

    train, validation, test = split_data(
        datetimes=datetimes, method=SplitMethod.DAY_RANDOM_TEST_YEAR
    )

    assert len(train) == 274  # 75% of days of 2020
    assert len(validation) == 92  # 25% of days of 2020
    assert len(test) == 365  # % of days in 2021

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = train_df.join(validation_df, how="inner")
    train_test_overlap = train_df.join(test_df, how="inner")
    validation_test_overlap = validation_df.join(test_df, how="inner")

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check all train and validation are in 2020
    assert (train_df.year == 2020).sum() == len(train_df)
    assert (validation_df.year == 2020).sum() == len(validation_df)
    assert (test.year == 2021).sum() == len(test)


def test_split_date():

    datetimes = pd.date_range("2020-01-01", "2022-01-01", freq="1D")
    train_validation_test_datetime_split = [pd.Timestamp("2020-07-01"), pd.Timestamp("2021-01-01")]

    train, validation, test = split_data(
        datetimes=datetimes,
        method=SplitMethod.DATE,
        train_validation_test_datetime_split=train_validation_test_datetime_split,
    )

    assert len(train) == 182  # first half of 2020
    assert len(validation) == 184  # second half of 2020
    assert len(test) == 366  # all of days in 2021

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = train_df.join(validation_df, how="inner")
    train_test_overlap = train_df.join(test_df, how="inner")
    validation_test_overlap = validation_df.join(test_df, how="inner")

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check datetimes are in the correct sections
    assert (train_df < pd.Timestamp("2020-07-01")).sum() == len(train_df)
    assert (
        (validation_df >= pd.Timestamp("2020-07-01")) & (validation_df < pd.Timestamp("2021-01-01"))
    ).sum() == len(validation_df)
    assert (test >= pd.Timestamp("2021-01-01")).sum() == len(test)


def test_split_day_random_test_date():

    datetimes = pd.date_range("2020-01-01", "2022-01-01", freq="1D")
    train_validation_test_datetime_split = [pd.Timestamp("2020-07-01"), pd.Timestamp("2021-07-01")]

    train, validation, test = split_data(
        datetimes=datetimes,
        method=SplitMethod.DAY_RANDOM_TEST_DATE,
        train_validation_test_datetime_split=train_validation_test_datetime_split,
    )

    assert len(train) == 410  # 75% of days of 2020 and half of 2021 (~365*1.5*0.75)
    assert len(validation) == 137  # 25% of days of 2020 and half of 2021 (~365*1.5*0.25)
    assert len(test) == 185  # and second half of 2021 of days in 2021

    train_df = pd.DatetimeIndex(train)
    validation_df = pd.DatetimeIndex(validation)
    test_df = pd.DatetimeIndex(test)

    train_validation_overlap = train_df.join(validation_df, how="inner")
    train_test_overlap = train_df.join(test_df, how="inner")
    validation_test_overlap = validation_df.join(test_df, how="inner")

    assert len(train_validation_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(validation_test_overlap) == 0

    # check datetimes are in the correct sections
    assert (train_df < pd.Timestamp("2021-07-01")).sum() == len(train_df)
    assert (validation_df < pd.Timestamp("2021-07-01")).sum() == len(validation_df)
    assert (test >= pd.Timestamp("2021-07-01")).sum() == len(test)
