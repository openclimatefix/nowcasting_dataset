from nowcasting_dataset.dataset.split.split import split
from nowcasting_dataset.dataset.split.year import TrainValidationTestYear
import pandas as pd
import pytest


def test_split_same():

    datetimes = pd.date_range("2021-01-01", "2021-01-02", freq="5T")

    train, validation, test = split(datetimes=datetimes, method="same")

    assert (train == datetimes).all()
    assert (validation == datetimes).all()
    assert (test == datetimes).all()


def test_split_day():

    datetimes = pd.date_range("2021-01-01", "2021-02-01", freq="5T")

    train, validation, test = split(datetimes=datetimes, method="day")

    for d in train:
        assert d.dayofyear % 5 in [0, 1, 2]
    for d in validation:
        assert d.dayofyear % 5 in [3]
    for d in test:
        assert d.dayofyear % 5 in [4]

    # check all first 288 datetimes are in the same day
    day = train[0].dayofyear
    for t in train[0 : 12 * 24]:
        assert t.dayofyear == day
    assert train[12 * 24 + 1] != day


def test_split_day_random():

    datetimes = pd.date_range("2021-01-01", "2021-12-31 23:59:00", freq="5T")

    train, validation, test = split(datetimes=datetimes, method="day_random")

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

    train, validation, test = split(datetimes=datetimes, method="year")

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
