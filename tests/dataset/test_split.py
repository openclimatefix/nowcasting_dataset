from nowcasting_dataset.dataset.split import split
import pandas as pd


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
