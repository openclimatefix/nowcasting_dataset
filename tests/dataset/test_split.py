from nowcasting_dataset.dataset.split import split
import pandas as pd


def test_split_same():

    datetimes = pd.date_range("2021-01-01", "2021-01-02", freq="5T")

    train, validation, test = split(datetimes=datetimes, method="same")

    assert (train == datetimes).all()
    assert (validation == datetimes).all()
    assert (test == datetimes).all()
