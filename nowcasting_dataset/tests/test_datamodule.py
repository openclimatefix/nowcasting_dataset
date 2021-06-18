from pathlib import Path
from nowcasting_dataset import datamodule
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def nowcasting_datamodule(sat_filename: Path):
    return datamodule.NowcastingDataModule(sat_filename=sat_filename)


def test_prepare_data(nowcasting_datamodule: datamodule.NowcastingDataModule):
    nowcasting_datamodule.prepare_data()


def test_get_daylight_datetime_index(
        nowcasting_datamodule: datamodule.NowcastingDataModule,
        use_cloud_data: bool):
    # Check it throws RuntimeError if we try running
    # _get_daylight_datetime_index() before running prepare_data():
    with pytest.raises(RuntimeError):
        nowcasting_datamodule._get_datetimes()
    nowcasting_datamodule.prepare_data()
    datetimes = nowcasting_datamodule._get_datetimes()
    assert isinstance(datetimes, pd.DatetimeIndex)
    if not use_cloud_data:
        correct_datetimes = pd.date_range(
            '2019-01-01 12:05', '2019-01-01 16:20', freq='5 min')
        np.testing.assert_array_equal(datetimes, correct_datetimes)


def test_setup(
        nowcasting_datamodule: datamodule.NowcastingDataModule):
    # Check it throws RuntimeError if we try running
    # setup() before running prepare_data():
    with pytest.raises(RuntimeError):
        nowcasting_datamodule.setup()
    nowcasting_datamodule.prepare_data()
    nowcasting_datamodule.setup()
