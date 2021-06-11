from nowcasting_dataset.data_sources import SatelliteDataSource
from nowcasting_dataset import consts
from nowcasting_dataset import Square
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

IMAGE_SIZE_PIXELS = 128


@pytest.fixture
def sat_data_source(use_cloud_data: bool):
    square = Square(size_pixels=IMAGE_SIZE_PIXELS, meters_per_pixel=2000)
    if use_cloud_data:
        filename = consts.SAT_DATA_ZARR
    else:
        filename = Path(__file__).parent.absolute() / 'data' / 'sat_data.zarr'
    print('\nLoading', filename)
    return SatelliteDataSource(
        image_size=square, filename=filename)


def test_satellite_data_source_init(sat_data_source):
    pass


def test_open(sat_data_source):
    sat_data_source.open()
    assert sat_data_source.sat_data is not None


def test_available_timestamps(sat_data_source):
    timestamps = sat_data_source.available_timestamps()
    assert isinstance(timestamps, pd.DatetimeIndex)
    assert len(timestamps) > 0
    assert len(np.unique(timestamps)) == len(timestamps)
    assert np.all(np.diff(timestamps.astype(int)) > 0)


def test_get_sample(sat_data_source):
    sat_data_source.open()
    sample = sat_data_source.get_sample(
        start=pd.Timestamp('2019-01-01T13:00'),
        end=pd.Timestamp('2019-01-01T14:00'),
        x_meters=0,
        y_meters=0)
    sat_data = sample['sat_data']
    assert len(sat_data.x) == IMAGE_SIZE_PIXELS
    assert len(sat_data.y) == IMAGE_SIZE_PIXELS
