from nowcasting_dataset.data_sources import SatelliteDataSource
from nowcasting_dataset import consts
from nowcasting_dataset import Square
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

IMAGE_SIZE_PIXELS = 128


@pytest.fixture
def sat_data_source(sat_filename: Path):
    square = Square(size_pixels=IMAGE_SIZE_PIXELS, meters_per_pixel=2000)
    return SatelliteDataSource(image_size=square, filename=sat_filename)


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


@pytest.mark.parametrize(
    "x, y, left, right, top, bottom",
    [
        (0, 0, -128_000, 126_000, 128_000, -126_000),
        (10, 0, -126_000, 128_000, 128_000, -126_000),
        (30, 0, -126_000, 128_000, 128_000, -126_000),
        (1000, 0, -126_000, 128_000, 128_000, -126_000),
        (0, 1000, -128_000, 126_000, 128_000, -126_000),
        (1000, 1000, -126_000, 128_000, 128_000, -126_000),
        (2000, 2000, -126_000, 128_000, 130_000, -124_000),
        (2000, 1000, -126_000, 128_000, 128_000, -126_000),
        (2001, 2001, -124_000, 130_000, 130_000, -124_000),
    ]
)
def test_get_sample(sat_data_source, x, y, left, right, top, bottom):
    sat_data_source.open()
    start = pd.Timestamp('2019-01-01T13:00')
    end = pd.Timestamp('2019-01-01T14:00')
    sample = sat_data_source.get_sample(
        start=start, end=end, x_meters=x, y_meters=y)
    sat_data = sample['sat_data']
    assert left == sat_data.x.values[0]
    assert right == sat_data.x.values[-1]
    # sat_data.y is top-to-bottom.
    assert top == sat_data.y.values[0]
    assert bottom == sat_data.y.values[-1]
    assert len(sat_data.x) == IMAGE_SIZE_PIXELS
    assert len(sat_data.y) == IMAGE_SIZE_PIXELS


def test_geospatial_border(sat_data_source):
    border = sat_data_source.geospatial_border()
    correct_border = [
        (-110000, 1094000),
        (-110000, -58000),
        (730000, 1094000),
        (730000, -58000)]
    np.testing.assert_array_equal(border, correct_border)
