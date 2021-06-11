from nowcasting_dataset.data_sources import SatelliteDataSource
from nowcasting_dataset import Square
import pytest


@pytest.fixture
def sat_data_source():
    square = Square(size_pixels=128, meters_per_pixel=1000)
    return SatelliteDataSource(image_size=square)


def test_satellite_data_source_init(sat_data_source):
    pass


def test_open(sat_data_source):
    sat_data_source.open()
    assert sat_data_source.sat_data is not None


def test_available_timestamps(sat_data_source):
    timestamps = sat_data_source.available_timestamps()
    print(timestamps)
