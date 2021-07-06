"""Configure PyTest"""
import pytest
from pathlib import Path
from nowcasting_dataset import consts
from nowcasting_dataset.data_sources import SatelliteDataSource


pytest.IMAGE_SIZE_PIXELS = 128


def pytest_addoption(parser):
    parser.addoption(
        "--use_cloud_data", action="store_true", default=False,
        help="Use large datasets on GCP instead of local test datasets.")


@pytest.fixture
def use_cloud_data(request):
    return request.config.getoption("--use_cloud_data")


@pytest.fixture
def sat_filename(use_cloud_data: bool) -> Path:
    if use_cloud_data:
        return consts.SAT_FILENAME
    else:
        filename = (
            Path(__file__).parent.absolute() /
            'nowcasting_dataset' / 'tests' / 'data' / 'sat_data.zarr')
        assert filename.exists()
        return filename


@pytest.fixture
def sat_data_source(sat_filename: Path):
    return SatelliteDataSource(
        image_size_pixels=pytest.IMAGE_SIZE_PIXELS,
        filename=sat_filename,
        history_len=0,
        forecast_len=1,
        channels=('HRV', ),
        n_timesteps_per_batch=2)
