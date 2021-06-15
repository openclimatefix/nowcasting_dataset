"""Configure PyTest"""
import pytest
from pathlib import Path
from nowcasting_dataset import consts


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
        return Path(__file__).parent.absolute() / 'data' / 'sat_data.zarr'
