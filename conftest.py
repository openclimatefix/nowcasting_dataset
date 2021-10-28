"""Configure PyTest"""
import os
from pathlib import Path

import pytest

import nowcasting_dataset
from nowcasting_dataset import consts
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.data_sources import SatelliteDataSource
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.metadata.metadata_data_source import MetadataDataSource
from nowcasting_dataset.dataset.xr_utils import (
    register_xr_data_array_to_tensor,
    register_xr_data_set_to_tensor,
)

pytest.IMAGE_SIZE_PIXELS = 128

# need to run these to ensure that xarray DataArray and Dataset have torch functions
register_xr_data_array_to_tensor()
register_xr_data_set_to_tensor()


def pytest_addoption(parser):
    parser.addoption(
        "--use_cloud_data",
        action="store_true",
        default=False,
        help="Use large datasets on GCP instead of local test datasets.",
    )


@pytest.fixture
def use_cloud_data(request):
    return request.config.getoption("--use_cloud_data")


@pytest.fixture
def sat_filename(use_cloud_data: bool) -> Path:
    if use_cloud_data:
        return consts.SAT_FILENAME
    else:
        filename = Path(__file__).parent.absolute() / "tests" / "data" / "sat_data.zarr"
        assert filename.exists()
        return filename


@pytest.fixture
def sat_data_source(sat_filename: Path):
    return SatelliteDataSource(
        image_size_pixels=pytest.IMAGE_SIZE_PIXELS,
        zarr_path=sat_filename,
        history_minutes=0,
        forecast_minutes=5,
        channels=("HRV",),
        n_timesteps_per_batch=2,
    )


@pytest.fixture
def general_data_source():
    return MetadataDataSource(history_minutes=0, forecast_minutes=5, object_at_center="GSP")


@pytest.fixture
def gsp_data_source():
    return GSPDataSource(
        image_size_pixels=16,
        meters_per_pixel=2000,
        zarr_path=Path(__file__).parent.absolute() / "tests" / "data" / "gsp" / "test.zarr",
        history_minutes=0,
        forecast_minutes=30,
    )


@pytest.fixture
def configuration():
    filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
    configuration = load_yaml_configuration(filename)

    return configuration


@pytest.fixture
def test_data_folder():
    return os.path.join(os.path.dirname(nowcasting_dataset.__file__), "../tests/data")
