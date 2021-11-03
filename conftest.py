"""Configure PyTest"""
import os
from pathlib import Path

import pytest

import nowcasting_dataset
from nowcasting_dataset import consts
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.data_sources import OpticalFlowDataSource, SatelliteDataSource
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.metadata.metadata_data_source import MetadataDataSource

pytest.IMAGE_SIZE_PIXELS = 128


def pytest_addoption(parser):  # noqa: D103
    parser.addoption(
        "--use_cloud_data",
        action="store_true",
        default=False,
        help="Use large datasets on GCP instead of local test datasets.",
    )


@pytest.fixture
def use_cloud_data(request):  # noqa: D103
    return request.config.getoption("--use_cloud_data")


@pytest.fixture
def sat_filename(use_cloud_data: bool) -> Path:  # noqa: D103
    if use_cloud_data:
        return consts.SAT_FILENAME
    else:
        filename = Path(__file__).parent.absolute() / "tests" / "data" / "sat_data.zarr"
        assert filename.exists()
        return filename


@pytest.fixture
def sat_data_source(sat_filename: Path):  # noqa: D103
    return SatelliteDataSource(
        image_size_pixels=pytest.IMAGE_SIZE_PIXELS,
        zarr_path=sat_filename,
        history_minutes=0,
        forecast_minutes=5,
        channels=("HRV",),
    )


@pytest.fixture
def optical_flow_data_source(sat_filename: Path):  # noqa: D103
    return OpticalFlowDataSource(
        image_size_pixels=pytest.IMAGE_SIZE_PIXELS,
        zarr_path=sat_filename,
        history_minutes=0,
        forecast_minutes=5,
        channels=("HRV",),
    )


@pytest.fixture
def general_data_source():  # noqa: D103
    return MetadataDataSource(history_minutes=0, forecast_minutes=5, object_at_center="GSP")


@pytest.fixture
def gsp_data_source():  # noqa: D103
    return GSPDataSource(
        image_size_pixels=16,
        meters_per_pixel=2000,
        zarr_path=Path(__file__).parent.absolute() / "tests" / "data" / "gsp" / "test.zarr",
        history_minutes=0,
        forecast_minutes=30,
    )


@pytest.fixture
def configuration():  # noqa: D103
    filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
    configuration = load_yaml_configuration(filename)

    return configuration


@pytest.fixture
def test_data_folder():  # noqa: D103
    return os.path.join(os.path.dirname(nowcasting_dataset.__file__), "../tests/data")
