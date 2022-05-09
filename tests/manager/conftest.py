""" Pytest fixtures for manager tests"""
from datetime import datetime
from pathlib import Path

import pytest

import nowcasting_dataset
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.satellite.satellite_data_source import (
    HRVSatelliteDataSource,
    SatelliteDataSource,
)
from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource


@pytest.fixture()
def sat():
    """Get Satellite data source"""
    filename = Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"

    return SatelliteDataSource(
        zarr_path=filename,
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels_height=24,
        image_size_pixels_width=24,
        meters_per_pixel=6000,
        channels=("IR_016",),
    )


@pytest.fixture()
def hrvsat():
    """Get HRV Satellite data source"""
    filename = (
        Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    )
    return HRVSatelliteDataSource(
        zarr_path=filename,
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
        channels=("HRV",),
    )


@pytest.fixture()
def gsp():
    """Get GSP data source"""
    filename = (
        Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    )

    return GSPDataSource(
        zarr_path=filename,
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
    )


@pytest.fixture()
def sun():
    """Get Sun data source"""
    filename = (
        Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "sun" / "test.zarr"
    )

    return SunDataSource(
        zarr_path=filename,
        history_minutes=30,
        forecast_minutes=60,
    )
