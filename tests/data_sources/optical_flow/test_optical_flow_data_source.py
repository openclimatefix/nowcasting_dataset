"""Test Optical Flow Data Source"""
from pathlib import Path

import pandas as pd
import pytest

from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.data_sources.optical_flow.optical_flow_data_source import (
    OpticalFlowDataSource,
)


@pytest.fixture
def optical_flow_configuration():  # noqa: D103
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    con.input_data.satellite.forecast_minutes = 60
    con.input_data.satellite.history_minutes = 30
    return con


def _get_optical_flow_data_source(
    sat_filename: Path, history_minutes: int = 5
) -> OpticalFlowDataSource:
    return OpticalFlowDataSource(
        zarr_path=sat_filename,
        channels=("IR_016",),
        history_minutes=history_minutes,
        forecast_minutes=120,
        input_image_size_pixels=64,
        output_image_size_pixels=32,
    )


@pytest.mark.parametrize("history_minutes", [5, 15])
def test_optical_flow_get_example(
    optical_flow_configuration, sat_filename: Path, history_minutes: int
):  # noqa: D103
    optical_flow_datasource = _get_optical_flow_data_source(
        sat_filename=sat_filename, history_minutes=history_minutes
    )
    optical_flow_datasource.open()
    t0_dt = pd.Timestamp("2020-04-01T13:00")
    example = optical_flow_datasource.get_example(
        t0_dt=t0_dt, x_meters_center=10_000, y_meters_center=10_000
    )
    assert example["data"].shape == (24, 32, 32, 1)  # timesteps, height, width, channels
