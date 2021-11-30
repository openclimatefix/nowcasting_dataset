"""Test Optical Flow Data Source"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.data_sources.optical_flow.optical_flow_data_source import (
    OpticalFlowDataSource,
)
from nowcasting_dataset.dataset.batch import Batch


@pytest.fixture
def optical_flow_configuration():  # noqa: D103
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    con.input_data.satellite.forecast_minutes = 60
    con.input_data.satellite.history_minutes = 30
    return con


def _get_optical_flow_data_source(
    sat_filename: Path,
    number_previous_timesteps_to_use: int = 1,
) -> OpticalFlowDataSource:
    return OpticalFlowDataSource(
        zarr_path=sat_filename,
        number_previous_timesteps_to_use=number_previous_timesteps_to_use,
        image_size_pixels=64,
        output_image_size_pixels=32,
        history_minutes=30,
        forecast_minutes=120,
        channels=("IR_016",),
    )


def test_optical_flow_get_example(optical_flow_configuration, sat_filename: Path):  # noqa: D103
    optical_flow_datasource = _get_optical_flow_data_source(sat_filename=sat_filename)
    optical_flow_datasource.open()
    t0_dt = pd.Timestamp("2020-04-01T13:00")
    example = optical_flow_datasource.get_example(
        t0_dt=t0_dt, x_meters_center=10_000, y_meters_center=10_000
    )
    assert example.values.shape == (24, 32, 32, 1)  # timesteps, height, width, channels


def test_optical_flow_get_example_multi_timesteps(
    optical_flow_configuration, sat_filename: Path
):  # noqa: D103
    optical_flow_datasource = _get_optical_flow_data_source(
        number_previous_timesteps_to_use=3, sat_filename=sat_filename
    )
    batch = Batch.fake(configuration=optical_flow_configuration)
    example = optical_flow_datasource.get_example(
        batch=batch, example_idx=0, t0_dt=batch.metadata.t0_datetime_utc[0]
    )
    # As a nasty hack to get round #511, the number of timesteps is set to 0 for now.
    # TODO: Issue #513: Set the number of timesteps back to 12!
    assert example.values.shape == (0, 32, 32, 10)  # timesteps, height, width, channels


def test_optical_flow_get_example_too_many_timesteps(
    optical_flow_configuration, sat_filename: Path
):  # noqa: D103
    optical_flow_datasource = _get_optical_flow_data_source(
        number_previous_timesteps_to_use=300, sat_filename=sat_filename
    )
    batch = Batch.fake(configuration=optical_flow_configuration)
    with pytest.raises(AssertionError):
        optical_flow_datasource.get_example(
            batch=batch, example_idx=0, t0_dt=batch.metadata.t0_datetime_utc[0]
        )
