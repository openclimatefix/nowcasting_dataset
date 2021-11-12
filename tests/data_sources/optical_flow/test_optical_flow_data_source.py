"""Test Optical Flow Data Source"""
import tempfile

import numpy as np
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


def test_optical_flow_get_example(optical_flow_configuration):
    optical_flow_datasource = OpticalFlowDataSource(
        number_previous_timesteps_to_use=1, image_size_pixels=32
    )
    batch = Batch.fake(configuration=optical_flow_configuration)
    example = optical_flow_datasource.get_example(batch=batch, example_idx=0,
                                                  t0_datetime=batch.metadata.t0_dt.values[0])
    assert example.values.shape == (12, 32, 32, 12)


def test_optical_flow_get_example_multi_timesteps(optical_flow_configuration):
    optical_flow_datasource = OpticalFlowDataSource(
        number_previous_timesteps_to_use=3, image_size_pixels=32
    )
    batch = Batch.fake(configuration=optical_flow_configuration)
    example = optical_flow_datasource.get_example(batch=batch, example_idx=0, t0_datetime=batch.metadata.t0_dt.values[0])
    assert example.values.shape == (12, 32, 32, 12)


def test_optical_flow_get_example_too_many_timesteps(optical_flow_configuration):
    optical_flow_datasource = OpticalFlowDataSource(
        number_previous_timesteps_to_use=300, image_size_pixels=32
    )
    batch = Batch.fake(configuration=optical_flow_configuration)
    with pytest.raises(AssertionError):
        optical_flow_datasource.get_example(batch=batch, example_idx=0, t0_datetime=batch.metadata.t0_dt.values[0])


def test_optical_flow_data_source_get_batch(optical_flow_configuration):  # noqa: D103
    optical_flow_datasource = OpticalFlowDataSource(
        number_previous_timesteps_to_use=1, image_size_pixels=32
    )
    with tempfile.TemporaryDirectory() as dirpath:
        batch = Batch.fake(configuration=optical_flow_configuration)
        batch.save_netcdf(path=dirpath, batch_i=0)
        optical_flow = optical_flow_datasource.get_batch(netcdf_path=dirpath, batch_idx=0,
                                                         t0_datetimes = batch.metadata.t0_dt.values)
        assert optical_flow.values.shape == (4, 12, 32, 32, 12)
