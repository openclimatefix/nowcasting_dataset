"""Test Optical Flow Data Source"""
import pytest
import tempfile

import pytest

from nowcasting_dataset.config.model import Configuration, InputData
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_dataset.data_sources.optical_flow.optical_flow_data_source import (
    OpticalFlowDataSource,
    )


@pytest.fixture
def configuration():  # noqa: D103
    con = Configuration()
    con.input_data = InputData.set_all_to_defaults()
    con.process.batch_size = 4
    return con


def test_optical_flow_data_source_get_batch(configuration):  # noqa: D103
    optical_flow_datasource = OpticalFlowDataSource(previous_timestep_for_flow = 1,
                                                    final_image_size_pixels = 64)
    with tempfile.TemporaryDirectory() as dirpath:
        Batch.fake(configuration=configuration).save_netcdf(path=dirpath, batch_i=0)

    optical_flow = optical_flow_datasource.get_batch(netcdf_path = dirpath, batch_idx = 0)
    print(optical_flow)
