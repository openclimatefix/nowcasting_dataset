"""Test Optical Flow model."""
import os
import tempfile

import numpy as np
import pytest

from nowcasting_dataset.data_sources.fake import optical_flow_fake
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow


def test_optical_flow_init():  # noqa: D103
    _ = optical_flow_fake()


def test_optical_flow_validation():  # noqa: D103
    sat = optical_flow_fake()

    OpticalFlow.model_validation(sat)

    sat.data[0, 0] = np.nan
    with pytest.raises(Exception):
        OpticalFlow.model_validation(sat)


def test_optical_flow_save():  # noqa: D103

    with tempfile.TemporaryDirectory() as dirpath:
        optical_flow_fake().save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/opticalflow/000000.nc")
