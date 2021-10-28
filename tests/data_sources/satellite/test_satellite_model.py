import os
import tempfile

import numpy as np
import pytest

from nowcasting_dataset.data_sources.fake import satellite_fake
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite


def test_satellite_init():
    _ = satellite_fake()


def test_satellite_validation():
    sat = satellite_fake()

    Satellite.model_validation(sat)

    sat.data[0, 0] = np.nan
    with pytest.raises(Exception):
        Satellite.model_validation(sat)


def test_satellite_save():

    with tempfile.TemporaryDirectory() as dirpath:
        satellite_fake().save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/0.nc")
