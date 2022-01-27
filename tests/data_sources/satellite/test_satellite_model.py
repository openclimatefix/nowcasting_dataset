"""Test Satellite model."""
import os
import tempfile

import numpy as np
import pytest

from nowcasting_dataset.data_sources.fake.batch import satellite_fake
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite


def test_satellite_init():  # noqa: D103
    satellite = satellite_fake()

    assert satellite.x_geostationary.dims == ("example", "x_geostationary_index")


def test_satellite_validation():  # noqa: D103
    sat = satellite_fake()

    Satellite.model_validation(sat)

    sat.data[0, 0] = np.nan
    with pytest.raises(Exception):
        Satellite.model_validation(sat)


def test_satellite_save():  # noqa: D103

    with tempfile.TemporaryDirectory() as dirpath:
        satellite_fake().save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/satellite/000000.nc")
