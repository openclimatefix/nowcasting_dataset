"""Test GSP."""
import os
import tempfile

import numpy as np
import pytest

from nowcasting_dataset.data_sources.fake.batch import gsp_fake
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP


def test_gsp_init(configuration):  # noqa: D103

    configuration.process.batch_size = 4
    configuration.input_data.gsp.history_minutes = 60
    configuration.input_data.gsp.forecast_minutes = 60
    configuration.input_data.gsp.n_gsp_per_example = 6

    _ = gsp_fake(configuration=configuration)


def test_gsp_normalized(configuration):
    """Test gsp normalization"""

    configuration.process.batch_size = 4
    configuration.input_data.gsp.history_minutes = 60
    configuration.input_data.gsp.forecast_minutes = 60
    configuration.input_data.gsp.n_gsp_per_example = 6

    gsp = gsp_fake(configuration=configuration)

    power_normalized = gsp.power_normalized

    assert (power_normalized.values >= 0).all()
    assert (power_normalized.values <= 1).all()


def test_gsp_validation(configuration):  # noqa: D103

    configuration.process.batch_size = 4
    configuration.input_data.gsp.history_minutes = 60
    configuration.input_data.gsp.forecast_minutes = 60
    configuration.input_data.gsp.n_gsp_per_example = 6

    gsp = gsp_fake(configuration)

    GSP.model_validation(gsp)

    gsp.power_mw[0, 0] = np.nan
    with pytest.raises(Exception):
        GSP.model_validation(gsp)


def test_gsp_save(configuration):  # noqa: D103

    configuration.process.batch_size = 4
    configuration.input_data.gsp.history_minutes = 60
    configuration.input_data.gsp.forecast_minutes = 60
    configuration.input_data.gsp.n_gsp_per_example = 6

    with tempfile.TemporaryDirectory() as dirpath:
        gsp = gsp_fake(configuration)
        gsp.save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/gsp/000000.nc")
