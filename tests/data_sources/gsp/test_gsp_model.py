"""Test GSP."""
import os
import tempfile

import numpy as np
import pytest

from nowcasting_dataset.data_sources.fake import gsp_fake
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP


def test_gsp_init():  # noqa: D103
    _ = gsp_fake(batch_size=4, seq_length_30=5, n_gsp_per_batch=6)


def test_gsp_normalized():
    """Test gsp normalization"""
    gsp = gsp_fake(batch_size=4, seq_length_30=5, n_gsp_per_batch=6)

    power_normalized = gsp.power_normalized

    assert (power_normalized.values >= 0).all()
    assert (power_normalized.values <= 1).all()


def test_gsp_validation():  # noqa: D103
    gsp = gsp_fake(batch_size=4, seq_length_30=5, n_gsp_per_batch=6)

    GSP.model_validation(gsp)

    gsp.power_mw[0, 0] = np.nan
    with pytest.raises(Exception):
        GSP.model_validation(gsp)


def test_gsp_save():  # noqa: D103

    with tempfile.TemporaryDirectory() as dirpath:
        gsp = gsp_fake(batch_size=4, seq_length_30=5, n_gsp_per_batch=6)
        gsp.save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/gsp/000000.nc")
