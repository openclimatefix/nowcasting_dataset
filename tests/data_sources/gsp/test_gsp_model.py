import os
import tempfile
import pytest
import numpy as np

from nowcasting_dataset.data_sources.fake import gsp_fake
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP


def test_gsp_init():
    _ = gsp_fake(batch_size=4, seq_length_30=5, n_gsp_per_batch=6)


def test_gsp_validation():
    gsp = gsp_fake(batch_size=4, seq_length_30=5, n_gsp_per_batch=6)

    GSP.model_validation(gsp)

    gsp.data[0, 0] = np.nan
    with pytest.raises(Exception):
        GSP.model_validation(gsp)


def test_gsp_save():

    with tempfile.TemporaryDirectory() as dirpath:
        gsp = gsp_fake(batch_size=4, seq_length_30=5, n_gsp_per_batch=6)
        gsp.save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/gsp/0.nc")
