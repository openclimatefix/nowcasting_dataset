import os
import tempfile
import pytest
import numpy as np

from nowcasting_dataset.data_sources.fake import sun_fake
from nowcasting_dataset.data_sources.sun.sun_model import Sun


def test_sun_init():
    _ = sun_fake(batch_size=4, seq_length_5=17)


def test_sun_validation():
    sun = sun_fake(batch_size=4, seq_length_5=17)

    Sun.model_validation(sun)

    sun.elevation[0, 0] = np.nan
    with pytest.raises(Exception):
        Sun.model_validation(sun)


def test_sun_validation_elevation():
    sun = sun_fake(batch_size=4, seq_length_5=17)

    Sun.model_validation(sun)

    sun.elevation[0, 0] = 1000
    with pytest.raises(Exception):
        Sun.model_validation(sun)


def test_sun_validation_azimuth():
    sun = sun_fake(batch_size=4, seq_length_5=17)

    Sun.model_validation(sun)

    sun.azimuth[0, 0] = 1000
    with pytest.raises(Exception):
        Sun.model_validation(sun)


def test_sun_save():

    with tempfile.TemporaryDirectory() as dirpath:
        sun = sun_fake(batch_size=4, seq_length_5=17)
        sun.save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/sun/0.nc")
