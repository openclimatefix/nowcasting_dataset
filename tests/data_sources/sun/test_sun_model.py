# noqa: D100
import os
import tempfile

import numpy as np
import pytest

from nowcasting_dataset.data_sources.fake.batch import sun_fake
from nowcasting_dataset.data_sources.sun.sun_model import Sun


def test_sun_init(configuration):  # noqa: D103
    configuration.process.batch_size = 4
    _ = sun_fake(configuration=configuration)


def test_sun_validation(configuration):  # noqa: D103
    configuration.process.batch_size = 4
    sun = sun_fake(configuration=configuration)

    Sun.model_validation(sun)

    sun.elevation[0, 0] = np.nan
    with pytest.raises(Exception):
        Sun.model_validation(sun)


def test_sun_validation_elevation(configuration):  # noqa: D103
    configuration.process.batch_size = 4
    sun = sun_fake(configuration=configuration)

    Sun.model_validation(sun)

    sun.elevation[0, 0] = 1000
    with pytest.raises(Exception):
        Sun.model_validation(sun)


def test_sun_validation_azimuth(configuration):  # noqa: D103
    configuration.process.batch_size = 4
    sun = sun_fake(configuration=configuration)

    Sun.model_validation(sun)

    sun.azimuth[0, 0] = 1000
    with pytest.raises(Exception):
        Sun.model_validation(sun)


def test_sun_save(configuration):  # noqa: D103

    configuration.process.batch_size = 4
    with tempfile.TemporaryDirectory() as dirpath:
        sun = sun_fake(configuration=configuration)
        sun.save_netcdf(path=dirpath, batch_i=0)

        assert os.path.exists(f"{dirpath}/sun/000000.nc")
