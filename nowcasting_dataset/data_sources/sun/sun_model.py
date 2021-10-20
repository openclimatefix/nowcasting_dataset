""" Model for Sun features """
import logging

import numpy as np
from pydantic import Field, validator

from nowcasting_dataset.consts import Array, SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class Sun(DataSourceOutput):
    """ Class to store Sun data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data
    __slots__ = ()
    _expected_dimensions = ("time",)

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        assert (v.elevation != np.NaN).all(), f"Some elevation data values are NaNs"
        assert (v.elevation != np.Inf).all(), f"Some elevation data values are Infinite"

        assert (v.azimuth != np.NaN).all(), f"Some azimuth data values are NaNs"
        assert (v.azimuth != np.Inf).all(), f"Some azimuth data values are Infinite"

        assert (0 <= v.azimuth).all(), f"Some azimuth data values are lower 0"
        assert (v.azimuth <= 180).all(), f"Some azimuth data values are greater than 180"

        assert (0 <= v.elevation).all(), f"Some azimuth data values are lower 0"
        assert (v.elevation <= 90).all(), f"Some azimuth data values are greater than 180"

        return v
