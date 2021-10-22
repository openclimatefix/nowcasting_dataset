""" Model for output of PV data """
import logging

import numpy as np
from xarray.ufuncs import isnan, isinf
from pydantic import Field, validator

from nowcasting_dataset.consts import (
    Array,
    PV_YIELD,
    PV_DATETIME_INDEX,
    PV_SYSTEM_Y_COORDS,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_ID,
)
from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class PV(DataSourceOutput):
    """ Class to store PV data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data

    __slots__ = ()
    _expected_dimensions = ("time", "id")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        assert (~isnan(v.data)).all(), f"Some pv data values are NaNs"
        assert (~isinf(v.data)).all(), f"Some pv data values are Infinite"

        assert (v.data >= 0).all(), f"Some pv data values are below 0"

        return v
