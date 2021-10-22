""" Model for output of satellite data """
from __future__ import annotations

import logging

import numpy as np
from xarray.ufuncs import isnan, isinf
from pydantic import Field

from nowcasting_dataset.consts import Array
from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class Satellite(DataSourceOutput):
    """ Class to store satellite data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non negative """
        assert (~isnan(v.data)).all(), f"Some satellite data values are NaNs"
        assert (~isinf(v.data)).all(), f"Some satellite data values are Infinite"
        assert (v.data != -1).all(), f"Some satellite data values are -1's"
        return v
