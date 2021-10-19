""" Model for output of NWP data """
from __future__ import annotations

import logging

import numpy as np

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class NWP(DataSourceOutput):
    """ Class to store NWP data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are not NaNs """
        assert (v.data != np.nan).all(), "Some nwp data values are NaNs"
        return v
