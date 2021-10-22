""" Model for output of NWP data """
from __future__ import annotations

import logging

from xarray.ufuncs import isnan, isinf

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
    check_nan_and_inf,
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
        check_nan_and_inf(data=v.data, class_name="nwp")
        return v
