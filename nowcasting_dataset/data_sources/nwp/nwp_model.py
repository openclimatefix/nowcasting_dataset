""" Model for output of NWP data """
from __future__ import annotations

import logging

from xarray.ufuncs import isinf, isnan

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class NWP(DataSourceOutput):
    """ Class to store NWP data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are not NaNs """
        assert (~isnan(v.data)).all(), "Some nwp data values are NaNs"
        assert (~isinf(v.data)).all(), "Some nwp data values are Infinite"
        return v
