""" Model for output of satellite data """
from __future__ import annotations

import logging

from xarray.ufuncs import isinf, isnan

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class Satellite(DataSourceOutput):
    """ Class to store satellite data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non negative """
        assert (~isnan(v.data)).all(), "Some satellite data values are NaNs"
        assert (~isinf(v.data)).all(), "Some satellite data values are Infinite"
        assert (v.data != -1).all(), "Some satellite data values are -1's"

        return v
