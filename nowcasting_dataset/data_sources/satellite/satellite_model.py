""" Model for output of satellite data """
from __future__ import annotations

import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class Satellite(DataSourceOutput):
    """Class to store satellite data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")
    _expected_data_vars = ("data",)

    @classmethod
    def model_validation(cls, v):
        """Check that all values are non negative"""
        v.check_nan_and_inf(data=v.data)
        # previously nans were filled with -1s, so lets make sure there are none
        v.check_dataset_not_equal(data=v.data, value=-1)
        v.check_data_var_dim(
            v.data, ("example", "time_index", "x_index", "y_index", "channels_index")
        )

        return v


class HRVSatellite(Satellite):
    """Class to store HRV satellite data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")
    _expected_data_vars = ("data",)
