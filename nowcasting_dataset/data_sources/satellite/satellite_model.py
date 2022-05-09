""" Model for output of satellite data """
from __future__ import annotations

import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)

satellite_expected_dims_order = (
    "example",
    "time_index",
    "channels_index",
    "y_geostationary_index",
    "x_geostationary_index",
)


class Satellite(DataSourceOutput):
    """Class to store satellite data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("time", "x_geostationary", "y_geostationary", "channels")
    _expected_data_vars = ("data",)

    @classmethod
    def model_validation(cls, v):
        """Check that all values are non negative"""
        v.check_nan_and_inf(data=v.data)
        # previously nans were filled with -1s, so lets make sure there are none
        v.check_dataset_not_equal(data=v.data, value=-1)
        v.check_data_var_dim(
            v.data,
            (
                "example",
                "time_index",
                "x_geostationary_index",
                "y_geostationary_index",
                "channels_index",
            ),
        )
        v.check_data_var_dim(v.x_geostationary, ("example", "x_geostationary_index"))
        v.check_data_var_dim(v.y_geostationary, ("example", "y_geostationary_index"))

        # re-order dims
        if v.data.dims != satellite_expected_dims_order:
            v.__setitem__("data", v.data.transpose(*satellite_expected_dims_order))

        return v


class HRVSatellite(Satellite):
    """Class to store HRV satellite data as a xr.Dataset with some validation"""

    __slots__ = ()
