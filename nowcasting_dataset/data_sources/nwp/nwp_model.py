""" Model for output of NWP data """
from __future__ import annotations

import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class NWP(DataSourceOutput):
    """Class to store NWP data as a xr.Dataset with some validation"""

    # Use to store xr.Dataset data

    __slots__ = ()
    _expected_dimensions = ("time", "y_osgb", "x_osgb", "channels")
    _expected_data_vars = ("data",)

    @classmethod
    def model_validation(cls, v):
        """Check that all values are not NaNs"""

        v.check_nan_and_inf(data=v.data)

        dims = ("example", "time_index", "y_osgb_index", "x_osgb_index", "channels_index")
        v.check_data_var_dim(v.data, dims)
        v = v.order_dims(v, variable="data", expected_dims=dims)

        return v
