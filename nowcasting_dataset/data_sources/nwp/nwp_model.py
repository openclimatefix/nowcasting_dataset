""" Model for output of NWP data """
from __future__ import annotations

import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class NWP(DataSourceOutput):
    """Class to store NWP data as a xr.Dataset with some validation"""

    # Use to store xr.Dataset data

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")
    _expected_data_vars = ("data",)

    @classmethod
    def model_validation(cls, v):
        """Check that all values are not NaNs"""

        # TODO issue 481, change back to 'check_nan_and_inf'
        v.__setitem__("data", v.check_nan_and_fill_warning(data=v.data))

        v.check_data_var_dim(
            v.data, ("example", "time_index", "x_index", "y_index", "channels_index")
        )

        return v
