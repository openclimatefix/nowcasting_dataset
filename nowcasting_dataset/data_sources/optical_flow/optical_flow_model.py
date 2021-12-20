""" Model for output of Optical Flow data """
from __future__ import annotations

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class OpticalFlow(DataSourceOutput):
    """Class to store optical flow data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("time", "x", "y", "channels")
    _expected_data_vars = ("data",)

    @classmethod
    def model_validation(cls, v):
        """Check that all values are not NaN, Infinite, or -1."""
        v.check_nan_and_inf(data=v.data)
        # previously nans were filled with -1s, so lets make sure there are none
        v.check_dataset_not_equal(data=v.data, value=-1)
        v.check_data_var_dim(v.x, ("example", "x_index"))
        v.check_data_var_dim(v.y, ("example", "y_index"))

        return v
