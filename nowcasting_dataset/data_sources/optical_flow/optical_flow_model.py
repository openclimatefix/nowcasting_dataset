""" Model for output of Optical Flow data """
from __future__ import annotations

import numpy as np

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class OpticalFlow(DataSourceOutput):
    """Class to store optical flow data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("time", "x_osgb", "y_osgb", "channels")
    _expected_data_vars = ("data",)

    @classmethod
    def model_validation(cls, v):
        """Check that all values are not NaN, Infinite, or -1."""
        assert (~np.isnan(v.data)).all(), "Some optical flow data values are NaNs"
        assert (~np.isinf(v.data)).all(), "Some optical flow data values are Infinite"
        assert (v.data != -1).all(), "Some optical flow data values are -1"
        return v
