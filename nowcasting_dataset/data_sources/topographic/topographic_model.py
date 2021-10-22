""" Model for Topogrpahic features """
import logging

import numpy as np
from xarray.ufuncs import isnan, isinf
from pydantic import Field, validator

from nowcasting_dataset.consts import Array
from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput, check_nan_and_inf

logger = logging.getLogger(__name__)


class Topographic(DataSourceOutput):
    """ Class to store topographic data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data
    __slots__ = ()
    _expected_dimensions = ("x", "y")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        check_nan_and_inf(data=v.data, class_name="topological")
        return v
