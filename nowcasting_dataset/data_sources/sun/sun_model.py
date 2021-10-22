""" Model for Sun features """
import logging

import numpy as np
from xarray.ufuncs import isnan, isinf
from pydantic import Field, validator

from nowcasting_dataset.consts import Array, SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
    check_nan_and_inf,
    check_dataset_greater_than,
    check_dataset_less_than,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class Sun(DataSourceOutput):
    """ Class to store Sun data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data
    __slots__ = ()
    _expected_dimensions = ("time",)

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        check_nan_and_inf(data=v.elevation, class_name="sun elevation")
        check_nan_and_inf(data=v.azimuth, class_name="sun azimuth")

        check_dataset_greater_than(data=v.azimuth, class_name="sun azimuth", min_value=0)
        check_dataset_less_than(data=v.azimuth, class_name="sun azimuth", max_value=360)

        check_dataset_greater_than(data=v.elevation, class_name="sun elevation", min_value=-90)
        check_dataset_less_than(data=v.elevation, class_name="sun elevation", max_value=90)

        return v
