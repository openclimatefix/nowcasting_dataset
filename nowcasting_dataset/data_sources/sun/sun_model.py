""" Model for Sun features """
import logging

import numpy as np
from pydantic import Field, validator

from nowcasting_dataset.consts import Array, SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class Sun(DataSourceOutput):
    """ Class to store Sun data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data
    __slots__ = ()
    _expected_dimensions = ("time",)

    # todo add validation here - https://github.com/openclimatefix/nowcasting_dataset/issues/233
