""" Model for output of datetime data """
from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)

from nowcasting_dataset.utils import coord_to_range


class Datetime(DataSourceOutput):
    """ Class to store Datetime data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data

    __slots__ = ()

    # todo add validation here - https://github.com/openclimatefix/nowcasting_dataset/issues/233
    _expected_dimensions = ("time",)
