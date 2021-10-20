""" Model for output of general/metadata data, useful for a batch """
from typing import Union

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)

from nowcasting_dataset.time import make_random_time_vectors


# seems to be a pandas dataseries


class Metadata(DataSourceOutput):
    """ Class to store metedata data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("t0_dt",)

    # todo add validation here - https://github.com/openclimatefix/nowcasting_dataset/issues/233
