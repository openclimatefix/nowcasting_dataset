""" Model for output of general/metadata data, useful for a batch """
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class Metadata(DataSourceOutput):
    """ Class to store metedata data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("t0_dt",)

    # TODO: Add validation here - https://github.com/openclimatefix/nowcasting_dataset/issues/233
