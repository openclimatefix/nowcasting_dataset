""" Model for Sun features """
import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class Sun(DataSourceOutput):
    """Class to store Sun data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("time",)
    _expected_data_vars = ("elevation", "azimuth")

    @classmethod
    def model_validation(cls, v):
        """Check that all values are non NaNs"""
        v.check_nan_and_inf(data=v.elevation, variable_name="elevation")
        v.check_nan_and_inf(data=v.azimuth, variable_name="azimuth")

        v.check_dataset_greater_than_or_equal_to(
            data=v.azimuth, variable_name="azimuth", min_value=0
        )
        v.check_dataset_less_than_or_equal_to(
            data=v.azimuth, variable_name="azimuth", max_value=360
        )

        v.check_dataset_greater_than_or_equal_to(
            data=v.elevation, variable_name="elevation", min_value=-90
        )
        v.check_dataset_less_than_or_equal_to(
            data=v.elevation, variable_name="elevation", max_value=90
        )

        v.check_data_var_dim(v.elevation, ("example", "time_index"))
        v.check_data_var_dim(v.azimuth, ("example", "time_index"))

        return v
