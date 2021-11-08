""" Model for output of GSP data """
import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class GSP(DataSourceOutput):
    """Class to store GSP data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("time", "id")

    @classmethod
    def model_validation(cls, v):
        """Check that all values are non NaNs"""

        v.check_nan_and_inf(data=v.data)
        v.check_dataset_greater_than_or_equal_to(data=v.data, min_value=0)

        return v
