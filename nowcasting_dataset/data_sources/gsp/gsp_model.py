""" Model for output of GSP data """
import logging

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
    check_dataset_greater_than,
    check_nan_and_inf,
)

logger = logging.getLogger(__name__)


class GSP(DataSourceOutput):
    """ Class to store GSP data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time", "id")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """

        check_nan_and_inf(data=v.data, class_name="gsp")
        check_dataset_greater_than(data=v.data, class_name="gsp", min_value=0)

        return v
