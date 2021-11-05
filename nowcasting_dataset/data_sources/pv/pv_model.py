""" Model for output of PV data """

import logging

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

logger = logging.getLogger(__name__)


class PV(DataSourceOutput):
    """ Class to store PV data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time", "id")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        v.check_nan_and_inf(data=v.data)
        v.check_dataset_greater_than_or_equal_to(data=v.data, min_value=0)

        assert v.time is not None
        assert v.x_coords is not None
        assert v.y_coords is not None
        assert v.pv_system_row_number is not None

        assert len(v.pv_system_row_number.shape) == 2

        return v
