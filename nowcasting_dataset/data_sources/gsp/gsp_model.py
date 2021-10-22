""" Model for output of GSP data """
import logging
from xarray.ufuncs import isnan, isinf

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutput,
)
from nowcasting_dataset.time import make_random_time_vectors

logger = logging.getLogger(__name__)


class GSP(DataSourceOutput):
    """ Class to store GSP data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time", "id")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        assert (~isnan(v.data)).all(), f"Some gsp data values are NaNs"
        assert (~isinf(v.data)).all(), f"Some gsp data values are Infinite"
        assert (v.data >= 0).all(), f"Some gsp data values are below 0 {v.data.min()}"

        return v
