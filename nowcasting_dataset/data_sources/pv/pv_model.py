""" Model for output of PV data """
from xarray.ufuncs import isinf, isnan

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class PV(DataSourceOutput):
    """ Class to store PV data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time", "id")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are not Nan, Infinite, or < 0."""
        assert (~isnan(v.data)).all(), "Some pv data values are NaNs"
        assert (~isinf(v.data)).all(), "Some pv data values are Infinite"
        assert (v.data >= 0).all(), "Some pv data values are below 0"

        assert v.time is not None
        assert v.x_coords is not None
        assert v.y_coords is not None
        assert v.pv_system_row_number is not None

        assert len(v.pv_system_row_number.shape) == 2

        return v
