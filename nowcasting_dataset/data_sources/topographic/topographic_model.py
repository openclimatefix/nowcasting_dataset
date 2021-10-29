""" Model for Topogrpahic features """
from xarray.ufuncs import isinf, isnan

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class Topographic(DataSourceOutput):
    """ Class to store topographic data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("x", "y")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        assert (~isnan(v.data)).all(), "Some topological data values are NaNs"
        assert (~isinf(v.data)).all(), "Some topological data values are Infinite"
        return v
