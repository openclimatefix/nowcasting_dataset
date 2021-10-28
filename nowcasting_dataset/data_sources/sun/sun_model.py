""" Model for Sun features """
from xarray.ufuncs import isinf, isnan

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class Sun(DataSourceOutput):
    """ Class to store Sun data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("time",)

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """
        assert (~isnan(v.elevation)).all(), "Some elevation data values are NaNs"
        assert (~isinf(v.elevation)).all(), "Some elevation data values are Infinite"

        assert (~isnan(v.azimuth)).all(), "Some azimuth data values are NaNs"
        assert (~isinf(v.azimuth)).all(), "Some azimuth data values are Infinite"

        assert (0 <= v.azimuth).all(), f"Some azimuth data values are lower 0, {v.azimuth.min()}"
        assert (
            v.azimuth <= 360
        ).all(), f"Some azimuth data values are greater than 360, {v.azimuth.max()}"

        assert (
            -90 <= v.elevation
        ).all(), f"Some elevation data values are lower -90, {v.elevation.min()}"
        assert (
            v.elevation <= 90
        ).all(), f"Some elevation data values are greater than 90, {v.elevation.max()}"

        return v
