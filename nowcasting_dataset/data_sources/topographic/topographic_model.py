""" Model for Topogrpahic features """

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class Topographic(DataSourceOutput):
    """ Class to store topographic data as a xr.Dataset with some validation """

    __slots__ = ()
    _expected_dimensions = ("x", "y")

    @classmethod
    def model_validation(cls, v):
        """ Check that all values are non NaNs """

        v.check_nan_and_inf(data=v.data)

        return v
