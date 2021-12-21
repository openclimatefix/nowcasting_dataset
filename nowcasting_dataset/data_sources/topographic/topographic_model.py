""" Model for Topogrpahic features """

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput


class Topographic(DataSourceOutput):
    """Class to store topographic data as a xr.Dataset with some validation"""

    __slots__ = ()
    _expected_dimensions = ("x_osgb", "y_osgb")
    _expected_data_vars = ("data",)

    @classmethod
    def model_validation(cls, v):
        """Check that all values are non NaNs"""

        v.check_nan_and_inf(data=v.data)

        v.check_data_var_dim(v.data, ("example", "x_osgb_index", "y_osgb_index"))

        return v
