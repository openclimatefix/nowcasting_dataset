""" Pydantic extension of xarray dataset """
import xarray as xr


class PydanticXArrayDataSet(xr.Dataset):
    """Pydantic Xarray Dataset Class

    Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__

    """

    __slots__ = []

    # TODO add validation

    @classmethod
    def __get_validators__(cls):
        """ Get validators """
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """ Do validation """
        return v
