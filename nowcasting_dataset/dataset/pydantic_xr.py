import xarray as xr


class PydanticXArrayDataSet(xr.Dataset):
    # Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__

    __slots__ = []

    # TODO add validation

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v
