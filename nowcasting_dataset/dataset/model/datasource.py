from pydantic import BaseModel
import pandas as pd
import xarray as xr
import numpy as np


class DataSource(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    def to_numpy(self):
        """Change to numpy"""

        dict_items = self.dict()

        return {k: to_numpy(v) for k, v in dict_items.items()}


def to_numpy(value):
    if isinstance(value, xr.DataArray):
        # TODO: Use to_numpy() or as_numpy(), introduced in xarray v0.19?
        value = value.data

    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.values
    elif isinstance(value, pd.DatetimeIndex):
        value = value.values.astype("datetime64[s]").astype(np.int32)
    elif isinstance(value, pd.Timestamp):
        value = np.int32(value.timestamp())
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.datetime64):
        value = value.astype("datetime64[s]").astype(np.int32)

    return value
