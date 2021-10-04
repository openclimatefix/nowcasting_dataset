import torch
from pydantic import BaseModel, Field
import pandas as pd
import xarray as xr
import numpy as np

from typing import List


class DataSourceOutput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    batch_size: int = Field(
        0,
        ge=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    def to_numpy(self):
        """Change to numpy"""

        for k, v in self.dict().items():
            self.__setattr__(k, to_numpy(v))

    def to_xr_data_array(self):
        pass

    @staticmethod
    def join(data):

        _ = [d.to_numpy() for d in data]

        batch = data[0]
        for k, v in batch.dict().items():

            if k == "batch_size":
                batch.batch_size = len(data)
            else:

                one_variable_in_batch = [d.__getattribute__(k) for d in data]

                batch.__setattr__(k, np.stack(one_variable_in_batch, axis=0))

        return batch

    def split(self):

        c = self.__class__

        items = []
        for batch_idx in range(self.batch_size):
            d = {k: v[batch_idx] for k, v in self.dict().items() if k != "batch_size"}
            d["batch_size"] = 0
            items.append(c(**d))

        return items


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
    elif isinstance(value, torch.Tensor):
        value = value.numpy()

    return value
