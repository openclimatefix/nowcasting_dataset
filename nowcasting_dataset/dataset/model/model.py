from pydantic import BaseModel, Field, validator
from typing import Union, List
import numpy as np
import pandas as pd
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.satellite import Satellite


Array = Union[xr.DataArray, np.ndarray, torch.Tensor]


class Batch(BaseModel):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    satellite: Satellite

    def change_type_to_xr_data_array(self):
        pass
        # go other datasoruces and change them to xr data arrays

    def change_type_to_numpy(self):

        self.satellite = Satellite.to_numpy()
        # other datasources

    def batch_to_dataset(self) -> xr.Dataset:
        """ Change batch to xr.Dataset so it can be saved and compressed"""
        pass

    def load_batch_from_dataset(self, xr_dataset: xr.Dataset):
        """ Change xr.Datatset to Batch object"""
        return Batch()


def join_data_to_batch(data=List[Batch]) -> Batch:
    """ Join several single data items together to make a Batch """
    pass


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
