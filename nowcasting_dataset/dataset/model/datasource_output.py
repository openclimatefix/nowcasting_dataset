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

    def to_xr_dataset(self):
        raise NotImplementedError

    def from_xr_dataset(self):
        raise NotImplementedError

    def select_time_period(
        self,
        keys: List[str],
        time_of_first_example: pd.DatetimeIndex,
        start_time: xr.DataArray,
        end_time: xr.DataArray,
    ):
        """
        Selects a subset of data between the indicies of [start, end] for each key in keys

        Args:
            batch: Example containing the data
            keys: Keys in batch to use
            time_of_first_example: Datetime of the current time in the first example of the batch
            start_time: Start time DataArray
            end_time: End time DataArray

        Returns:
            Example containing the subselected data
        """
        start_i, end_i = np.searchsorted(time_of_first_example, [start_time, end_time])
        for key in keys:
            self.__setattr__(key, self.__getattribute__(key).isel(time=slice(start_i, end_i)))


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


def pad_nans(array, pad_width) -> np.ndarray:
    """ Pad nans with nans"""
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)


def pad_data(
    data: DataSourceOutput,
    pad_size: int,
    one_dimensional_arrays: List[str],
    two_dimensional_arrays: List[str],
):
    """
    Pad (if necessary) so returned arrays are always of size

    data has two types of arrays in it, one dimensional arrays and two dimensional arrays
    the one dimensional arrays are padded in that dimension
    the two dimensional arrays are padded in the second dimension

    Args:
        data: typed dictionary of data objects
        pad_size: the maount that should be padded
        one_dimensional_arrays: list of data items that should be padded by one dimension
        two_dimensional_arrays: list of data tiems that should be padded in the third dimension (and more)

    Returns: Example data

    """
    # Pad (if necessary) so returned arrays are always of size
    pad_shape = (0, pad_size)  # (before, after)

    for name in one_dimensional_arrays:
        data.__setattr__(name, pad_nans(data.__getattribute__(name), pad_width=pad_shape))

    for variable in two_dimensional_arrays:
        data.__setattr__(
            variable, pad_nans(data.__getattribute__(variable), pad_width=((0, 0), pad_shape))
        )  # (axis0, axis1)
