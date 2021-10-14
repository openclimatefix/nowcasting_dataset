"""xr array and xr dataset --> to torch functions """

import torch
import xarray as xr
from typing import List
import numpy as np


def make_xr_data_array_to_tensor():
    if not hasattr(xr.DataArray, "torch"):

        @xr.register_dataarray_accessor("torch")
        class TorchAccessor:
            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            def to_tensor(self):
                """Convert this DataArray to a torch.Tensor"""
                return torch.tensor(self._obj.data)

            # def to_named_tensor(self):
            #     """Convert this DataArray to a torch.Tensor with named dimensions"""
            #     import torch
            #
            #     return torch.tensor(self._obj.data, names=self._obj.dims)


def make_xr_data_set_to_tensor():
    if not hasattr(xr.Dataset, "torch"):

        @xr.register_dataset_accessor("torch")
        class TorchAccessor:
            def __init__(self, xdataset_obj: xr.Dataset):
                self._obj = xdataset_obj

            def to_tensor(self, dims: List[str]) -> dict:
                """Convert this Dataset to dictionary of torch tensors"""

                torch_dict = {}

                for dim in dims:
                    v = getattr(self._obj, dim)
                    if dim.find("time") != -1:
                        v = v.astype(np.int32)

                    torch_dict[dim] = v.torch.to_tensor()

                return torch_dict
