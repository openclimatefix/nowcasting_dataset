""" Useful functions for xarray objects

1. joining data arrays to datasets
2. pydantic exentions model of xr.Dataset
3. xr array and xr dataset --> to torch functions
"""
from typing import List, Any

import numpy as np
import torch
import xarray as xr


def join_list_data_array_to_batch_dataset(image_data_arrays: List[xr.DataArray]) -> xr.Dataset:
    """ Join a list of data arrays to a dataset byt expanding dims """
    image_data_arrays = [
        convert_data_array_to_dataset(image_data_arrays[i]) for i in range(len(image_data_arrays))
    ]

    return join_dataset_to_batch_dataset(image_data_arrays)


def join_dataset_to_batch_dataset(image_data_arrays: List[xr.Dataset]) -> xr.Dataset:
    """ Join a list of data arrays to a dataset byt expanding dims """
    image_data_arrays = [
        image_data_arrays[i].expand_dims(dim="example").assign_coords(example=("example", [i]))
        for i in range(len(image_data_arrays))
    ]

    return xr.concat(image_data_arrays, dim="example")


def convert_data_array_to_dataset(data_xarray):
    """ Convert data array to dataset. Reindex dim so that it can be merged with batch"""
    data = xr.Dataset({"data": data_xarray})

    return make_dim_index(data_xarray_dataset=data)


def make_dim_index(data_xarray_dataset: xr.Dataset) -> xr.Dataset:
    """ Reindex dataset dims so that it can be merged with batch"""

    dims = data_xarray_dataset.dims

    for dim in dims:
        coord = data_xarray_dataset[dim]
        data_xarray_dataset[dim] = np.arange(len(coord))

        data_xarray_dataset = data_xarray_dataset.rename({dim: f"{dim}_index"})

        data_xarray_dataset[dim] = xr.DataArray(
            coord, coords=data_xarray_dataset[f"{dim}_index"].coords, dims=[f"{dim}_index"]
        )

    return data_xarray_dataset


class PydanticXArrayDataSet(xr.Dataset):
    """Pydantic Xarray Dataset Class

    Adapted from https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__

    """

    _expected_dimensions = ()  # Subclasses should set this.

    # xarray doesnt support sub classing at the moment - https://github.com/pydata/xarray/issues/3980
    __slots__ = ()

    @classmethod
    def model_validation(cls, v):
        """ Specific model validation, to be overwritten by class """
        return v

    @classmethod
    def __get_validators__(cls):
        """Get validators"""
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> Any:
        """Do validation"""
        v = cls.validate_dims(v)
        v = cls.validate_coords(v)
        v = cls.model_validation(v)
        return v

    @classmethod
    def validate_dims(cls, v: Any) -> Any:
        """Validate the dims"""
        assert all(
            dim.replace("_index", "") in cls._expected_dimensions
            for dim in v.dims
            if dim != "example"
        ), (
            f"{cls.__name__}.dims is wrong! "
            f"{cls.__name__}.dims is {v.dims}. "
            f"But we expected {cls._expected_dimensions}. Note that '_index' is removed, and 'example' is ignored"
        )
        return v

    @classmethod
    def validate_coords(cls, v: Any) -> Any:
        """Validate the coords"""
        for dim in cls._expected_dimensions:
            coord = v.coords[f"{dim}_index"]
            assert len(coord) > 0, f"{dim}_index is empty in {cls.__name__}!"
        return v


def register_xr_data_array_to_tensor():
    """ Add torch object to data array """
    if not hasattr(xr.DataArray, "torch"):

        @xr.register_dataarray_accessor("torch")
        class TorchAccessor:
            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            def to_tensor(self):
                """Convert this DataArray to a torch.Tensor"""
                return torch.tensor(self._obj.data)

            # torch tensor names does not working in dataloader yet - 2021-10-15
            # https://discuss.pytorch.org/t/collating-named-tensors/78650
            # https://github.com/openclimatefix/nowcasting_dataset/issues/25
            # def to_named_tensor(self):
            #     """Convert this DataArray to a torch.Tensor with named dimensions"""
            #     import torch
            #
            #     return torch.tensor(self._obj.data, names=self._obj.dims)


def register_xr_data_set_to_tensor():
    """ Add torch object to dataset """
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
