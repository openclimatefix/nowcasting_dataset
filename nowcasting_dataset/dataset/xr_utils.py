""" Useful functions for xarray objects """
import xarray as xr
import numpy as np
from typing import List


def from_list_data_array_to_batch_dataset(image_data_arrays: List[xr.DataArray]) -> xr.Dataset:
    """ Join a list of data arrays to a dataset byt expanding dims """
    image_data_arrays = [
        convert_data_array_to_dataset(image_data_arrays[i]) for i in range(len(image_data_arrays))
    ]

    return join_data_set_to_batch_dataset(image_data_arrays)


def join_data_set_to_batch_dataset(image_data_arrays: List[xr.Dataset]) -> xr.Dataset:
    """ Join a list of data arrays to a dataset byt expanding dims """
    image_data_arrays = [
        image_data_arrays[i].expand_dims(dim="example").assign_coords(example=("example", [i]))
        for i in range(len(image_data_arrays))
    ]

    return xr.concat(image_data_arrays, dim="example")


def convert_data_array_to_dataset(data_xarray):
    """ Convert data array to dataset. Reindex dim so that it can be merged with batch"""
    # convert data array to dataset, and re index dims
    dims = data_xarray.dims
    data = xr.Dataset({"data": data_xarray})

    for dim in dims:
        coord = data[dim]
        data[dim] = np.arange(len(coord))

        data = data.rename({dim: f"{dim}_index"})

        data[dim] = xr.DataArray(coord, coords=data[f"{dim}_index"].coords, dims=[f"{dim}_index"])

    return data
