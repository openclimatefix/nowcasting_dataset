""" Utils Functions to for fake data """
from typing import List

import xarray as xr

from nowcasting_dataset.dataset.xr_utils import (
    convert_coordinates_to_indexes,
    join_list_dataset_to_batch_dataset,
)


def join_list_data_array_to_batch_dataset(data_arrays: List[xr.DataArray]) -> xr.Dataset:
    """Join a list of xr.DataArrays into an xr.Dataset by concatenating on the example dim."""
    datasets = [
        convert_coordinates_to_indexes(data_arrays[i].to_dataset()) for i in range(len(data_arrays))
    ]

    return join_list_dataset_to_batch_dataset(datasets)
