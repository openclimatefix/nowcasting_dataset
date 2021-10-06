""" batch functions """
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

from nowcasting_dataset.consts import (
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_DATETIME_INDEX,
    DATETIME_FEATURE_NAMES,
    T0_DT,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_X_COORDS,
    TOPOGRAPHIC_Y_COORDS,
)

# from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.utils import get_netcdf_filename

_LOG = logging.getLogger(__name__)

# TODO
# def write_batch_locally(batch: List[Example], batch_i: int, path: Path):
#     """
#     Write a batch to a locally file
#
#     Args:
#         batch: A batch of data
#         batch_i: The number of the batch
#         path: The directory to write the batch into.
#     """
#     dataset = batch_to_dataset(batch)
#     dataset = fix_dtypes(dataset)
#     encoding = {name: {"compression": "lzf"} for name in dataset.data_vars}
#     filename = get_netcdf_filename(batch_i)
#     local_filename = path / filename
#     dataset.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)
#


def fix_dtypes(concat_ds):
    """
    TODO
    """
    ds_dtypes = {
        "example": np.int32,
        "sat_x_coords": np.int32,
        "sat_y_coords": np.int32,
        "nwp": np.float32,
        "nwp_x_coords": np.float32,
        "nwp_y_coords": np.float32,
        "pv_system_id": np.float32,
        "pv_system_row_number": np.float32,
        "pv_system_x_coords": np.float32,
        "pv_system_y_coords": np.float32,
        GSP_YIELD: np.float32,
        GSP_ID: np.float32,
        GSP_X_COORDS: np.float32,
        GSP_Y_COORDS: np.float32,
        TOPOGRAPHIC_X_COORDS: np.float32,
        TOPOGRAPHIC_Y_COORDS: np.float32,
        TOPOGRAPHIC_DATA: np.float32,
    }

    for name, dtype in ds_dtypes.items():
        concat_ds[name] = concat_ds[name].astype(dtype)

    assert concat_ds["sat_data"].dtype == np.int16
    return concat_ds


def coord_to_range(
    da: xr.DataArray, dim: str, prefix: Optional[str], dtype=np.int32
) -> xr.DataArray:
    """
    TODO

    TODO: Actually, I think this is over-complicated?  I think we can
    just strip off the 'coord' from the dimension.

    """
    coord = da[dim]
    da[dim] = np.arange(len(coord), dtype=dtype)
    if prefix is not None:
        da[f"{prefix}_{dim}_coords"] = xr.DataArray(coord, coords=[da[dim]], dims=[dim])
    return da
