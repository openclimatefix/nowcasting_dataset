""" General Data Source output pydantic class. """
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field

from nowcasting_dataset.dataset.pydantic_xr import PydanticXArrayDataSet
from nowcasting_dataset.dataset.xr_utils import convert_data_array_to_dataset
from nowcasting_dataset.filesystem.utils import make_folder
from nowcasting_dataset.utils import get_netcdf_filename

logger = logging.getLogger(__name__)


def create_image_array(
    dims=("time", "x", "y", "channels"),
    seq_length_5=19,
    image_size_pixels=64,
    number_channels=7,
):
    """ Create Satellite or NWP fake image data"""
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq="5T", periods=seq_length_5),
        "x": np.random.randint(low=0, high=1000, size=image_size_pixels),
        "y": np.random.randint(low=0, high=1000, size=image_size_pixels),
        "channels": np.arange(number_channels),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    image_data_array = xr.DataArray(
        abs(
            np.random.randn(
                seq_length_5,
                image_size_pixels,
                image_size_pixels,
                number_channels,
            )
        ),
        coords=coords,
    )  # Fake data for testing!
    return image_data_array


def create_gsp_pv_dataset(
    dims=("time", "id"),
    freq="5T",
    seq_length=19,
    number_of_systems=128,
):
    """ Create gsp or pv fake dataset """
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq=freq, periods=seq_length),
        "id": np.random.randint(low=0, high=1000, size=number_of_systems),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    data_array = xr.DataArray(
        np.random.randn(
            seq_length,
            number_of_systems,
        ),
        coords=coords,
    )  # Fake data for testing!

    data = convert_data_array_to_dataset(data_array)

    x_coords = xr.DataArray(
        data=np.sort(np.random.randn(number_of_systems)),
        dims=["id_index"],
        coords=dict(
            id_index=range(number_of_systems),
        ),
    )

    y_coords = xr.DataArray(
        data=np.sort(np.random.randn(number_of_systems)),
        dims=["id_index"],
        coords=dict(
            id_index=range(number_of_systems),
        ),
    )

    data["x_coords"] = x_coords
    data["y_coords"] = y_coords

    return data


def create_sun_dataset(
    dims=("time",),
    freq="5T",
    seq_length=19,
) -> xr.Dataset:
    """
    Create sun fake dataset

    Args:
        dims: # TODO
        freq: # TODO
        seq_length: # TODO

    Returns: # TODO

    """
    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq=freq, periods=seq_length),
    }
    coords = [(dim, ALL_COORDS[dim]) for dim in dims]
    data_array = xr.DataArray(
        np.random.randn(
            seq_length,
        ),
        coords=coords,
    )  # Fake data for testing!

    data = convert_data_array_to_dataset(data_array)
    sun = data.rename({"data": "elevation"})
    sun["azimuth"] = data.data

    return sun


def create_metadata_dataset() -> xr.Dataset:
    """ Create fake metadata dataset"""
    d = {
        "dims": ("t0_dt",),
        "data": pd.date_range("2021-01-01", freq="5T", periods=1) + pd.Timedelta("30T"),
    }

    data = convert_data_array_to_dataset(xr.DataArray.from_dict(d))

    for v in ["x_meters_center", "y_meters_center", "object_at_center_label"]:
        d: dict = {"dims": ("t0_dt",), "data": [np.random.randint(0, 1000)]}
        d: xr.Dataset = convert_data_array_to_dataset(xr.DataArray.from_dict(d)).rename({"data": v})
        data[v] = getattr(d, v)

    return data


def create_datetime_dataset(
    seq_length=19,
) -> xr.Dataset:
    """ Create fake datetime dataset"""

    ALL_COORDS = {
        "time": pd.date_range("2021-01-01", freq="5T", periods=seq_length),
    }
    coords = [("time", ALL_COORDS["time"])]
    data_array = xr.DataArray(
        np.random.randn(
            seq_length,
        ),
        coords=coords,
    )  # Fake data

    data = convert_data_array_to_dataset(data_array)

    ds = data.rename({"data": "day_of_year_cos"})
    ds["day_of_year_sin"] = data.rename({"data": "day_of_year_sin"}).day_of_year_sin
    ds["hour_of_day_cos"] = data.rename({"data": "hour_of_day_cos"}).hour_of_day_cos
    ds["hour_of_day_sin"] = data.rename({"data": "hour_of_day_sin"}).hour_of_day_sin

    return data


class DataSourceOutput(PydanticXArrayDataSet):
    """General Data Source output pydantic class.

    Data source output classes should inherit from this class
    """

    __slots__ = []

    def get_name(self) -> str:
        """Get the name of the class"""
        return self.__class__.__name__.lower()

    def save_netcdf(self, batch_i: int, path: Path):
        """
        Save batch to netcdf file

        Args:
            batch_i: the batch id, used to make the filename
            path: the path where it will be saved. This can be local or in the cloud.
        """
        filename = get_netcdf_filename(batch_i)

        name = self.get_name()

        # make folder
        folder = os.path.join(path, name)
        if batch_i == 0:
            # only need to make the folder once, or check that there folder is there once
            make_folder(path=folder)

        # make file
        local_filename = os.path.join(folder, filename)

        encoding = {name: {"compression": "lzf"} for name in self.data_vars}
        self.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)

    # def from_xr_dataset(self):
    #     pass


class DataSourceOutputML(BaseModel):
    """General Data Source output pydantic class.

    Data source output classes should inherit from this class
    """

    class Config:
        """Allowed classes e.g. tensor.Tensor"""

        # TODO maybe there is a better way to do this
        arbitrary_types_allowed = True

    batch_size: int = Field(
        0,
        ge=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item i.e Example",
    )

    def get_name(self) -> str:
        """Get the name of the class"""
        return self.__class__.__name__.lower()

    def get_datetime_index(self):
        """Datetime index for the data"""
        pass


def pad_nans(array, pad_width) -> np.ndarray:
    """Pad nans with nans"""
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)


def pad_data(
    data: DataSourceOutputML,
    pad_size: int,
    one_dimensional_arrays: List[str],
    two_dimensional_arrays: List[str],
):
    """
    Pad (if necessary) so returned arrays are always of size

    data has two types of arrays in it, one dimensional arrays and two dimensional arrays
    the one dimensional arrays are padded in that dimension
    the two dimensional arrays are padded in the second dimension

    Note that class is edited so nothing is returned.

    Args:
        data: typed dictionary of data objects
        pad_size: the maount that should be padded
        one_dimensional_arrays: list of data items that should be padded by one dimension
        two_dimensional_arrays: list of data tiems that should be padded in the third dimension (and more)

    """
    # Pad (if necessary) so returned arrays are always of size
    pad_shape = (0, pad_size)  # (before, after)

    for name in one_dimensional_arrays:
        data.__setattr__(name, pad_nans(data.__getattribute__(name), pad_width=pad_shape))

    for variable in two_dimensional_arrays:
        data.__setattr__(
            variable, pad_nans(data.__getattribute__(variable), pad_width=((0, 0), pad_shape))
        )  # (axis0, axis1)
