""" General Data Source output pydantic class. """
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel, Field

from nowcasting_dataset.dataset.xr_utils import PydanticXArrayDataSet
from nowcasting_dataset.filesystem.utils import make_folder
from nowcasting_dataset.utils import get_netcdf_filename

logger = logging.getLogger(__name__)


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
