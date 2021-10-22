""" General Data Source output pydantic class. """
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import xarray as xr
from xarray.ufuncs import isnan, isinf
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


def pad_nans(array, pad_width) -> np.ndarray:
    """Pad nans with nans"""
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)


def check_nan_and_inf(data: xr.Dataset, class_name: str):
    """ Check that all values are non NaNs and not infinite"""

    if isnan(data).any():
        message = f"Some {class_name} data values are NaNs"
        logger.error(message)
        raise Exception(message)

    if isinf(data).any():
        message = f"Some {class_name} data values are Infinite"
        logger.error(message)
        raise Exception(message)


def check_dataset_greater_than(data: xr.Dataset, class_name: str, min_value: int):
    """ Check data is greater than a certain value """
    if (data < min_value).any():
        message = f"Some {class_name} data values are less than {min_value}"
        logger.error(message)
        raise Exception(message)


def check_dataset_less_than(data: xr.Dataset, class_name: str, max_value: int):
    """ Check data is less than a certain value """
    if (data > max_value).any():
        message = f"Some {class_name} data values are less than {max_value}"
        logger.error(message)
        raise Exception(message)


def check_dataset_not_equal(data: xr.Dataset, class_name: str, value: int):
    """ Check data is not equal than a certain value """
    if (data == value).any():
        message = f"Some {class_name} data values are equal to {value}"
        logger.error(message)
        raise Exception(message)
