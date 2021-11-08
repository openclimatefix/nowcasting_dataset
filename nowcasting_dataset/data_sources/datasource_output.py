""" General Data Source output pydantic class. """
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr
from xarray.ufuncs import isinf, isnan

from nowcasting_dataset.dataset.xr_utils import PydanticXArrayDataSet
from nowcasting_dataset.filesystem.utils import makedirs
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
        Save batch to netcdf file in path/<DataSourceOutputName>/.

        Args:
            batch_i: the batch id, used to make the filename
            path: the path where it will be saved. This can be local or in the cloud.
        """
        filename = get_netcdf_filename(batch_i)

        name = self.get_name()

        # make folder
        folder = os.path.join(path, name)
        if batch_i == 0:
            # only need to make the folder once, or check that the folder is there once
            makedirs(path=folder)

        # make file
        local_filename = os.path.join(folder, filename)

        encoding = {name: {"compression": "lzf"} for name in self.data_vars}
        self.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)

    def check_nan_and_inf(self, data: xr.Dataset, variable_name: str = None):
        """Check that all values are non NaNs and not infinite"""

        if isnan(data).any():
            message = f"Some {self.__class__.__name__} data values are NaNs"
            message += f" ({variable_name})" if variable_name is not None else None
            logger.error(message)
            raise Exception(message)

        if isinf(data).any():
            message = f"Some {self.__class__.__name__} data values are Infinite"
            message += f" ({variable_name})" if variable_name is not None else None
            logger.error(message)
            raise Exception(message)

    def check_dataset_greater_than_or_equal_to(
        self, data: xr.Dataset, min_value: int, variable_name: str = None
    ):
        """Check data is greater than a certain value"""
        if (data < min_value).any():
            message = f"Some {self.__class__.__name__} data values are less than {min_value}"
            message += f" ({variable_name})" if variable_name is not None else None
            logger.error(message)
            raise Exception(message)

    def check_dataset_less_than_or_equal_to(
        self, data: xr.Dataset, max_value: int, variable_name: str = None
    ):
        """Check data is less than a certain value"""
        if (data > max_value).any():
            message = f"Some {self.__class__.__name__} data values are less than {max_value}"
            message += f" ({variable_name})" if variable_name is not None else None
            logger.error(message)
            raise Exception(message)

    def check_dataset_not_equal(
        self, data: xr.Dataset, value: int, raise_error: bool = True, variable_name: str = None
    ):
        """Check data is not equal than a certain value"""
        if np.isclose(data, value).any():
            message = f"Some {self.__class__.__name__} data values are equal to {value}"
            message += f" ({variable_name})" if variable_name is not None else None
            if raise_error:
                logger.error(message)
                raise Exception(message)
            else:
                logger.warning(message)


def pad_nans(array, pad_width) -> np.ndarray:
    """Pad nans with nans"""
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)
