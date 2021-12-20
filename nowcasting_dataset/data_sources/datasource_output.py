""" General Data Source output pydantic class. """
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

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

    def save_netcdf(self, batch_i: int, path: Path, add_data_source_name_to_path: bool = True):
        """
        Save batch to netcdf file in path/<DataSourceOutputName>/.

        Args:
            batch_i: the batch id, used to make the filename
            path: the path where it will be saved. This can be local or in the cloud.
            add_data_source_name_to_path: add data source path to 'path'
        """
        filename = get_netcdf_filename(batch_i)

        if add_data_source_name_to_path:
            name = self.get_name()
            path = os.path.join(path, name)

        # make folder
        if batch_i == 0:
            # only need to make the folder once, or check that the folder is there once
            makedirs(path=path)

        # make file
        local_filename = os.path.join(path, filename)

        encoding = {name: {"compression": "lzf"} for name in self.data_vars}
        self.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)

    def check_nan_and_inf(self, data: xr.Dataset, variable_name: str = None):
        """Check that all values are non NaNs and not infinite"""

        if np.isnan(data).any():
            message = f"Some {self.__class__.__name__} data values are NaNs"
            if variable_name is not None:
                message += f" ({variable_name})"
            logger.error(message)
            raise Exception(message)

        if np.isinf(data).any():
            message = f"Some {self.__class__.__name__} data values are Infinite"
            if variable_name is not None:
                message += f" ({variable_name})"
            logger.error(message)
            raise Exception(message)

    def check_nan_and_fill_warning(self, data: xr.Dataset, variable_name: str = None) -> xr.Dataset:
        """Check that all values are non NaNs and not infinite"""

        if np.isnan(data).any():
            message = f"Some {self.__class__.__name__} data values are NaNs"
            if variable_name is not None:
                message += f" ({variable_name})"
            logger.warning(message)
            data = data.fillna(0)

        return data

    def check_dataset_greater_than_or_equal_to(
        self, data: xr.Dataset, min_value: int, variable_name: str = None
    ):
        """Check data is greater than a certain value"""
        if (data < min_value).any():
            message = f"Some {self.__class__.__name__} data values are less than {min_value}"
            if variable_name is not None:
                message += f" ({variable_name})"
            logger.error(message)
            raise Exception(message)

    def check_dataset_less_than_or_equal_to(
        self, data: xr.Dataset, max_value: int, variable_name: str = None
    ):
        """Check data is less than a certain value"""
        if (data > max_value).any():
            message = f"Some {self.__class__.__name__} data values are less than {max_value}"
            if variable_name is not None:
                message += f" ({variable_name})"
            logger.error(message)
            raise Exception(message)

    def check_dataset_not_equal(
        self, data: xr.Dataset, value: int, raise_error: bool = True, variable_name: str = None
    ):
        """Check data is not equal than a certain value"""
        if np.isclose(data, value).any():
            message = f"Some {self.__class__.__name__} data values are equal to {value}"
            if variable_name is not None:
                message += f" ({variable_name})"
            if raise_error:
                logger.error(message)
                raise Exception(message)
            else:
                logger.warning(message)

    def check_data_var_dim(self, data: xr.Dataset, expected_dims: Tuple[str]):
        """Check the data var has the correct dims"""

        actual_dims = data.dims
        # check the dims are the same as expected,
        # we are using 'set' so ordering doesnt matter
        if set(actual_dims) != set(expected_dims):
            message = (
                f"Actual dims {actual_dims} does not equal {expected_dims} "
                f"for {data.name} {self.__class__.__name__}"
            )
            logger.error(message)
            raise Exception(message)


def pad_nans(array, pad_width) -> np.ndarray:
    """Pad nans with nans"""
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)
