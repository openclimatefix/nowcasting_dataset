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


def pad_nans(array, pad_width) -> np.ndarray:
    """Pad nans with nans"""
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)
