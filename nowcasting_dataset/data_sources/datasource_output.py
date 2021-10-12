""" General Data Source output pydantic class. """
from __future__ import annotations
import os
from nowcasting_dataset.filesystem.utils import make_folder
from nowcasting_dataset.utils import get_netcdf_filename

from pathlib import Path
from pydantic import BaseModel, Field
import pandas as pd
import xarray as xr
import numpy as np
from typing import List, Union
import logging
from datetime import datetime

from nowcasting_dataset.utils import to_numpy

logger = logging.getLogger(__name__)


class DataSourceOutput(BaseModel):
    """General Data Source output pydantic class.

    Data source output classes should inherit from this class
    """

    class Config:
        """ Allowed classes e.g. tensor.Tensor"""

        # TODO maybe there is a better way to do this
        arbitrary_types_allowed = True

    batch_size: int = Field(
        0,
        ge=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item i.e Example",
    )

    def get_name(self) -> str:
        """ Get the name of the class """
        return self.__class__.__name__.lower()

    def to_numpy(self):
        """Change to numpy"""
        for k, v in self.dict().items():
            self.__setattr__(k, to_numpy(v))

    def to_xr_data_array(self):
        """ Change to xr DataArray"""
        raise NotImplementedError()

    @staticmethod
    def create_batch_from_examples(data):
        """
        Join a list of data source items to a batch.

        Note that this only works for numpy objects, so objects are changed into numpy
        """
        _ = [d.to_numpy() for d in data]

        # use the first item in the list, and then update each item
        batch = data[0]
        for k in batch.dict().keys():

            # set batch size to the list of the items
            if k == "batch_size":
                batch.batch_size = len(data)
            else:

                # get list of one variable from the list of data items.
                one_variable_list = [d.__getattribute__(k) for d in data]
                batch.__setattr__(k, np.stack(one_variable_list, axis=0))

        return batch

    def split(self) -> List[DataSourceOutput]:
        """
        Split the datasource from a batch to a list of items

        Returns: List of single data source items
        """
        cls = self.__class__

        items = []
        for batch_idx in range(self.batch_size):
            d = {k: v[batch_idx] for k, v in self.dict().items() if k != "batch_size"}
            d["batch_size"] = 0
            items.append(cls(**d))

        return items

    def to_xr_dataset(self, **kwargs):
        """ Make a xr dataset. Each data source needs to define this """
        raise NotImplementedError

    def from_xr_dataset(self):
        """ Load from xr dataset. Each data source needs to define this """
        raise NotImplementedError

    def get_datetime_index(self):
        """ Datetime index for the data """
        pass

    def save_netcdf(self, batch_i: int, path: Path, xr_dataset: xr.Dataset):
        """
        Save batch to netcdf file

        Args:
            batch_i: the batch id, used to make the filename
            path: the path where it will be saved. This can be local or in the cloud.
            xr_dataset: xr dataset that has batch information in it
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

        encoding = {name: {"compression": "lzf"} for name in xr_dataset.data_vars}
        xr_dataset.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)

    def select_time_period(
        self,
        keys: List[str],
        history_minutes: int,
        forecast_minutes: int,
        t0_dt_of_first_example: Union[datetime, pd.Timestamp],
    ):
        """
        Selects a subset of data between the indicies of [start, end] for each key in keys

        Note that class is edited so nothing is returned.

        Args:
            keys: Keys in batch to use
            t0_dt_of_first_example: datetime of the current time (t0) in the first example of the batch
            history_minutes: How many minutes of history to use
            forecast_minutes: How many minutes of future data to use for forecasting

        """
        logger.debug(
            f"Taking a sub-selection of the batch data based on a history minutes of {history_minutes} "
            f"and forecast minutes of {forecast_minutes}"
        )

        start_time_of_first_batch = t0_dt_of_first_example - pd.to_timedelta(
            f"{history_minutes} minute 30 second"
        )
        end_time_of_first_example = t0_dt_of_first_example + pd.to_timedelta(
            f"{forecast_minutes} minute 30 second"
        )

        logger.debug(f"New start time for first batch is {start_time_of_first_batch}")
        logger.debug(f"New end time for first batch is {end_time_of_first_example}")

        start_time_of_first_example = to_numpy(start_time_of_first_batch)
        end_time_of_first_example = to_numpy(end_time_of_first_example)

        if self.get_datetime_index() is not None:

            time_of_first_example = to_numpy(pd.to_datetime(self.get_datetime_index()[0]))

            # find the start and end index, that we will then use to slice the data
            start_i, end_i = np.searchsorted(
                time_of_first_example, [start_time_of_first_example, end_time_of_first_example]
            )

            # slice all the data
            for key in keys:
                if "time" in self.__getattribute__(key).dims:
                    self.__setattr__(
                        key, self.__getattribute__(key).isel(time=slice(start_i, end_i))
                    )
                elif "time_30" in self.__getattribute__(key).dims:
                    self.__setattr__(
                        key, self.__getattribute__(key).isel(time_30=slice(start_i, end_i))
                    )

                logger.debug(f"{self.__class__.__name__} {key}: {self.__getattribute__(key).shape}")


def pad_nans(array, pad_width) -> np.ndarray:
    """ Pad nans with nans"""
    array = array.astype(np.float32)
    return np.pad(array, pad_width, constant_values=np.NaN)


def pad_data(
    data: DataSourceOutput,
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
