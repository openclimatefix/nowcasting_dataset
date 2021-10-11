""" Model for output of NWP data """
from pydantic import Field, validator
from typing import Union, List
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.consts import (
    Array,
    NWP_VARIABLE_NAMES,
    NWP_DATA,
)
from nowcasting_dataset.utils import coord_to_range
from nowcasting_dataset.time import make_random_time_vectors
import logging

logger = logging.getLogger(__name__)


class NWP(DataSourceOutput):
    """ Model for output of NWP data """

    # Shape: [batch_size,] seq_length, width, height, channel
    nwp: Array = Field(
        ...,
        description=" Numerical weather predictions (NWPs) \
    : Shape: [batch_size,] channel, seq_length, width, height",
    )

    nwp_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the NWP data. "
        "Shape: [batch_size,] width",
    )
    nwp_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the NWP data. "
        "Shape: [batch_size,] height",
    )

    nwp_target_time: Array = Field(
        ...,
        description="Time index of nwp data at 5 minutes past the hour {0, 5, ..., 55}. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.  The 'target time' is the time the NWP is _about_.",
    )

    nwp_init_time: Union[xr.DataArray, np.ndarray, torch.Tensor, int] = Field(
        ..., description="The time when the nwp forecast was made"
    )

    nwp_channel_names: Union[List[List[str]], List[str], np.ndarray] = Field(
        ..., description="List of the nwp channels"
    )

    @property
    def width(self):
        """The width of the nwp data"""
        return self.nwp.shape[-2]

    @property
    def height(self):
        """The width of the nwp data"""
        return self.nwp.shape[-1]

    @property
    def sequence_length(self):
        """The sequence length of the NWP timeseries"""
        return self.nwp.shape[-3]

    @validator("nwp_x_coords")
    def x_coordinates_shape(cls, v, values):
        """ Validate 'nwp_x_coords' """
        assert v.shape[-1] == values["nwp"].shape[-2]
        return v

    @validator("nwp_y_coords")
    def y_coordinates_shape(cls, v, values):
        """ Validate 'nwp_y_coords' """
        assert v.shape[-1] == values["nwp"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5, nwp_image_size_pixels, number_nwp_channels, time_5=None):
        """ Create fake data """
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=seq_length_5, seq_len_30_minutes=0
            )

        return NWP(
            batch_size=batch_size,
            nwp=np.random.randn(
                batch_size,
                number_nwp_channels,
                seq_length_5,
                nwp_image_size_pixels,
                nwp_image_size_pixels,
            ),
            nwp_x_coords=np.sort(np.random.randn(batch_size, nwp_image_size_pixels)),
            nwp_y_coords=np.sort(np.random.randn(batch_size, nwp_image_size_pixels))[
                :, ::-1
            ].copy(),
            # copy is needed as torch doesnt not support negative strides
            nwp_target_time=time_5,
            nwp_init_time=np.sort(
                np.random.randn(
                    batch_size,
                )
            ),
            nwp_channel_names=[
                NWP_VARIABLE_NAMES[0:number_nwp_channels] for _ in range(batch_size)
            ],
        )

    def get_datetime_index(self) -> Array:
        """ Get the datetime index of this data """
        return self.nwp_target_time

    def to_xr_data_array(self):
        """ Change to data_array.  Sets the nwp field in-place."""
        self.nwp = xr.DataArray(
            self.nwp,
            dims=["variable", "target_time", "x", "y"],
            coords={
                "variable": self.nwp_channel_names,
                "target_time": self.nwp_target_time,
                "init_time": self.nwp_init_time,
                "x": self.nwp_x_coords,
                "y": self.nwp_y_coords,
            },
        )

    def to_xr_dataset(self, i):
        """ Make a xr dataset """
        logger.debug(f"Making xr dataset for batch {i}")
        if type(self.nwp) != xr.DataArray:
            self.to_xr_data_array()

        ds = self.nwp.to_dataset(name="nwp")
        ds["nwp"] = ds["nwp"].astype(np.float32)
        ds = ds.round(2)

        ds = ds.rename({"target_time": "time"})
        for dim in ["time", "x", "y"]:
            ds = coord_to_range(ds, dim, prefix="nwp")
        ds = ds.rename(
            {
                "variable": f"nwp_variable",
                "x": "nwp_x",
                "y": "nwp_y",
            }
        )

        ds["nwp_x_coords"] = ds["nwp_x_coords"].astype(np.float32)
        ds["nwp_y_coords"] = ds["nwp_y_coords"].astype(np.float32)

        return ds

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if NWP_DATA in xr_dataset.keys():
            return NWP(
                batch_size=xr_dataset[NWP_DATA].shape[0],
                nwp=xr_dataset[NWP_DATA],
                nwp_channel_names=xr_dataset[NWP_DATA].nwp_variable.values,
                nwp_init_time=xr_dataset[NWP_DATA].init_time,
                nwp_target_time=xr_dataset["nwp_time_coords"],
                nwp_x_coords=xr_dataset[NWP_DATA].nwp_x,
                nwp_y_coords=xr_dataset[NWP_DATA].nwp_y,
            )
        else:
            return None
