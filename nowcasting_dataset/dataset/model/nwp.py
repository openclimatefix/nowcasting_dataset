from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSource
from nowcasting_dataset.consts import Array


class NWP(DataSource):

    # Shape: [batch_size,] seq_length, width, height, channel
    nwp: Array = Field(
        ...,
        description=" Numerical weather predictions (NWPs) \
    : Shape: [batch_size,] channel, seq_length, width, height",
    )

    nwp_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )
    nwp_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )

    nwp_target_time: Array = None  # TODO

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
        """The equence length of the pv data"""
        return self.nwp.shape[-3]

    @validator("nwp_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["nwp"].shape[-2]

    @validator("nwp_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["nwp"].shape[-3]

    @staticmethod
    def fake(batch_size, seq_length_5, nwp_image_size_pixels, number_nwp_channels):
        return NWP(
            nwp=torch.randn(
                batch_size,
                seq_length_5,
                nwp_image_size_pixels,
                nwp_image_size_pixels,
                number_nwp_channels,
            ),
            nwp_x_coords=torch.sort(torch.randn(batch_size, nwp_image_size_pixels))[0],
            nwp_y_coords=torch.sort(
                torch.randn(batch_size, nwp_image_size_pixels), descending=True
            )[0],
        )
