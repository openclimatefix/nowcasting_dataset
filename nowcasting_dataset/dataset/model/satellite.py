from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.model import Array, to_numpy


class Satellite(BaseModel):

    # Shape: [batch_size,] seq_length, width, height, channel
    sat_data: Array = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )
    sat_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] width",
    )
    sat_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] width",
    )

    @property
    def num_channels(self):
        """ The number of channels of the satellite image """
        return self.image_data.shape[-1]

    @property
    def height(self):
        """ The width of the satellite image """
        return self.image_data.shape[-2]

    @property
    def width(self):
        """ The width of the satellite image """
        return self.image_data.shape[-3]

    @validator("sat_data")
    def image_shape(cls, v):
        assert v.shape[-1] == cls.num_channels
        assert v.shape[-2] == cls.height
        assert v.shape[-3] == cls.width

    @validator("sat_x_coords")
    def x_coords_shape(cls, v):
        assert v.shape[-1] == cls.width

    @validator("sat_y_coords")
    def y_coords_shape(cls, v):
        assert v.shape[-1] == cls.height

    def to_numpy(self):
        """ Change to numpy """
        return Satellite(
            sat_data=to_numpy(self.sat_data),
            sat_x_coords=to_numpy(self.sat_x_coords),
            sat_y_coords=to_numpy(self.sat_y_coords),
        )
