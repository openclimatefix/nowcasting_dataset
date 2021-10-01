from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSource
from nowcasting_dataset.consts import Array


class Satellite(DataSource):

    # Shape: [batch_size,] seq_length, width, height, channel
    sat_data: Union[xr.DataArray, np.ndarray, torch.Tensor] = Field(
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

    # sat_datetime_index: Array = None  # TODO

    @validator("sat_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["sat_data"].shape[-3]

    @validator("sat_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["sat_data"].shape[-2]

    @staticmethod
    def fake(batch_size, seq_length_5, satellite_image_size_pixels, number_sat_channels):

        return Satellite(
            sat_data=torch.randn(
                batch_size,
                seq_length_5,
                satellite_image_size_pixels,
                satellite_image_size_pixels,
                number_sat_channels,
            ),
            sat_x_coords=torch.sort(torch.randn(batch_size, satellite_image_size_pixels))[0],
            sat_y_coords=torch.sort(
                torch.randn(batch_size, satellite_image_size_pixels), descending=True
            )[0],
        )
