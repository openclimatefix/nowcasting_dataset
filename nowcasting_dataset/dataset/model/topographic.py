from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch
from nowcasting_dataset.dataset.model.datasource import DataSource
from nowcasting_dataset.consts import Array


class Topographic(DataSource):

    # Shape: [batch_size,] seq_length, width, height, channel
    topo_data: Array = Field(
        ...,
        description="Elevation map of the area covered by the satellite data. "
        "Shape: [batch_size,] seq_length, width, height",
    )
    topo_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] width",
    )
    topo_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the topographic images. Shape: [batch_size,] width",
    )

    @property
    def height(self):
        """ The width of the topographic image """
        return self.topo_data.shape[-1]

    @property
    def width(self):
        """ The width of the topographic image """
        return self.topo_data.shape[-2]

    @validator("topo_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["topo_data"].shape[-2]

    @validator("topo_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["topo_data"].shape[-1]

    @staticmethod
    def fake(batch_size, seq_length_5, satellite_image_size_pixels):
        return Topographic(
            topo_data=torch.randn(
                batch_size,
                seq_length_5,
                satellite_image_size_pixels,
                satellite_image_size_pixels,
            ),
            topo_x_coords=torch.sort(torch.randn(batch_size, satellite_image_size_pixels))[0],
            topo_y_coords=torch.sort(
                torch.randn(batch_size, satellite_image_size_pixels), descending=True
            )[0],
        )
