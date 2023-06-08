from typing import Union

import numpy as np
import torch
import xarray as xr
from pydantic import BaseModel, Field

from nowcasting_dataset.config.model import Configuration

Array = Union[xr.DataArray, np.ndarray, torch.Tensor]


class Satellite(BaseModel):

    # width: int = Field(..., g=0, description="The width of the satellite image")
    # height: int = Field(..., g=0, description="The width of the satellite image")
    # num_channels: int = Field(..., g=0, description="The width of the satellite image")

    # Shape: [batch_size,] seq_length, width, height, channel
    image_data: Array = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )
    x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] width",
    )
    y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] width",
    )

    # @validator("sat_data")
    # def image_shape(cls, v):
    #     assert v.shape[-1] == cls.num_channels
    #     assert v.shape[-2] == cls.height
    #     assert v.shape[-3] == cls.width
    #
    # @validator("x_coords")
    # def x_coords_shape(cls, v):
    #     assert v.shape[-1] == cls.width
    #
    # @validator("y_coords")
    # def y_coords_shape(cls, v):
    #     assert v.shape[-1] == cls.height
    #
    class Config:
        arbitrary_types_allowed = True


class Batch(BaseModel):

    batch_size: int = Field(
        ...,
        g=0,
        description="The size of this batch. If the batch size is 0, "
        "then this item stores one data item",
    )

    satellite: Satellite


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, configuration: Configuration = Configuration(), length: int = 10):
        """
        Init

        Args:
            configuration: configuration object
            length: length of dataset
        """
        self.batch_size = configuration.process.batch_size
        self.seq_length_5 = (
            configuration.process.seq_length_5_minutes
        )  # the sequence data in 5 minute steps
        self.seq_length_30 = (
            configuration.process.seq_length_30_minutes
        )  # the sequence data in 30 minute steps
        self.satellite_image_size_pixels = configuration.process.satellite_image_size_pixels
        self.nwp_image_size_pixels = configuration.process.nwp_image_size_pixels
        self.number_sat_channels = len(configuration.process.sat_channels)
        self.number_nwp_channels = len(configuration.process.nwp_channels)
        self.length = length

    def __len__(self):
        """Number of pieces of data"""
        return self.length

    def per_worker_init(self, worker_id: int):
        """Not needed"""
        pass

    def __getitem__(self, idx):
        """
        Get item, use for iter and next method

        Args:
            idx: batch index

        Returns: Dictionary of random data

        """

        sat = Satellite(
            image_data=np.random.randn(
                self.batch_size,
                self.seq_length_5,
                self.satellite_image_size_pixels,
                self.satellite_image_size_pixels,
                self.number_sat_channels,
            ),
            x_coords=torch.sort(torch.randn(self.batch_size, self.satellite_image_size_pixels))[0],
            y_coords=torch.sort(
                torch.randn(self.batch_size, self.satellite_image_size_pixels), descending=True
            )[0],
        )

        # Note need to return as nested dict
        return Batch(satellite=sat, batch_size=self.batch_size).dict()


train = torch.utils.data.DataLoader(FakeDataset())
i = iter(train)
x = next(i)

x = Batch(**x)
# IT WORKS
assert type(x.satellite.image_data) == torch.Tensor
