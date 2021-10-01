from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSource
from nowcasting_dataset.consts import Array


class Sun(DataSource):

    azimuth_data: Array = Field(
        ...,
        description="PV azimuth angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )

    elevation_data: Array = Field(
        ...,
        description="PV azimuth angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )

    @validator("elevation_data")
    def elevation_shape(cls, v, values):
        assert v.shape[-1] == values["azimuth_data"].shape[-1]

    @staticmethod
    def fake(batch_size, seq_length_5):
        return Sun(
            azimuth_data=torch.randn(
                batch_size,
                seq_length_5,
            ),
            elevation_data=torch.randn(
                batch_size,
                seq_length_5,
            ),
        )
