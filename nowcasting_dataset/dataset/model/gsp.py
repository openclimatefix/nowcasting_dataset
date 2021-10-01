from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSource
from nowcasting_dataset.consts import Array


class GSP(DataSource):

    # Shape: [batch_size,] seq_length, width, height, channel
    gsp_yield: Array = Field(
        ...,
        description=" GSP yield from all GSP in the region of interest (ROI). \
    : Includes central GSP system, which will always be the first entry. \
    : shape = [batch_size, ] seq_length, n_gsp_per_example",
    )

    #: GSP identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    gsp_id: Array = Field(..., description="gsp id fomr NG")

    gsp_datetime_index: Array = Field(
        ...,
        description="The datetime associated with the gsp data. shape = [batch_size, ] sequence length,",
    )

    gsp_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the gsp. "
        "Shape: [batch_size,] n_gsp_per_example",
    )
    gsp_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the gsp. "
        "Shape: [batch_size,] n_gsp_per_example",
    )

    @property
    def number_of_gsp(self):
        """The number of pv systems"""
        return self.gsp_yield.shape[-1]

    @property
    def sequence_length(self):
        """The equence length of the pv data"""
        return self.gsp_yield.shape[-2]

    @validator("gsp_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["gsp_yield"].shape[-1]

    @validator("gsp_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["gsp_yield"].shape[-1]

    @staticmethod
    def fake(batch_size, seq_length_30, n_gsp_per_batch):
        return GSP(
            gsp_yield=torch.randn(
                batch_size,
                seq_length_30,
                n_gsp_per_batch,
            ),
            gsp_id=torch.sort(torch.randint(340, (batch_size, n_gsp_per_batch)))[0],
            gsp_datetime_index=torch.sort(torch.randn(batch_size, seq_length_30), descending=True)[
                0
            ],
            gsp_x_coords=torch.sort(torch.randn(batch_size, n_gsp_per_batch))[0],
            gsp_y_coords=torch.sort(torch.randn(batch_size, n_gsp_per_batch), descending=True)[0],
        )
