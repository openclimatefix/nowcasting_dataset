from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSource
from nowcasting_dataset.consts import Array


class PV(DataSource):

    # Shape: [batch_size,] seq_length, width, height, channel
    pv_yield: Array = Field(
        ...,
        description=" PV yield from all PV systems in the region of interest (ROI). \
    : Includes central PV system, which will always be the first entry. \
    : shape = [batch_size, ] seq_length, n_pv_systems_per_example",
    )

    #: PV identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    pv_system_id: Array = Field(..., description="pv id fomr pvoutpu.org? TODO")
    pv_system_row_number: Array = Field(..., description="pv row number, made by OCF  TODO")

    pv_datetime_index: Array = Field(
        ...,
        description="The datetime associated with the pv system data. shape = [batch_size, ] sequence length,",
    )

    pv_system_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )
    pv_system_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )

    @property
    def number_of_pv_systems(self):
        """The number of pv systems"""
        return self.pv_yield.shape[-1]

    @property
    def sequence_length(self):
        """The equence length of the pv data"""
        return self.pv_yield.shape[-2]

    @validator("pv_system_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["pv_yield"].shape[-1]

    @validator("pv_system_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["pv_yield"].shape[-1]

    @staticmethod
    def fake(batch_size, seq_length_5, n_pv_systems_per_batch):
        return PV(
            pv_yield=torch.randn(
                batch_size,
                seq_length_5,
                n_pv_systems_per_batch,
            ),
            pv_system_id=torch.sort(torch.randint(10000, (batch_size, n_pv_systems_per_batch)))[0],
            pv_system_row_number=torch.sort(
                torch.randint(1000, (batch_size, n_pv_systems_per_batch))
            )[0],
            pv_datetime_index=torch.sort(torch.randn(batch_size, seq_length_5), descending=True)[0],
            pv_system_x_coords=torch.sort(torch.randn((batch_size, n_pv_systems_per_batch)))[0],
            pv_system_y_coords=torch.sort(
                torch.randn(batch_size, n_pv_systems_per_batch), descending=True
            )[0],
        )
