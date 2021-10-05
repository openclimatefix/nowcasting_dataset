from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource_output import DataSourceOutput, pad_data
from nowcasting_dataset.consts import (
    Array,
    PV_YIELD,
    PV_DATETIME_INDEX,
    PV_SYSTEM_Y_COORDS,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_ID,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
)


class PV(DataSourceOutput):

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
        return v

    @validator("pv_system_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["pv_yield"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5, n_pv_systems_per_batch):
        return PV(
            batch_size=batch_size,
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

    def pad(self, n_pv_systems_per_example: int = DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE):

        pad_size = n_pv_systems_per_example - self.pv_yield.shape[-1]
        # Pad (if necessary) so returned arrays are always of size
        pad_shape = (0, pad_size)  # (before, after)

        one_dimensional_arrays = [
            PV_SYSTEM_ID,
            PV_SYSTEM_ROW_NUMBER,
            PV_SYSTEM_X_COORDS,
            PV_SYSTEM_Y_COORDS,
        ]

        pad_nans_variables = [PV_YIELD]

        pad_data(
            data=self,
            pad_size=pad_size,
            one_dimensional_arrays=one_dimensional_arrays,
            two_dimensional_arrays=pad_nans_variables,
        )

    def to_xr_dataset(self, i):

        assert self.batch_size == 0

        example_dim = {"example": np.array([i], dtype=np.int32)}

        # PV
        one_dataset = xr.DataArray(self.pv_yield, dims=["time", "pv_system"])
        one_dataset = one_dataset.to_dataset(name="pv_yield")
        n_pv_systems = len(self.pv_system_id)

        # 1D
        for name in [
            "pv_system_id",
            "pv_system_row_number",
            "pv_system_x_coords",
            "pv_system_y_coords",
        ]:
            var = self.__getattribute__(name)

            one_dataset[name] = xr.DataArray(
                var[None, :],
                coords={
                    **example_dim,
                    **{"pv_system": np.arange(n_pv_systems, dtype=np.int32)},
                },
                dims=["example", "pv_system"],
            )

        return one_dataset

    @staticmethod
    def from_xr_dataset(xr_dataset):

        return PV(
            batch_size=xr_dataset[PV_YIELD].shape[0],
            pv_yield=xr_dataset[PV_YIELD],
            pv_system_id=xr_dataset[PV_SYSTEM_ID],
            pv_system_row_number=xr_dataset[PV_SYSTEM_ROW_NUMBER],
            pv_datetime_index=xr_dataset[PV_YIELD].time,
            pv_system_x_coords=xr_dataset[PV_SYSTEM_X_COORDS],
            pv_system_y_coords=xr_dataset[PV_SYSTEM_Y_COORDS],
        )
