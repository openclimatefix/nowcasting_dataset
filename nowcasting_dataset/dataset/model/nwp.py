from pydantic import BaseModel, Field, validator
from typing import Union, List
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource_output import DataSourceOutput
from nowcasting_dataset.consts import (
    Array,
    NWP_VARIABLE_NAMES,
    NWP_DATA,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    NWP_TARGET_TIME,
)
from nowcasting_dataset.dataset.batch import coord_to_range


class NWP(DataSourceOutput):

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

    nwp_target_time: Array = Field(
        ...,
        description="Time index of nwp data at 5 minutes past the hour {0, 5, ..., 55}. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.",
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
        """The equence length of the pv data"""
        return self.nwp.shape[-3]

    @validator("nwp_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["nwp"].shape[-2]
        return v

    @validator("nwp_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["nwp"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5, nwp_image_size_pixels, number_nwp_channels):
        return NWP(
            batch_size=batch_size,
            nwp=torch.randn(
                batch_size,
                number_nwp_channels,
                seq_length_5,
                nwp_image_size_pixels,
                nwp_image_size_pixels,
            ),
            nwp_x_coords=torch.sort(torch.randn(batch_size, nwp_image_size_pixels))[0],
            nwp_y_coords=torch.sort(
                torch.randn(batch_size, nwp_image_size_pixels), descending=True
            )[0],
            nwp_target_time=torch.sort(torch.randn(batch_size, seq_length_5))[0],
            nwp_init_time=torch.sort(
                torch.randn(
                    batch_size,
                )
            )[0],
            nwp_channel_names=[
                NWP_VARIABLE_NAMES[0:number_nwp_channels] for _ in range(batch_size)
            ],
        )

    def to_xr_data_array(self):

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

    def to_xr_dataset(self):

        if type(self.nwp) != xr.DataArray:
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

        ds = self.nwp.to_dataset(name="nwp")
        ds["nwp"] = ds["nwp"].astype(np.float32)
        ds = ds.round(2)

        ds = ds.rename({"target_time": "time"})
        for dim in ["time", "x", "y"]:
            ds = coord_to_range(ds, dim, prefix="nwp")
        return ds.rename(
            {
                "variable": f"nwp_variable",
                "x": "nwp_x",
                "y": "nwp_y",
            }
        )

    @staticmethod
    def from_xr_dataset(xr_dataset):

        if NWP_DATA in xr_dataset.keys():
            return NWP(
                batch_size=xr_dataset[NWP_DATA].shape[0],
                nwp=xr_dataset[NWP_DATA],
                nwp_channel_names=xr_dataset[NWP_DATA].nwp_variable.values,
                nwp_init_time=xr_dataset[NWP_DATA].init_time,
                nwp_target_time=xr_dataset[NWP_DATA].time,
                nwp_x_coords=xr_dataset[NWP_DATA].nwp_x,
                nwp_y_coords=xr_dataset[NWP_DATA].nwp_y,
            )
        else:
            return None
