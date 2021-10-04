from pydantic import BaseModel, Field, validator
from typing import Union, List
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSourceOutput
from nowcasting_dataset.consts import Array, SAT_VARIABLE_NAMES
from nowcasting_dataset.dataset.batch import coord_to_range


class Satellite(DataSourceOutput):

    # Shape: [batch_size,] seq_length, width, height, channel
    sat_data: Array = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )
    sat_x_coords: Array = Field(
        ...,
        description="aThe x (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] width",
    )
    sat_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] width",
    )

    sat_datetime_index: Array = Field(
        ...,
        description="Time index of satellite data at 5 minutes past the hour {0, 5, ..., 55}. "
        "*not* the {4, 9, ..., 59} timings of the satellite imagery. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.",
    )

    sat_channel_names: Union[List[List[str]], List[str], np.ndarray] = Field(
        ..., description="List of the satellite channels"
    )

    @validator("sat_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["sat_data"].shape[-3]
        return v

    @validator("sat_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["sat_data"].shape[-2]
        return v

    @staticmethod
    def fake(batch_size=32, seq_length_5=19, satellite_image_size_pixels=64, number_sat_channels=7):

        s = Satellite(
            batch_size=batch_size,
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
            sat_datetime_index=torch.sort(torch.randn(batch_size, seq_length_5))[0],
            sat_channel_names=[
                SAT_VARIABLE_NAMES[0:number_sat_channels] for _ in range(batch_size)
            ],
        )

        return s

    def to_xr_dataset(self):

        data = xr.DataArray(
            self.sat_data,
            coords={
                "time": self.sat_datetime_index,
                "x": self.sat_x_coords,
                "y": self.sat_y_coords,
                "variable": self.sat_channel_names,  # assuem all channels are the same
            },
        )

        ds = data.to_dataset(name="sat")

        for dim in ["time", "x", "y"]:
            ds = coord_to_range(ds, dim, prefix="sat")
        return ds.rename(
            {
                "variable": f"sat_variable",
                "x": f"sat_x",
                "y": f"sat_y",
            }
        )

    @staticmethod
    def from_xr_dataset(xr_dataset):

        return Satellite(
            batch_size=xr_dataset["sat"].shape[0],
            sat_data=xr_dataset["sat"],
            sat_x_coords=xr_dataset["sat"].sat_x,
            sat_y_coords=xr_dataset["sat"].sat_y,
            sat_datetime_index=xr_dataset["sat"].time,
            sat_channel_names=xr_dataset["sat"].sat_variable.values,
        )
