from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource_output import DataSourceOutput
from nowcasting_dataset.consts import Array, SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from nowcasting_dataset.dataset.batch import coord_to_range


class Sun(DataSourceOutput):

    sun_azimuth_angle: Array = Field(
        ...,
        description="PV azimuth angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )

    sun_elevation_angle: Array = Field(
        ...,
        description="PV azimuth angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )
    sun_datetime_index: Array

    @validator("sun_elevation_angle")
    def elevation_shape(cls, v, values):
        assert v.shape[-1] == values["sun_azimuth_angle"].shape[-1]
        return v

    @validator("sun_datetime_index")
    def sun_datetime_index_shape(cls, v, values):
        assert v.shape[-1] == values["sun_azimuth_angle"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5):
        return Sun(
            batch_size=batch_size,
            sun_azimuth_angle=torch.randn(
                batch_size,
                seq_length_5,
            ),
            sun_elevation_angle=torch.randn(
                batch_size,
                seq_length_5,
            ),
            sun_datetime_index=torch.sort(torch.randn(batch_size, seq_length_5), descending=True)[
                0
            ],
        )

    def to_xr_dataset(self):

        individual_datasets = []
        for name in [SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE]:

            var = self.__getattribute__(name)

            data = xr.DataArray(
                var,
                dims=["time"],
                coords={"time": self.sun_datetime_index},
                name=name,
            )

            ds = data.to_dataset()
            ds = coord_to_range(ds, "time", prefix=None)
            individual_datasets.append(ds)

        return xr.merge(individual_datasets)

    @staticmethod
    def from_xr_dataset(xr_dataset):

        return Sun(
            batch_size=xr_dataset[SUN_AZIMUTH_ANGLE].shape[0],
            sun_azimuth_angle=xr_dataset[SUN_AZIMUTH_ANGLE],
            sun_elevation_angle=xr_dataset[SUN_ELEVATION_ANGLE],
            sun_datetime_index=xr_dataset[SUN_AZIMUTH_ANGLE].time,
        )
