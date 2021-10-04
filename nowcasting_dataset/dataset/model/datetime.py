from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSourceOutput
from nowcasting_dataset.consts import Array, DATETIME_FEATURE_NAMES
from nowcasting_dataset.dataset.batch import coord_to_range

# seems to be a pandas dataseries


class Datetime(DataSourceOutput):

    hour_of_day_sin: Array  #: Shape: [batch_size,] seq_length
    hour_of_day_cos: Array
    day_of_year_sin: Array
    day_of_year_cos: Array
    datetime_index: Array

    @property
    def sequence_length(self):
        """The equence length of the pv data"""
        return self.hour_of_day_sin.shape[-1]

    @validator("hour_of_day_cos")
    def v_hour_of_day_cos(cls, v, values):
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]
        return v

    @validator("day_of_year_sin")
    def v_day_of_year_sin(cls, v, values):
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]
        return v

    @validator("day_of_year_cos")
    def v_day_of_year_cos(cls, v, values):
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5):
        return Datetime(
            batch_size=batch_size,
            hour_of_day_sin=torch.randn(
                batch_size,
                seq_length_5,
            ),
            hour_of_day_cos=torch.randn(
                batch_size,
                seq_length_5,
            ),
            day_of_year_sin=torch.randn(
                batch_size,
                seq_length_5,
            ),
            day_of_year_cos=torch.randn(
                batch_size,
                seq_length_5,
            ),
            datetime_index=torch.sort(torch.randn(batch_size, seq_length_5), descending=True)[0],
        )

    def to_xr_dataset(self):

        individual_datasets = []
        for name in DATETIME_FEATURE_NAMES:

            var = self.__getattribute__(name)

            data = xr.DataArray(
                var,
                dims=["time"],
                coords={"time": self.datetime_index},
                name=name,
            )

            ds = data.to_dataset()
            ds = coord_to_range(ds, "time", prefix=None)
            individual_datasets.append(ds)

        return xr.merge(individual_datasets)

    @staticmethod
    def from_xr_dataset(xr_dataset):

        return Datetime(
            batch_size=xr_dataset["hour_of_day_sin"].shape[0],
            hour_of_day_sin=xr_dataset["hour_of_day_sin"],
            hour_of_day_cos=xr_dataset["hour_of_day_cos"],
            day_of_year_sin=xr_dataset["day_of_year_sin"],
            day_of_year_cos=xr_dataset["day_of_year_cos"],
            datetime_index=xr_dataset["hour_of_day_sin"].time,
        )