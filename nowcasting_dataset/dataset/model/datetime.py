from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource import DataSource
from nowcasting_dataset.consts import Array


class Datetime(DataSource):

    hour_of_day_sin: Array  #: Shape: [batch_size,] seq_length
    hour_of_day_cos: Array
    day_of_year_sin: Array
    day_of_year_cos: Array

    @property
    def sequence_length(self):
        """The equence length of the pv data"""
        return self.hour_of_day_sin.shape[-1]

    @validator("hour_of_day_cos")
    def v_hour_of_day_cos(cls, v, values):
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]

    @validator("day_of_year_sin")
    def v_day_of_year_sin(cls, v, values):
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]

    @validator("day_of_year_cos")
    def v_day_of_year_cos(cls, v, values):
        assert v.shape[-1] == values["hour_of_day_sin"].shape[-1]

    @staticmethod
    def fake(batch_size, seq_length_5):
        return Datetime(
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
        )
