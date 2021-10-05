from pydantic import BaseModel, Field, validator
from typing import Union, List
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource_output import DataSourceOutput
from nowcasting_dataset.consts import Array, DATETIME_FEATURE_NAMES
from nowcasting_dataset.dataset.batch import coord_to_range
from nowcasting_dataset.time import make_time_vectors

# seems to be a pandas dataseries


class General(DataSourceOutput):

    t0_dt: Union[xr.DataArray, np.ndarray, torch.Tensor, int]  #: Shape: [batch_size,]
    x_meters_center: Union[xr.DataArray, np.ndarray, torch.Tensor, int]
    y_meters_center: Union[xr.DataArray, np.ndarray, torch.Tensor, int]
    object_at_center: Union[List[List[str]], List[str], str, np.ndarray, xr.DataArray]

    @staticmethod
    def fake(batch_size, t0_dt=None):

        if t0_dt is None:
            t0_dt, _, _ = make_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=0, seq_len_30_minutes=0
            )

        return General(
            batch_size=batch_size,
            t0_dt=t0_dt,
            x_meters_center=torch.randn(
                batch_size,
            ),
            y_meters_center=torch.randn(
                batch_size,
            ),
            object_at_center=np.array(["GSP"] * batch_size),
        )

    def to_xr_dataset(self, i):

        individual_datasets = []
        for name in ["t0_dt", "x_meters_center", "y_meters_center", "object_at_center"]:

            var = self.__getattribute__(name)

            example_dim = {"example": np.array([i], dtype=np.int32)}

            data = xr.DataArray([var], coords=example_dim, dims=["example"], name=name)

            ds = data.to_dataset()
            individual_datasets.append(ds)

        return xr.merge(individual_datasets)

    @staticmethod
    def from_xr_dataset(xr_dataset):

        return General(
            batch_size=xr_dataset["t0_dt"].shape[0],
            t0_dt=xr_dataset["t0_dt"],
            x_meters_center=xr_dataset["x_meters_center"],
            y_meters_center=xr_dataset["y_meters_center"],
            object_at_center=xr_dataset["object_at_center"],
        )
