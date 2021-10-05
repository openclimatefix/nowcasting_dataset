from pydantic import BaseModel, Field, validator
from typing import Union
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.dataset.model.datasource_output import DataSourceOutput, pad_data
from nowcasting_dataset.consts import Array

from nowcasting_dataset.consts import (
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_DATETIME_INDEX,
    DEFAULT_N_GSP_PER_EXAMPLE,
)
from nowcasting_dataset.time import make_time_vectors


class GSP(DataSourceOutput):

    # Shape: [batch_size,] seq_length, width, height, channel
    gsp_yield: Array = Field(
        ...,
        description=" GSP yield from all GSP in the region of interest (ROI). \
    : Includes central GSP system, which will always be the first entry. \
    : shape = [batch_size, ] seq_length, n_gsp_per_example",
    )

    #: GSP identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    gsp_id: Array = Field(..., description="gsp id from NG")

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

    @validator("gsp_yield")
    def gsp_yield_shape(cls, v, values):
        if values["batch_size"] > 0:
            assert len(v.shape) == 3
        else:
            assert len(v.shape) == 2
        return v

    @validator("gsp_x_coords")
    def x_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["gsp_yield"].shape[-1]
        return v

    @validator("gsp_y_coords")
    def y_coordinates_shape(cls, v, values):
        assert v.shape[-1] == values["gsp_yield"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_30, n_gsp_per_batch, time_30=None):

        if time_30 is None:
            _, _, time_30 = make_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=0, seq_len_30_minutes=seq_length_30
            )

        return GSP(
            batch_size=batch_size,
            gsp_yield=torch.randn(
                batch_size,
                seq_length_30,
                n_gsp_per_batch,
            ),
            gsp_id=torch.sort(torch.randint(340, (batch_size, n_gsp_per_batch)))[0],
            gsp_datetime_index=time_30,
            gsp_x_coords=torch.sort(torch.randn(batch_size, n_gsp_per_batch))[0],
            gsp_y_coords=torch.sort(torch.randn(batch_size, n_gsp_per_batch), descending=True)[0],
        )

    def pad(self, n_gsp_per_example: int = DEFAULT_N_GSP_PER_EXAMPLE):

        pad_size = n_gsp_per_example - self.gsp_yield.shape[-1]
        return pad_data(
            data=self,
            one_dimensional_arrays=[GSP_ID, GSP_X_COORDS, GSP_Y_COORDS],
            two_dimensional_arrays=[GSP_YIELD],
            pad_size=pad_size,
        )

    def to_xr_dataset(self, i):

        assert self.batch_size == 0

        example_dim = {"example": np.array([i], dtype=np.int32)}

        # GSP
        n_gsp = len(self.gsp_id)

        one_dataset = xr.DataArray(self.gsp_yield, dims=["time_30", "gsp"], name="gsp_yield")
        one_dataset = one_dataset.to_dataset(name="gsp_yield")
        one_dataset[GSP_DATETIME_INDEX] = xr.DataArray(
            self.gsp_datetime_index,
            dims=["time_30"],
            coords=[np.arange(len(self.gsp_datetime_index))],
        )

        # GSP
        for name in [GSP_ID, GSP_X_COORDS, GSP_Y_COORDS]:

            var = self.__getattribute__(name)

            one_dataset[name] = xr.DataArray(
                var[None, :],
                coords={
                    **example_dim,
                    **{"gsp": np.arange(n_gsp, dtype=np.int32)},
                },
                dims=["example", "gsp"],
            )

        return one_dataset

    @staticmethod
    def from_xr_dataset(xr_dataset):

        if "gsp_yield" in xr_dataset.keys():
            return GSP(
                batch_size=xr_dataset["gsp_yield"].shape[0],
                gsp_yield=xr_dataset[GSP_YIELD],
                gsp_id=xr_dataset[GSP_ID],
                gsp_datetime_index=xr_dataset[GSP_DATETIME_INDEX],
                gsp_x_coords=xr_dataset[GSP_X_COORDS],
                gsp_y_coords=xr_dataset[GSP_Y_COORDS],
            )
        else:
            return None
