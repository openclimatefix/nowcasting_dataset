""" Model for output of NWP data """
from __future__ import annotations
from pydantic import Field, validator
from typing import Union, List
import numpy as np
import xarray as xr
import torch

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutputML, DataSourceOutput
from nowcasting_dataset.consts import (
    Array,
    NWP_VARIABLE_NAMES,
    NWP_DATA,
)
from nowcasting_dataset.utils import coord_to_range
from nowcasting_dataset.time import make_random_time_vectors
from nowcasting_dataset.dataset.pydantic_xr import PydanticXArrayDataSet

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutputML,
    DataSourceOutput,
    create_image_array,
)
from nowcasting_dataset.consts import Array, SAT_VARIABLE_NAMES
from nowcasting_dataset.utils import coord_to_range
from nowcasting_dataset.time import make_random_time_vectors
from nowcasting_dataset.dataset.pydantic_xr import PydanticXArrayDataSet
from nowcasting_dataset.dataset.xr_utils import from_list_data_array_to_batch_dataset

import logging

logger = logging.getLogger(__name__)


class NWP(DataSourceOutput):
    """ Class to store NWP data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data

    __slots__ = []

    # todo add validation here
    pass

    @staticmethod
    def fake(
        batch_size=32,
        seq_length_5=19,
        image_size_pixels=64,
        number_nwp_channels=7,
    ) -> NWP:
        """ Create fake data """
        # make batch of arrays
        xr_arrays = [
            create_image_array(
                seq_length_5=seq_length_5,
                image_size_pixels=image_size_pixels,
                number_channels=number_nwp_channels,
            )
            for _ in range(batch_size)
        ]

        # make dataset
        xr_dataset = from_list_data_array_to_batch_dataset(xr_arrays)

        xr_dataset = xr_dataset.rename({"time": "target_time"})
        xr_dataset["init_time"] = xr_dataset.target_time[:, 0]

        return NWP(xr_dataset)


class NWPML(DataSourceOutputML):
    """ Model for output of NWP data """

    # Shape: [batch_size,] seq_length, width, height, channel
    data: Array = Field(
        ...,
        description=" Numerical weather predictions (NWPs) \
    : Shape: [batch_size,] channel, seq_length, width, height",
    )

    x: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the NWP data. "
        "Shape: [batch_size,] width",
    )
    y: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the NWP data. "
        "Shape: [batch_size,] height",
    )

    target_time: Array = Field(
        ...,
        description="Time index of nwp data at 5 minutes past the hour {0, 5, ..., 55}. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"
        "passed into the ML model.  The 'target time' is the time the NWP is _about_.",
    )

    init_time: Array = Field(..., description="The time when the nwp forecast was made")

    channels: Array = Field(..., description="List of the nwp channels")

    @staticmethod
    def fake(
        batch_size=32,
        seq_length_5=19,
        image_size_pixels=64,
        number_nwp_channels=7,
        time_5=None,
    ):
        """Create fake data"""
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=seq_length_5, seq_len_30_minutes=0
            )

        s = NWPML(
            batch_size=batch_size,
            data=np.random.randn(
                batch_size,
                seq_length_5,
                image_size_pixels,
                image_size_pixels,
                number_nwp_channels,
            ),
            x=np.sort(np.random.randn(batch_size, image_size_pixels)),
            y=np.sort(np.random.randn(batch_size, image_size_pixels))[:, ::-1].copy()
            # copy is needed as torch doesnt not support negative strides
            ,
            target_time=time_5,
            init_time=time_5[0],
            channels=np.array([list(range(number_nwp_channels)) for _ in range(batch_size)]),
        )

        return s

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.time

    @staticmethod
    def from_xr_dataset(xr_dataset: xr.Dataset):
        """Change xr dataset to model with tensors"""
        nwp_batch_ml = xr_dataset.torch.to_tensor(
            ["data", "target_time", "init_time", "x", "y", "channels"]
        )

        return NWPML(**nwp_batch_ml)

    # @property
    # def width(self):
    #     """The width of the nwp data"""
    #     return self.nwp.shape[-2]
    #
    # @property
    # def height(self):
    #     """The width of the nwp data"""
    #     return self.nwp.shape[-1]
    #
    # @property
    # def sequence_length(self):
    #     """The sequence length of the NWP timeseries"""
    #     return self.nwp.shape[-3]
    #
    # @validator("nwp_x_coords")
    # def x_coordinates_shape(cls, v, values):
    #     """ Validate 'nwp_x_coords' """
    #     assert v.shape[-1] == values["nwp"].shape[-2]
    #     return v
    #
    # @validator("nwp_y_coords")
    # def y_coordinates_shape(cls, v, values):
    #     """ Validate 'nwp_y_coords' """
    #     assert v.shape[-1] == values["nwp"].shape[-1]
    #     return v
    #
