""" Model for output of satellite data """
from __future__ import annotations
from pydantic import Field, validator
from typing import Union, List
import numpy as np
import xarray as xr
import pandas as pd

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


class Satellite(DataSourceOutput):
    # Use to store xr.Dataset data

    __slots__ = []

    # todo add validation here
    pass

    @staticmethod
    def fake(
        batch_size=32,
        seq_length_5=19,
        satellite_image_size_pixels=64,
        number_sat_channels=7,
    ) -> Satellite:
        pass

        # make batch of arrays
        xr_arrays = [
            create_image_array(
                seq_length_5=seq_length_5,
                image_size_pixels=satellite_image_size_pixels,
                number_channels=number_sat_channels,
            )
            for _ in range(batch_size)
        ]

        # make dataset
        xr_dataset = from_list_data_array_to_batch_dataset(xr_arrays)

        return Satellite(xr_dataset)


class SatelliteML(DataSourceOutputML):
    """Model for output of satellite data"""

    # Shape: [batch_size,] seq_length, width, height, channel
    data: Array = Field(
        ...,
        description="Satellites images. Shape: [batch_size,] seq_length, width, height, channel",
    )
    x: Array = Field(
        ...,
        description="aThe x (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] width",
    )
    y: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] height",
    )

    time: Array = Field(
        ...,
        description="Time index of satellite data at 5 minutes past the hour {0, 5, ..., 55}. "
        "*not* the {4, 9, ..., 59} timings of the satellite imagery. "
        "Datetimes become Unix epochs (UTC) represented as int64 just before being"  # TOOD, is it int32?
        "passed into the ML model.",
    )

    channels: Array = Field(..., description="List of the satellite channels")

    @staticmethod
    def fake(
        batch_size=32,
        seq_length_5=19,
        satellite_image_size_pixels=64,
        number_sat_channels=7,
        time_5=None,
    ):
        """Create fake data"""
        # if time_5 is None:
        #     _, time_5, _ = make_random_time_vectors(
        #         batch_size=batch_size, seq_len_5_minutes=seq_length_5, seq_len_30_minutes=0
        #     )

        s = Satellite.fake(
            batch_size=batch_size,
            seq_length_5=seq_length_5,
            satellite_image_size_pixels=satellite_image_size_pixels,
            number_sat_channels=number_sat_channels,
        )

        return SatelliteML.from_xr_dataset(xr_dataset=s)

    def get_datetime_index(self) -> Array:
        """Get the datetime index of this data"""
        return self.time

    @staticmethod
    def from_xr_dataset(xr_dataset: xr.Dataset):
        """Change xr dataset to model. If data does not exist, then return None"""

        satellite_batch_ml = xr_dataset.torch.to_tensor(["data", "time", "x", "y", "channels"])

        return SatelliteML(**satellite_batch_ml)
