""" Model for output of satellite data """
from pydantic import Field, validator
from typing import Union, List
import numpy as np
import xarray as xr

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.consts import Array, SAT_VARIABLE_NAMES
from nowcasting_dataset.utils import coord_to_range
from nowcasting_dataset.time import make_random_time_vectors
import logging

logger = logging.getLogger(__name__)


class Satellite(DataSourceOutput):
    """ Model for output of satellite data """

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
        description="The y (OSGB geo-spatial) coordinates of the satellite images. Shape: [batch_size,] height",
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
        """ Validate 'sat_x_coords' """
        assert v.shape[-1] == values["sat_data"].shape[-3]
        return v

    @validator("sat_y_coords")
    def y_coordinates_shape(cls, v, values):
        """ Validate 'sat_y_coords' """
        assert v.shape[-1] == values["sat_data"].shape[-2]
        return v

    @staticmethod
    def fake(
        batch_size=32,
        seq_length_5=19,
        satellite_image_size_pixels=64,
        number_sat_channels=7,
        time_5=None,
    ):
        """ Create fake data """
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=seq_length_5, seq_len_30_minutes=0
            )

        s = Satellite(
            batch_size=batch_size,
            sat_data=np.random.randn(
                batch_size,
                seq_length_5,
                satellite_image_size_pixels,
                satellite_image_size_pixels,
                number_sat_channels,
            ),
            sat_x_coords=np.sort(np.random.randn(batch_size, satellite_image_size_pixels)),
            sat_y_coords=np.sort(np.random.randn(batch_size, satellite_image_size_pixels))[
                :, ::-1
            ].copy()
            # copy is needed as torch doesnt not support negative strides
            ,
            sat_datetime_index=time_5,
            sat_channel_names=[
                SAT_VARIABLE_NAMES[0:number_sat_channels] for _ in range(batch_size)
            ],
        )

        return s

    def get_datetime_index(self) -> Array:
        """ Get the datetime index of this data """
        return self.sat_datetime_index

    def to_xr_dataset(self, i):
        """ Make a xr dataset """
        logger.debug(f"Making xr dataset for batch {i}")
        if type(self.sat_data) != xr.DataArray:
            self.sat_data = xr.DataArray(
                self.sat_data,
                coords={
                    "time": self.sat_datetime_index,
                    "x": self.sat_x_coords,
                    "y": self.sat_y_coords,
                    "variable": self.sat_channel_names,  # assume all channels are the same
                },
            )

        ds = self.sat_data.to_dataset(name="sat_data")
        ds["sat_data"] = ds["sat_data"].astype(np.int16)
        ds = ds.round(2)

        for dim in ["time", "x", "y"]:
            ds = coord_to_range(ds, dim, prefix="sat")
        ds = ds.rename(
            {
                "variable": f"sat_variable",
                "x": f"sat_x",
                "y": f"sat_y",
            }
        )

        ds["sat_x_coords"] = ds["sat_x_coords"].astype(np.int32)
        ds["sat_y_coords"] = ds["sat_y_coords"].astype(np.int32)

        return ds

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if "sat_data" in xr_dataset.keys():
            return Satellite(
                batch_size=xr_dataset["sat_data"].shape[0],
                sat_data=xr_dataset["sat_data"],
                sat_x_coords=xr_dataset["sat_x_coords"],
                sat_y_coords=xr_dataset["sat_y_coords"],
                sat_datetime_index=xr_dataset["sat_time_coords"],
                sat_channel_names=xr_dataset["sat_data"].sat_variable.values,
            )
        else:
            return None
