""" Model for Sun features """
from pydantic import Field, validator
import numpy as np
import xarray as xr

from nowcasting_dataset.data_sources.datasource_output import (
    DataSourceOutputML,
    DataSourceOutput,
    create_sun_dataset,
)
from nowcasting_dataset.consts import Array, SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from nowcasting_dataset.utils import coord_to_range
from nowcasting_dataset.time import make_random_time_vectors
from nowcasting_dataset.dataset.xr_utils import join_data_set_to_batch_dataset
import logging
from nowcasting_dataset.dataset.pydantic_xr import PydanticXArrayDataSet

logger = logging.getLogger(__name__)


class Sun(DataSourceOutput):
    """ Class to store Sun data as a xr.Dataset with some validation """

    # Use to store xr.Dataset data
    __slots__ = []
    _expected_dimensions = ("time",)

    # todo add validation here

    @staticmethod
    def fake(batch_size, seq_length_5):
        """ Create fake data """
        # create dataset with both azimuth and elevation, index with time
        # make batch of arrays
        xr_arrays = [
            create_sun_dataset(
                seq_length=seq_length_5,
            )
            for _ in range(batch_size)
        ]

        # make dataset
        xr_dataset = join_data_set_to_batch_dataset(xr_arrays)

        return Sun(xr_dataset)


class SunML(DataSourceOutputML):
    """ Model for Sun features """

    sun_azimuth_angle: Array = Field(
        ...,
        description="PV azimuth angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )

    sun_elevation_angle: Array = Field(
        ...,
        description="PV elevation angles i.e where the sun is. " "Shape: [batch_size,] seq_length",
    )
    sun_datetime_index: Array

    @validator("sun_elevation_angle")
    def elevation_shape(cls, v, values):
        """
        Validate 'sun_elevation_angle'.

        This is done by change shape is the same as the "sun_azimuth_angle"
        """
        assert v.shape == values["sun_azimuth_angle"].shape
        return v

    @validator("sun_datetime_index")
    def sun_datetime_index_shape(cls, v, values):
        """
        Validate 'sun_datetime_index'.

        This is done by checking last dimension is the same as the last dim of 'sun_azimuth_angle'
        i.e the time dimension
        """
        assert v.shape[-1] == values["sun_azimuth_angle"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5, time_5=None):
        """ Create fake data """
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=seq_length_5, seq_len_30_minutes=0
            )

        return SunML(
            batch_size=batch_size,
            sun_azimuth_angle=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            sun_elevation_angle=np.random.randn(
                batch_size,
                seq_length_5,
            ),
            sun_datetime_index=time_5,
        )

    def get_datetime_index(self):
        """ Get the datetime index of this data """
        return self.sun_datetime_index

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if SUN_AZIMUTH_ANGLE in xr_dataset.keys():
            return SunML(
                batch_size=xr_dataset[SUN_AZIMUTH_ANGLE].shape[0],
                sun_azimuth_angle=xr_dataset[SUN_AZIMUTH_ANGLE],
                sun_elevation_angle=xr_dataset[SUN_ELEVATION_ANGLE],
                sun_datetime_index=xr_dataset["sun_datetime_index"],
            )
        else:
            return None
