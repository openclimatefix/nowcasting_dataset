""" Model for Sun features """
from pydantic import Field, validator
import numpy as np
import xarray as xr

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.consts import Array, SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE
from nowcasting_dataset.utils import coord_to_range
from nowcasting_dataset.time import make_random_time_vectors
import logging

logger = logging.getLogger(__name__)


class Sun(DataSourceOutput):
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

        return Sun(
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

    def to_xr_dataset(self, i):
        """ Make a xr dataset """
        logger.debug(f"Making xr dataset for batch {i}")
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

        data = xr.DataArray(
            self.sun_datetime_index,
            dims=["time"],
            coords=[np.arange(len(self.sun_datetime_index))],
        )
        ds = data.to_dataset(name="sun_datetime_index")
        individual_datasets.append(ds)

        return xr.merge(individual_datasets)

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if SUN_AZIMUTH_ANGLE in xr_dataset.keys():
            return Sun(
                batch_size=xr_dataset[SUN_AZIMUTH_ANGLE].shape[0],
                sun_azimuth_angle=xr_dataset[SUN_AZIMUTH_ANGLE],
                sun_elevation_angle=xr_dataset[SUN_ELEVATION_ANGLE],
                sun_datetime_index=xr_dataset["sun_datetime_index"],
            )
        else:
            return None
