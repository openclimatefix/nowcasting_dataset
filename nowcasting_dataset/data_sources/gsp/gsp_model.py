""" Model for output of GSP data """
from pydantic import Field, validator
import numpy as np
import xarray as xr

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput, pad_data
from nowcasting_dataset.consts import Array

from nowcasting_dataset.consts import (
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_DATETIME_INDEX,
    DEFAULT_N_GSP_PER_EXAMPLE,
)
from nowcasting_dataset.time import make_random_time_vectors
import logging

logger = logging.getLogger(__name__)


class GSP(DataSourceOutput):
    """ Model for output of GSP data """

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
        "This is in fact the x centroid of the GSP region"
        "Shape: [batch_size,] n_gsp_per_example",
    )
    gsp_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the gsp. "
        "This are in fact the y centroid of the GSP region"
        "Shape: [batch_size,] n_gsp_per_example",
    )

    @property
    def number_of_gsp(self):
        """The number of Grid Supply Points in this example"""
        return self.gsp_yield.shape[-1]

    @property
    def sequence_length(self):
        """The sequence length of the GSP PV power timeseries data"""
        return self.gsp_yield.shape[-2]

    @validator("gsp_yield")
    def gsp_yield_shape(cls, v, values):
        """ Validate 'gsp_yield' """
        if values["batch_size"] > 0:
            assert len(v.shape) == 3
        else:
            assert len(v.shape) == 2
        return v

    @validator("gsp_x_coords")
    def x_coordinates_shape(cls, v, values):
        """ Validate 'gsp_x_coords' """
        assert v.shape[-1] == values["gsp_yield"].shape[-1]
        return v

    @validator("gsp_y_coords")
    def y_coordinates_shape(cls, v, values):
        """ Validate 'gsp_y_coords' """
        assert v.shape[-1] == values["gsp_yield"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_30, n_gsp_per_batch, time_30=None):
        """ Make a fake GSP object """
        if time_30 is None:
            _, _, time_30 = make_random_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=0, seq_len_30_minutes=seq_length_30
            )

        return GSP(
            batch_size=batch_size,
            gsp_yield=np.random.randn(
                batch_size,
                seq_length_30,
                n_gsp_per_batch,
            ),
            gsp_id=np.sort(np.random.randint(0, 340, (batch_size, n_gsp_per_batch))),
            gsp_datetime_index=time_30,
            gsp_x_coords=np.sort(np.random.randn(batch_size, n_gsp_per_batch)),
            gsp_y_coords=np.sort(np.random.randn(batch_size, n_gsp_per_batch))[:, ::-1].copy(),
        )
        # copy is needed as torch doesnt not support negative strides

    def pad(self, n_gsp_per_example: int = DEFAULT_N_GSP_PER_EXAMPLE):
        """
        Pad out data

        Args:
            n_gsp_per_example: The number of gsp's there are per example.

        Note that nothing is returned as the changes are made inplace.
        """
        assert self.batch_size == 0, "Padding only works for batch_size=0, i.e one Example"

        pad_size = n_gsp_per_example - self.gsp_yield.shape[-1]
        pad_data(
            data=self,
            one_dimensional_arrays=[GSP_ID, GSP_X_COORDS, GSP_Y_COORDS],
            two_dimensional_arrays=[GSP_YIELD],
            pad_size=pad_size,
        )

    def get_datetime_index(self) -> Array:
        """ Get the datetime index of this data """
        return self.gsp_datetime_index

    def to_xr_dataset(self, i):
        """ Make a xr dataset """
        logger.debug(f"Making xr dataset for batch {i}")
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

        one_dataset[GSP_YIELD] = one_dataset[GSP_YIELD].astype(np.float32)
        one_dataset[GSP_ID] = one_dataset[GSP_ID].astype(np.float32)
        one_dataset[GSP_X_COORDS] = one_dataset[GSP_X_COORDS].astype(np.float32)
        one_dataset[GSP_Y_COORDS] = one_dataset[GSP_Y_COORDS].astype(np.float32)

        return one_dataset

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
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
