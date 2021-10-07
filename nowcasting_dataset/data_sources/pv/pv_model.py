""" Model for output of PV data """
from pydantic import Field, validator
import numpy as np
import xarray as xr

from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput, pad_data
from nowcasting_dataset.consts import (
    Array,
    PV_YIELD,
    PV_DATETIME_INDEX,
    PV_SYSTEM_Y_COORDS,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_ROW_NUMBER,
    PV_SYSTEM_ID,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
)
from nowcasting_dataset.time import make_random_time_vectors
import logging

logger = logging.getLogger(__name__)


class PV(DataSourceOutput):
    """ Model for output of PV data """

    # Shape: [batch_size,] seq_length, width, height, channel
    pv_yield: Array = Field(
        ...,
        description=" PV yield from all PV systems in the region of interest (ROI). \
    : Includes central PV system, which will always be the first entry. \
    : shape = [batch_size, ] seq_length, n_pv_systems_per_example",
    )

    #: PV identification.
    #: shape = [batch_size, ] n_pv_systems_per_example
    pv_system_id: Array = Field(..., description="PV system ID, e.g. from PVoutput.org")
    pv_system_row_number: Array = Field(..., description="pv row number, made by OCF  TODO")

    pv_datetime_index: Array = Field(
        ...,
        description="The datetime associated with the pv system data. shape = [batch_size, ] sequence length,",
    )

    pv_system_x_coords: Array = Field(
        ...,
        description="The x (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )
    pv_system_y_coords: Array = Field(
        ...,
        description="The y (OSGB geo-spatial) coordinates of the pv systems. "
        "Shape: [batch_size,] n_pv_systems_per_example",
    )

    @property
    def number_of_pv_systems(self):
        """The number of pv systems"""
        return self.pv_yield.shape[-1]

    @property
    def sequence_length(self):
        """The sequence length of the pv data"""
        return self.pv_yield.shape[-2]

    @validator("pv_system_x_coords")
    def x_coordinates_shape(cls, v, values):
        """ Validate 'pv_system_x_coords' """
        assert v.shape[-1] == values["pv_yield"].shape[-1]
        return v

    @validator("pv_system_y_coords")
    def y_coordinates_shape(cls, v, values):
        """ Validate 'pv_system_y_coords' """
        assert v.shape[-1] == values["pv_yield"].shape[-1]
        return v

    @staticmethod
    def fake(batch_size, seq_length_5, n_pv_systems_per_batch, time_5=None):
        """ Create fake data """
        if time_5 is None:
            _, time_5, _ = make_random_time_vectors(
                batch_size=batch_size, seq_len_5_minutes=seq_length_5, seq_len_30_minutes=0
            )

        return PV(
            batch_size=batch_size,
            pv_yield=np.random.randn(
                batch_size,
                seq_length_5,
                n_pv_systems_per_batch,
            ),
            pv_system_id=np.sort(np.random.randint(0, 10000, (batch_size, n_pv_systems_per_batch))),
            pv_system_row_number=np.sort(
                np.random.randint(0, 1000, (batch_size, n_pv_systems_per_batch))
            ),
            pv_datetime_index=time_5,
            pv_system_x_coords=np.sort(np.random.randn(batch_size, n_pv_systems_per_batch)),
            pv_system_y_coords=np.sort(np.random.randn(batch_size, n_pv_systems_per_batch))[
                :, ::-1
            ].copy(),  # copy is needed as torch doesnt not support negative strides
        )

    def pad(self, n_pv_systems_per_example: int = DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE):
        """
        Pad out data

        Args:
            n_pv_systems_per_example: The number of pv systems there are per example.

        Note that nothing is returned as the changes are made inplace.
        """
        assert self.batch_size == 0, "Padding only works for batch_size=0, i.e one Example"

        pad_size = n_pv_systems_per_example - self.pv_yield.shape[-1]
        # Pad (if necessary) so returned arrays are always of size
        pad_shape = (0, pad_size)  # (before, after)

        one_dimensional_arrays = [
            PV_SYSTEM_ID,
            PV_SYSTEM_ROW_NUMBER,
            PV_SYSTEM_X_COORDS,
            PV_SYSTEM_Y_COORDS,
        ]

        pad_data(
            data=self,
            pad_size=pad_size,
            one_dimensional_arrays=one_dimensional_arrays,
            two_dimensional_arrays=[PV_YIELD],
        )

    def get_datetime_index(self) -> Array:
        """ Get the datetime index of this data """
        return self.pv_datetime_index

    def to_xr_dataset(self, i):
        """ Make a xr dataset """
        logger.debug(f"Making xr dataset for batch {i}")
        assert self.batch_size == 0

        example_dim = {"example": np.array([i], dtype=np.int32)}

        # PV
        one_dataset = xr.DataArray(self.pv_yield, dims=["time", "pv_system"])
        one_dataset = one_dataset.to_dataset(name="pv_yield")
        n_pv_systems = len(self.pv_system_id)

        one_dataset[PV_DATETIME_INDEX] = xr.DataArray(
            self.pv_datetime_index,
            dims=["time"],
            coords=[np.arange(len(self.pv_datetime_index))],
        )

        # 1D
        for name in [
            PV_SYSTEM_ID,
            PV_SYSTEM_ROW_NUMBER,
            PV_SYSTEM_X_COORDS,
            PV_SYSTEM_Y_COORDS,
        ]:
            var = self.__getattribute__(name)

            one_dataset[name] = xr.DataArray(
                var[None, :],
                coords={
                    **example_dim,
                    **{"pv_system": np.arange(n_pv_systems, dtype=np.int32)},
                },
                dims=["example", "pv_system"],
            )

        one_dataset["pv_system_id"] = one_dataset["pv_system_id"].astype(np.float32)
        one_dataset["pv_system_row_number"] = one_dataset["pv_system_row_number"].astype(np.float32)
        one_dataset["pv_system_x_coords"] = one_dataset["pv_system_x_coords"].astype(np.float32)
        one_dataset["pv_system_y_coords"] = one_dataset["pv_system_y_coords"].astype(np.float32)

        return one_dataset

    @staticmethod
    def from_xr_dataset(xr_dataset):
        """ Change xr dataset to model. If data does not exist, then return None """
        if PV_YIELD in xr_dataset.keys():
            return PV(
                batch_size=xr_dataset[PV_YIELD].shape[0],
                pv_yield=xr_dataset[PV_YIELD],
                pv_system_id=xr_dataset[PV_SYSTEM_ID],
                pv_system_row_number=xr_dataset[PV_SYSTEM_ROW_NUMBER],
                pv_datetime_index=xr_dataset[PV_DATETIME_INDEX],
                pv_system_x_coords=xr_dataset[PV_SYSTEM_X_COORDS],
                pv_system_y_coords=xr_dataset[PV_SYSTEM_Y_COORDS],
            )
        else:
            return None
