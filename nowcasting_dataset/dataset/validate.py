"""
A class to validate the prepare ml dataset
"""
from typing import Union

import numpy as np
import pandas as pd
import torch

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    GSP_DATETIME_INDEX,
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    OBJECT_AT_CENTER,
    PV_SYSTEM_ID,
    PV_YIELD,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_Y_COORDS,
    PV_SYSTEM_ROW_NUMBER,
    PV_AZIMUTH_ANGLE,
    PV_ELEVATION_ANGLE,
    DATETIME_FEATURE_NAMES,
    TOPOGRAPHIC_X_COORDS,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_Y_COORDS,
)
from nowcasting_dataset.dataset import example
from nowcasting_dataset.dataset.datasets import NetCDFDataset, logger
from nowcasting_dataset.dataset.example import Example


class ValidatorDataset:
    """
    Validation of a dataset
    """

    def __init__(
        self,
        batches: Union[NetCDFDataset, torch.utils.data.DataLoader],
        configuration: Configuration,
    ):
        """
        Initialize class and run validation

        Args:
            batches: Dataset that needs validating
            configuration: Configuration file
        """

        self.batches = batches
        self.configuration = configuration

        self.validate()

    def validate(self):
        """
        This validates the batches, and calculates unique days that are in the all the batches

        """
        logger.debug("Validating dataset")
        assert self.configuration is not None

        day_datetimes = None
        for batch_idx, batch in enumerate(self.batches):
            logger.info(f"Validating batch {batch_idx}")

            all_day_from_batch_unique = self.validate_and_get_day_datetimes_for_one_batch(
                batch=batch
            )
            if day_datetimes is None:
                day_datetimes = all_day_from_batch_unique
            else:
                day_datetimes = day_datetimes.join(all_day_from_batch_unique)

        self.day_datetimes = day_datetimes

    def validate_and_get_day_datetimes_for_one_batch(self, batch):
        """
        For one batch, validate, and return the day datetimes in that batch

        Args:
            batch: batch data

        Returns: list of days that the batch has data for

        """
        validate_batch_from_configuration(batch, configuration=self.configuration)

        if type(batch[GSP_DATETIME_INDEX]) == torch.Tensor:
            batch[GSP_DATETIME_INDEX] = batch[GSP_DATETIME_INDEX].detach().numpy()

        all_datetimes_from_batch = pd.to_datetime(batch[GSP_DATETIME_INDEX].reshape(-1), unit="s")
        return pd.DatetimeIndex(all_datetimes_from_batch.date).unique()


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, configuration: Configuration, length: int = 10):

        self.batch_size = configuration.process.batch_size
        self.seq_length_5 = (
            configuration.process.seq_len_5_minutes
        )  # the sequence data in 5 minute steps
        self.seq_length_30 = (
            configuration.process.seq_len_30_minutes
        )  # the sequence data in 30 minute steps
        self.satellite_image_size_pixels = configuration.process.satellite_image_size_pixels
        self.nwp_image_size_pixels = configuration.process.nwp_image_size_pixels
        self.number_sat_channels = len(configuration.process.sat_channels)
        self.number_nwp_channels = len(configuration.process.nwp_channels)
        self.length = length

    def __len__(self):
        return self.length

    def per_worker_init(self, worker_id: int):
        pass

    def __getitem__(self, idx):

        x = {
            "sat_data": torch.randn(
                self.batch_size,
                self.seq_length_5,
                self.satellite_image_size_pixels,
                self.satellite_image_size_pixels,
                self.number_sat_channels,
            ),
            "pv_yield": torch.randn(
                self.batch_size, self.seq_length_5, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE
            ),
            "pv_system_id": torch.randint(940, (self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE)),
            "pv_system_x_coords": torch.randn(self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE),
            "pv_system_y_coords": torch.randn(self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE),
            "pv_system_row_number": torch.randint(
                940, (self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE)
            ),
            "nwp": torch.randn(
                self.batch_size,
                self.number_nwp_channels,
                self.seq_length_5,
                self.nwp_image_size_pixels,
                self.nwp_image_size_pixels,
            ),
            "hour_of_day_sin": torch.randn(self.batch_size, self.seq_length_5),
            "hour_of_day_cos": torch.randn(self.batch_size, self.seq_length_5),
            "day_of_year_sin": torch.randn(self.batch_size, self.seq_length_5),
            "day_of_year_cos": torch.randn(self.batch_size, self.seq_length_5),
            "gsp_yield": torch.randn(
                self.batch_size, self.seq_length_30, DEFAULT_N_GSP_PER_EXAMPLE
            ),
            "gsp_id": torch.randint(340, (self.batch_size, DEFAULT_N_GSP_PER_EXAMPLE)),
            "topo_data": torch.randn(
                self.batch_size, self.satellite_image_size_pixels, self.satellite_image_size_pixels
            ),
        }

        # add a nan
        x["pv_yield"][0, 0, :] = float("nan")

        # add fake x and y coords, and make sure they are sorted
        x["sat_x_coords"], _ = torch.sort(
            torch.randn(self.batch_size, self.satellite_image_size_pixels)
        )
        x["sat_y_coords"], _ = torch.sort(
            torch.randn(self.batch_size, self.satellite_image_size_pixels), descending=True
        )
        x["gsp_x_coords"], _ = torch.sort(torch.randn(self.batch_size, DEFAULT_N_GSP_PER_EXAMPLE))
        x["gsp_y_coords"], _ = torch.sort(
            torch.randn(self.batch_size, DEFAULT_N_GSP_PER_EXAMPLE), descending=True
        )
        x["topo_x_coords"], _ = torch.sort(
            torch.randn(self.batch_size, self.satellite_image_size_pixels)
        )
        x["topo_y_coords"], _ = torch.sort(
            torch.randn(self.batch_size, self.satellite_image_size_pixels), descending=True
        )

        x["nwp_x_coords"], _ = torch.sort(torch.randn(self.batch_size, self.nwp_image_size_pixels))
        x["nwp_y_coords"], _ = torch.sort(
            torch.randn(self.batch_size, self.nwp_image_size_pixels), descending=True
        )

        # add sorted (fake) time series
        x["sat_datetime_index"], _ = torch.sort(torch.randn(self.batch_size, self.seq_length_5))
        x["nwp_target_time"], _ = torch.sort(torch.randn(self.batch_size, self.seq_length_5))
        x["gsp_datetime_index"], _ = torch.sort(torch.randn(self.batch_size, self.seq_length_30))

        x["x_meters_center"], _ = torch.sort(torch.randn(self.batch_size))
        x["y_meters_center"], _ = torch.sort(torch.randn(self.batch_size))

        # clip yield values from 0 to 1
        x["pv_yield"] = torch.clip(x["pv_yield"], min=0, max=1)
        x["gsp_yield"] = torch.clip(x["gsp_yield"], min=0, max=1)

        return x


def validate_example(
    data: Example,
    seq_len_30_minutes: int,
    seq_len_5_minutes: int,
    sat_image_size: int = 64,
    n_sat_channels: int = 1,
    nwp_image_size: int = 0,
    n_nwp_channels: int = 1,
    n_pv_systems_per_example: int = DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
    n_gsp_per_example: int = DEFAULT_N_GSP_PER_EXAMPLE,
    batch: bool = False,
):
    """
    Validate the size and shape of the data
    Args:
        data: Typed dictionary of the data
        seq_len_30_minutes: the length of the sequence for 30 minutely data
        seq_len_5_minutes: the length of the sequence for 5 minutely data
        sat_image_size: the satellite image size
        n_sat_channels: the number of satellite channgles
        nwp_image_size: the nwp image size
        n_nwp_channels: the number of nwp channels
        n_pv_systems_per_example: the number pv systems with nan padding
        n_gsp_per_example: the number gsp systems with nan padding
        batch: if this example class is a batch or not
    """

    n_gsp_id = data[GSP_ID].shape[-1]
    assert (
        n_gsp_id == n_gsp_per_example
    ), f"gsp_is is len {n_gsp_id}, but should be {n_gsp_per_example}"
    assert data[GSP_YIELD].shape[-2:] == (
        seq_len_30_minutes,
        n_gsp_id,
    ), f"gsp_yield is size {data[GSP_YIELD].shape}, but should be {(seq_len_30_minutes, n_gsp_id)}"
    assert data[GSP_X_COORDS].shape[-1] == n_gsp_id
    assert data[GSP_Y_COORDS].shape[-1] == n_gsp_id
    assert data[GSP_DATETIME_INDEX].shape[-1] == seq_len_30_minutes

    # check the GSP data is between 0 and 1
    assert (
        np.nanmax(data[GSP_YIELD]) <= 1.0
    ), f"Maximum GSP value is {np.nanmax(data[GSP_YIELD])} but it should be <= 1"
    assert (
        np.nanmin(data[GSP_YIELD]) >= 0.0
    ), f"Maximum GSP value is {np.nanmin(data[GSP_YIELD])} but it should be >= 0"

    if OBJECT_AT_CENTER in data.keys():
        assert data[OBJECT_AT_CENTER] == "gsp"

    if not batch:
        # add an extract dimension so that its similar to batch data
        data["x_meters_center"] = np.expand_dims(data["x_meters_center"], axis=0)
        data["y_meters_center"] = np.expand_dims(data["y_meters_center"], axis=0)

    # loop over batch
    for d in data["x_meters_center"]:
        assert type(d) in [
            np.float64,
            torch.Tensor,
        ], f"x_meters_center should be np.float64 but is {type(d)}"
    for d in data["y_meters_center"]:
        assert type(d) in [
            np.float64,
            torch.Tensor,
        ], f"y_meters_center should be np.float64 but is {type(d)}"

    assert data[PV_SYSTEM_ID].shape[-1] == n_pv_systems_per_example
    assert data[PV_YIELD].shape[-2:] == (seq_len_5_minutes, n_pv_systems_per_example)
    assert data[PV_SYSTEM_X_COORDS].shape[-1] == n_pv_systems_per_example
    assert data[PV_SYSTEM_Y_COORDS].shape[-1] == n_pv_systems_per_example

    if not batch:
        # add an extract dimension so that its similar to batch data
        data[PV_SYSTEM_ID] = np.expand_dims(data[PV_SYSTEM_ID], axis=0)
        data[PV_SYSTEM_ROW_NUMBER] = np.expand_dims(data[PV_SYSTEM_ID], axis=0)

    # loop over batch
    for i in range(len(data[PV_SYSTEM_ID])):
        n_pv_systems = (data[PV_SYSTEM_ID][i, ~np.isnan(data[PV_SYSTEM_ID][i])]).shape[-1]
        n_pv_syetem_row_numbers = (
            data[PV_SYSTEM_ROW_NUMBER][i, ~np.isnan(data[PV_SYSTEM_ROW_NUMBER][i])]
        ).shape[-1]
        assert n_pv_syetem_row_numbers == n_pv_systems, (
            f"Number of PV systems ({n_pv_systems}) does not match the "
            f"pv systems row numbers ({n_pv_syetem_row_numbers})"
        )

        if n_pv_systems > 0:
            # check the PV data is between 0 and 1
            assert (
                np.nanmax(data[PV_YIELD]) <= 1.0
            ), f"Maximum PV value is {np.nanmax(data[PV_YIELD])} but it should be <= 1"
            assert (
                np.nanmin(data[PV_YIELD]) >= 0.0
            ), f"Maximum PV value is {np.nanmin(data[PV_YIELD])} but it should be <= 1"

    if PV_AZIMUTH_ANGLE in data.keys():
        assert data[PV_AZIMUTH_ANGLE].shape[-2:] == (seq_len_5_minutes, n_pv_systems_per_example)
    if PV_AZIMUTH_ANGLE in data.keys():
        assert data[PV_ELEVATION_ANGLE].shape[-2:] == (seq_len_5_minutes, n_pv_systems_per_example)

    assert data["sat_data"].shape[-4:] == (
        seq_len_5_minutes,
        sat_image_size,
        sat_image_size,
        n_sat_channels,
    )
    assert data["sat_x_coords"].shape[-1] == sat_image_size
    assert data["sat_y_coords"].shape[-1] == sat_image_size
    assert data["sat_datetime_index"].shape[-1] == seq_len_5_minutes

    assert data[TOPOGRAPHIC_DATA].shape[-2:] == (sat_image_size, sat_image_size)
    assert data[TOPOGRAPHIC_Y_COORDS].shape[-1] == sat_image_size
    assert data[TOPOGRAPHIC_X_COORDS].shape[-1] == sat_image_size

    nwp_correct_shape = (
        n_nwp_channels,
        seq_len_5_minutes,
        nwp_image_size,
        nwp_image_size,
    )
    nwp_shape = data["nwp"].shape[-4:]
    assert (
        nwp_shape == nwp_correct_shape
    ), f"NWP shape should be ({nwp_correct_shape}), but instead it is {nwp_shape}"
    assert data["nwp_x_coords"].shape[-1] == nwp_image_size
    assert data["nwp_y_coords"].shape[-1] == nwp_image_size
    assert data["nwp_target_time"].shape[-1] == seq_len_5_minutes

    for feature in DATETIME_FEATURE_NAMES:
        assert data[feature].shape[-1] == seq_len_5_minutes


def validate_batch_from_configuration(data: Example, configuration: Configuration):
    """
    Validate data using a configuration

    Args:
        data: batch of data
        configuration: confgiruation of the data

    """

    validate_example(
        data=data,
        seq_len_30_minutes=configuration.process.seq_len_30_minutes,
        seq_len_5_minutes=configuration.process.seq_len_5_minutes,
        sat_image_size=configuration.process.satellite_image_size_pixels,
        n_sat_channels=len(configuration.process.sat_channels),
        nwp_image_size=configuration.process.nwp_image_size_pixels,
        n_nwp_channels=len(configuration.process.nwp_channels),
        n_pv_systems_per_example=DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
        n_gsp_per_example=DEFAULT_N_GSP_PER_EXAMPLE,
        batch=True,
    )
