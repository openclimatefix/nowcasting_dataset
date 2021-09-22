"""
A class to validate the prepare ml dataset
"""
from typing import Union

import pandas as pd
import torch

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import (
    GSP_DATETIME_INDEX,
    DEFAULT_N_GSP_PER_EXAMPLE,
    DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
)
from nowcasting_dataset.dataset import example
from nowcasting_dataset.dataset.datasets import NetCDFDataset, logger


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
        Initilize data set and run validation

        Args:
            batches: Dataset that needs validating
            configuration: Configuration file
        """

        self.batches = batches
        self.configuration = configuration

        self.validate()

    def validate(self):
        """
        Validate the batches, and calculates unique days that are in the all the batches

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
        example.validate_batch_from_configuration(batch, configuration=self.configuration)

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
            "pv_system_id": torch.randn(self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE),
            "pv_system_x_coords": torch.randn(self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE),
            "pv_system_y_coords": torch.randn(self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE),
            "pv_system_row_number": torch.randn(self.batch_size, DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE),
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
            "gsp_id": torch.randn(self.batch_size, DEFAULT_N_GSP_PER_EXAMPLE),
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
