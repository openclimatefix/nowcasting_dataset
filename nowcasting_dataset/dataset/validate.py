""" A class to validate the prepare ml dataset """
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
    SUN_AZIMUTH_ANGLE,
    SUN_ELEVATION_ANGLE,
    DATETIME_FEATURE_NAMES,
    TOPOGRAPHIC_X_COORDS,
    TOPOGRAPHIC_DATA,
    TOPOGRAPHIC_Y_COORDS,
)
from nowcasting_dataset.dataset.datasets import NetCDFDataset, logger

# from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.dataset.batch import Batch


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

            # change dict to Batch, this does some validation
            if type(batch) == dict:
                batch = Batch(**batch)

            all_day_from_batch_unique = self.validate_and_get_day_datetimes_for_one_batch(
                batch=batch
            )
            if day_datetimes is None:
                day_datetimes = all_day_from_batch_unique
            else:
                day_datetimes = day_datetimes.join(all_day_from_batch_unique)

        self.day_datetimes = day_datetimes

    def validate_and_get_day_datetimes_for_one_batch(self, batch: Batch):
        """
        For one batch, validate, and return the day datetimes in that batch

        Args:
            batch: batch data

        Returns: list of days that the batch has data for

        """
        if type(batch.metadata.t0_dt) == torch.Tensor:
            batch.metadata.t0_dt = batch.metadata.t0_dt.detach().numpy()

        all_datetimes_from_batch = pd.to_datetime(batch.metadata.t0_dt.reshape(-1), unit="s")
        return pd.DatetimeIndex(all_datetimes_from_batch.date).unique()


class FakeDataset(torch.utils.data.Dataset):
    """Fake dataset."""

    def __init__(self, configuration: Configuration, length: int = 10):
        """
        Init

        Args:
            configuration: configuration object
            length: length of dataset
        """
        self.number_nwp_channels = len(configuration.process.nwp_channels)
        self.length = length
        self.configuration = configuration

    def __len__(self):
        """ Number of pieces of data """
        return self.length

    def per_worker_init(self, worker_id: int):
        """ Not needed """
        pass

    def __getitem__(self, idx):
        """
        Get item, use for iter and next method

        Args:
            idx: batch index

        Returns: Dictionary of random data

        """
        x = Batch.fake(configuration=self.configuration)
        x.change_type_to_numpy()

        return x.dict()
