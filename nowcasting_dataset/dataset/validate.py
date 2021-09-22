"""
A class to validate the prepare ml dataset
"""
from typing import Union

import pandas as pd
import torch

from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import GSP_DATETIME_INDEX
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
