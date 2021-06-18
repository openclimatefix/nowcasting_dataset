from typing import Union
from pathlib import Path
import pandas as pd
import torch
from nowcasting_dataset import data_sources
from nowcasting_dataset import time as nd_time
from nowcasting_dataset import utils
from nowcasting_dataset import consts
from nowcasting_dataset import square
from nowcasting_dataset import dataset
from dataclasses import dataclass
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pytorch_lightning as pl


@dataclass
class NowcastingDataModule(pl.LightningDataModule):
    """
    Attributes (additional to the dataclass attributes):
      sat_data_source: SatelliteDataSource
      data_sources: List[DataSource]
      train_t0_datetimes: pd.DatetimeIndex
      val_t0_datetimes: pd.DatetimeIndex
    """
    batch_size: int = 8
    history_len: int = 2  #: Number of timesteps of history, not including t0.
    forecast_len: int = 12  #: Number of timesteps of forecast.
    sat_filename: Union[str, Path] = consts.SAT_FILENAME
    image_size_pixels: int = 128
    meters_per_pixel: int = 2000
    pin_memory: bool = True  #: Passed to DataLoader.
    num_workers: int = 16  #: Passed to DataLoader.
    prefetch_factor: int = 64  #: Passed to DataLoader.
    n_samples_per_timestep: int = 2  #: Passed to NowcastingDataset

    def __post_init__(self):
        super().__init__()
        self.total_seq_len = self.history_len + self.forecast_len

    def prepare_data(self) -> None:
        # Satellite data!
        image_size = square.Square(
            size_pixels=self.image_size_pixels,
            meters_per_pixel=self.meters_per_pixel)
        self.sat_data_source = data_sources.SatelliteDataSource(
            filename=self.sat_filename,
            image_size=image_size,
            history_len=self.history_len,
            forecast_len=0
        )
        self.data_sources = [self.sat_data_source]

    def setup(self):
        """Split data, etc.

        ## Selecting daytime data.

        We're interested in forecasting solar power generation, so we
        don't care about nighttime data :)

        In the UK in summer, the sun rises first in the north east, and
        sets last in the north west [1].  In summer, the north gets more
        hours of sunshine per day.

        In the UK in winter, the sun rises first in the south east, and
        sets last in the south west [2].  In winter, the south gets more
        hours of sunshine per day.

        |                        | Summer | Winter |
        |           ---:         |  :---: |  :---: |
        | Sun rises first in     | N.E.   | S.E.   |
        | Sun sets last in       | N.W.   | S.W.   |
        | Most hours of sunlight | North  | South  |

        Before training, we select timesteps which have at least some
        sunlight.  We do this by computing the clearsky global horizontal
        irradiance (GHI) for the four corners of the satellite imagery,
        and for all the timesteps in the dataset.  We only use timesteps
        where the maximum global horizontal irradiance across all four
        corners is above some threshold.

        The 'clearsky solar irradiance' is the amount of sunlight we'd
        expect on a clear day at a specific time and location. The SI unit
        of irradiance is watt per square meter.  The 'global horizontal
        irradiance' (GHI) is the total sunlight that would hit a
        horizontal surface on the surface of the Earth.  The GHI is the
        sum of the direct irradiance (sunlight which takes a direct path
        from the Sun to the Earth's surface) and the diffuse horizontal
        irradiance (the sunlight scattered from the atmosphere).  For more
        info, see: https://en.wikipedia.org/wiki/Solar_irradiance

        References:
          1. [Video of June 2019](https://www.youtube.com/watch?v=IOp-tj-IJpk)
          2. [Video of Jan 2019](https://www.youtube.com/watch?v=CJ4prUVa2nQ)
        """
        self._check_has_prepared_data()
        all_datetimes = self._get_datetimes()
        t0_datetimes = nd_time.get_t0_datetimes(
            datetimes=all_datetimes, total_seq_len=self.total_seq_len,
            history_len=self.history_len)
        del all_datetimes

        # Split dt_index into train and test.
        # TODO: Better way to split into train and val date ranges!
        # Split at day boundary, at least. Maybe take every first day
        # of the month for validation? Need to be careful to make sure the
        # validation dataset always takes exactly the same PV panels and
        # datetimes when sampling.
        assert len(t0_datetimes) > 5
        split = len(t0_datetimes) // 5
        assert split > 0
        split = len(t0_datetimes) - split
        self.train_t0_datetimes = t0_datetimes[:split]
        self.val_t0_datetimes = t0_datetimes[split:]

        # Create datasets
        common_dataset_params = dict(
            batch_size=self.batch_size,
            data_sources=self.data_sources,
            n_samples_per_timestep=self.n_samples_per_timestep)
        self.train_dataset = dataset.NowcastingDataset(
            t0_datetimes=self.train_t0_datetimes,
            **common_dataset_params)
        self.val_dataset = dataset.NowcastingDataset(
            t0_datetimes=self.val_t0_datetimes,
            **common_dataset_params)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset, **self._common_dataloader_params())

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset, **self._common_dataloader_params())

    def _common_dataloader_params(self):
        return dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            worker_init_fn=dataset.worker_init_fn,
            prefetch_factor=self.prefetch_factor,

            # Disable automatic batching because NowcastingDataset.__iter__
            # returns complete batches
            batch_size=None,
            batch_sampler=None)

    def _get_datetimes(self) -> pd.DatetimeIndex:
        """Compute the datetime index.

        Returns the intersection of the datetime indicies of all the
        data_sources; filtered by daylight hours."""
        self._check_has_prepared_data()

        # Get the intersection of datetimes from all data sources.
        all_datetime_indexes = [
            data_source.datetime_index()
            for data_source in self.data_sources]
        datetimes = nd_time.intersection_of_datetimeindexes(
            all_datetime_indexes)
        del all_datetime_indexes  # save memory

        # Select datetimes that have at least some sunlight
        border_locations = self.sat_data_source.geospatial_border()
        dt_index = nd_time.select_daylight_datetimes(
            datetimes=datetimes, locations=border_locations)

        # Sanity check
        assert len(dt_index) > 2
        assert utils.is_monotonically_increasing(dt_index)
        return dt_index

    def _check_has_prepared_data(self):
        if not self.has_prepared_data:
            raise RuntimeError('Must run prepare_data() first!')
