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
      dt_index: pd.DatetimeIndex  #: Filtered datetime index.
      contiguous_segments: List[Segment]
    """
    batch_size: int = 8
    history_len: int = 1  #: Number of timesteps of 'history', including 'now'.
    forecast_len: int = 1  #: Number of timesteps of forecast.
    sat_filename: Union[str, Path] = consts.SAT_FILENAME
    image_size_pixels: int = 128
    meters_per_pixel: int = 2000
    pin_memory: bool = True  # Passed to DataLoader.
    num_workers: int = 4  # Passed to DataLoader.

    def __post_init__(self):
        super().__init__()

    def prepare_data(self) -> None:
        # Satellite data!
        image_size = square.Square(
            size_pixels=self.image_size_pixels,
            meters_per_pixel=self.meters_per_pixel)
        self.sat_data_source = data_sources.SatelliteDataSource(
            filename=self.sat_filename, image_size=image_size)
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
        dt_index = self._get_daylight_datetime_index()
        # TODO: IMPORTANT! Instead of contiguous_segments, instead
        # just have a dt_index which lists all the valid start dates.
        # For each contiguous_segment, remove the last total_seq_len datetimes,
        # and then check the resulting segment is large enough.
        # Check get_contiguous_segments() to see if it can be simplified.
        #         contiguous_segments = nd_time.get_contiguous_segments(
        #    dt_index=self.dt_index, min_timesteps=self.total_seq_len * 2)

        # Split dt_index into train and test.
        # TODO: Better way to split into train and val date ranges!
        # Split at day boundary, at least.
        assert len(dt_index) > 5
        split = len(dt_index) // 5
        assert split > 0
        split = len(dt_index) - split
        self.train_dt_index = dt_index[:split]
        self.val_dt_index = dt_index[split:]

        # Create datasets
        common_dataset_params = dict(
            batch_size=self.batch_size,
            history_len=self.history_len,
            forecast_len=self.forecast_len,
            data_sources=self.data_sources)
        self.train_dataset = dataset.NowcastingDataset(
            start_dt_index=self.train_dt_index,
            **common_dataset_params)
        self.val_dataset = dataset.NowcastingDataset(
            start_dt_index=self.val_dt_index,
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

            # Disable automatic batching because NowcastingDataset.__iter__
            # returns complete batches
            batch_size=None,
            batch_sampler=None)

    def _get_daylight_datetime_index(self) -> pd.DatetimeIndex:
        """Compute the datetime index.

        Returns the intersection of the datetime indicies of all the
        data_sources; filtered by daylight hours."""
        self._check_has_prepared_data()

        # Get the intersection of datetimes from all data sources.
        available_timestamps = [
            data_source.available_timestamps()
            for data_source in self.data_sources]
        dt_index = nd_time.intersection_of_datetimeindexes(
            available_timestamps)
        del available_timestamps  # save memory

        # Select datetimes that have at least some sunlight
        border_locations = self.sat_data_source.geospatial_border()
        dt_index = nd_time.select_daylight_timestamps(
            dt_index=dt_index, locations=border_locations)

        # Sanity check
        assert len(dt_index) > 2
        assert utils.is_monotonically_increasing(dt_index)
        return dt_index

    def _check_has_prepared_data(self):
        if not self.has_prepared_data:
            raise RuntimeError('Must run prepare_data() first!')
