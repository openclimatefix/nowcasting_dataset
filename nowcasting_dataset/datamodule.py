from typing import Union, List, Tuple
from numbers import Number
from pathlib import Path
import pandas as pd
import itertools
from nowcasting_dataset import data_sources
from nowcasting_dataset import time as nd_time
from nowcasting_dataset import utils
from nowcasting_dataset import consts
from nowcasting_dataset import square
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
      total_seq_len: int  #: Total Number of timesteps.
      dt_index: pd.DatetimeIndex  #: Filtered datetime index.
      contiguous_segments: List[Segment]
    """
    batch_size: int = 8
    history_len: int = 1  #: Number of timesteps of 'history', including 'now'.
    forecast_len: int = 1  #: Number of timesteps of forecast.
    sat_filename: Union[str, Path] = consts.SAT_FILENAME
    image_size_pixels: int = 128
    meters_per_pixel: int = 2000

    def __post_init__(self):
        super().__init__()
        self.total_seq_len = self.history_len + self.forecast_len

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
        # TODO: Split into train and val date ranges!
        self.dt_index = self._get_daylight_datetime_index()
        self.contiguous_segments = nd_time.get_contiguous_segments(
            dt_index=self.dt_index, min_timesteps=self.total_seq_len * 2)

    def _get_daylight_datetime_index(self) -> pd.DatetimeIndex:
        """Compute the datetime index.

        Returns the intersection of the datetime indicies of all the
        data_sources; filtered by daylight hours."""
        self._check_has_prepared_data()

        available_timestamps = [
            data_source.available_timestamps()
            for data_source in self.data_sources]
        dt_index = nd_time.intersection_of_datetimeindexes(
            available_timestamps)
        del available_timestamps  # save memory

        border_locations = self.sat_data_source.geospatial_border()
        dt_index = nd_time.select_daylight_timestamps(
            dt_index=dt_index, locations=border_locations)

        assert len(dt_index) > 2
        assert utils.is_monotonically_increasing(dt_index)
        return dt_index

    def _check_has_prepared_data(self):
        if not self.has_prepared_data:
            raise RuntimeError('Must run prepare_data() first!')
