from numbers import Number
import pandas as pd
import numpy as np
from nowcasting_dataset.example import Example
from nowcasting_dataset import square
import nowcasting_dataset.time as nd_time
from dataclasses import dataclass, InitVar
from typing import List, Tuple, Union
from pathlib import Path


@dataclass
class DataSource:
    """Abstract base class.

    Attributes:
      filename: Filename of the data source.
      history_len: Number of timesteps of history to include in each example.
        Does NOT include t0.  That is, if history_len = 0 then the example
        will start at t0.
      forecast_len: Number of timesteps of forecast to include in each example.
        Does NOT include t0.  If forecast_len = 0 then the example will end
        at t0.  If both history_len and forecast_len are 0, then the example
        will consist of a single timestep at t0.
      image_size_pixels: Size of the width and height of the image crop
        returned by get_sample().
    """
    filename: Union[str, Path]
    history_len: int
    forecast_len: int
    image_size_pixels: InitVar[int]
    meters_per_pixel: InitVar[int]

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        assert self.history_len >= 0
        assert self.forecast_len >= 0
        # Plus 1 because neither history_len nor forecast_len include t0.
        self._total_seq_len = self.history_len + self.forecast_len + 1
        self._history_dur = nd_time.timesteps_to_duration(self.history_len)
        self._forecast_dur = nd_time.timesteps_to_duration(self.forecast_len)
        self._square = square.Square(
            size_pixels=image_size_pixels,
            meters_per_pixel=meters_per_pixel)
        self._cache = {}

    def open(self):
        """Open the data source, if necessary.

        Called from each worker process.  Useful for data sources where the
        underlying data source cannot be forked (like Zarr on GCP!).

        Data sources which can be forked safely should call open()
        from __init__().
        """
        pass

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes."""
        raise NotImplementedError()

    def pick_locations_for_batch(
            self,
            t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """Find a valid geographical location for each t0_datetime.

        Returns:  x_locations, y_locations. Each has one entry per t0_datetime.
            Locations are in OSGB coordinates.
        """
        # TODO: Do this properly, using PV locations!
        locations = [
            20_000, 40_000,
            500_000, 600_000,
            100_000, 100_000,
            250_000, 250_000]

        location = np.random.choice(locations, size=(len(t0_datetimes), 2))
        
        return location[:, 0], location[:, 1]

    def get_example(
            self,
            x_meters_center: Number,  #: Centre, in OSGB coordinates.
            y_meters_center: Number,  #: Centre, in OSGB coordinates.
            t0_dt: pd.Timestamp  #: Datetime of "now": The most recent obs.
    ) -> Example:
        """Must be overridden by child classes."""
        raise NotImplementedError()
        
    def batch_end(self):
        self._cache = {}
        
    def _get_timestep(self, t0_dt: pd.Timestamp):
        """Get a single timestep of data.  Must be overridden."""
        raise NotImplementedError()
        
    def _get_timestep_with_cache(self, t0_dt: pd.Timestamp):
        try:
            return self._cache[t0_dt]
        except KeyError:
            data = self._get_timestep(t0_dt)
            self._cache[t0_dt] = data
            return data

    def _get_start_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt - self._history_dur

    def _get_end_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt + self._forecast_dur