from numbers import Number
import pandas as pd
from nowcasting_dataset.example import Example
from nowcasting_dataset.square import Square
import nowcasting_dataset.time as nd_time
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class DataSource:
    """Abstract base class.

    Attributes:
      history_len: Number of timesteps of history to include in each example.
        Does NOT include t0.  That is, if history_len = 0 then the example
        will start at t0.
      forecast_len: Number of timesteps of forecast to include in each example.
        Does NOT include t0.  If forecast_len = 0 then the example will end
        at t0.  If both history_len and forecast_len are 0, then the example
        will consist of a single timestep at t0.
    """

    history_len: int
    forecast_len: int

    def __post_init__(self):
        assert self.history_len >= 0
        assert self.forecast_len >= 0
        self._history_dur = nd_time.timesteps_to_duration(self.history_len)
        self._forecast_dur = nd_time.timesteps_to_duration(self.forecast_len)

    def open(self):
        """Open the data source, if necessary.

        Called from each worker process.  Useful for data sources where the
        underlying data source cannot be forked (like Zarr on GCP!).

        Data sources which can be forked safely will open
        the underlying data source in __init__().
        """
        pass

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes."""
        raise NotImplementedError()

    def pick_locations(self, t0_datetimes: pd.DatetimeIndex, n_locations: int) -> List[Tuple[Number, Number]]:
        """Picks n_locations locations for time periods defined by t0_datetimes.
        
        Returns: Locations: A list of 2-tuples (<x_meters_center, y_meters_center> in OSGB coordinates)
        """
        # TODO: Do this properly, using PV locations!
        locations = [
            (20_000, 40_000),
            (500_000, 600_000),
            (100_000, 100_000),
            (250_000, 250_000)]
        return locations[:n_locations]

    def get_example(
            self,
            x_meters_center: Number,  #: Centre, in OSGB coordinates.
            y_meters_center: Number,  #: Centre, in OSGB coordinates.
            t0_dt: pd.Timestamp  #: Datetime of "now": The most recent obs.
    ) -> Example:
        """Must be overridden by child classes."""
        raise NotImplementedError()

    def _get_start_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt - self._history_dur

    def _get_end_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt + self._forecast_dur
