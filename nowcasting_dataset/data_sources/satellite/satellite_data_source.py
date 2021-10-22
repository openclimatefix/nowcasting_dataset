""" Satellite Data Source """
import logging
from concurrent import futures
from dataclasses import dataclass, InitVar
from numbers import Number
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite
from nowcasting_dataset.dataset.xr_utils import join_list_data_array_to_batch_dataset
import nowcasting_dataset.time as nd_time

_LOG = logging.getLogger("nowcasting_dataset")


from nowcasting_dataset.consts import SAT_VARIABLE_NAMES


@dataclass
class SatelliteDataSource(ZarrDataSource):
    """
    Satellite Data Source

    filename: Must start with 'gs://' if on GCP.
    """

    filename: str = None
    channels: Optional[Iterable[str]] = SAT_VARIABLE_NAMES
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """ Post Init """
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._cache = {}
        n_channels = len(self.channels)
        self._shape_of_example = (
            self._total_seq_length,
            image_size_pixels,
            image_size_pixels,
            n_channels,
        )

    def open(self) -> None:
        """
        Open Satellite data

        We don't want to open_sat_data in __init__.
        If we did that, then we couldn't copy SatelliteDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        self._data = self._open_data()
        self._data = self._data.sel(variable=list(self.channels))

    def _open_data(self) -> xr.DataArray:
        return open_sat_data(filename=self.filename, consolidated=self.consolidated)

    def get_batch(
        self,
        t0_datetimes: pd.DatetimeIndex,
        x_locations: Iterable[Number],
        y_locations: Iterable[Number],
    ) -> Satellite:
        """
        Get batch data

        Load the first _n_timesteps_per_batch concurrently.  This
        loads the timesteps from disk concurrently, and fills the
        cache.  If we try loading all examples
        concurrently, then SatelliteDataSource will try reading from
        empty caches, and things are much slower!

        Args:
            t0_datetimes: list of timestamps for the datetime of the batches. The batch will also
                include data for historic and future depending on `history_minutes` and
                `future_minutes`.
            x_locations: x center batch locations
            y_locations: y center batch locations

        Returns: Batch data

        """
        # Load the first _n_timesteps_per_batch concurrently.  This
        # loads the timesteps from disk concurrently, and fills the
        # cache.  If we try loading all examples
        # concurrently, then SatelliteDataSource will try reading from
        # empty caches, and things are much slower!
        zipped = list(zip(t0_datetimes, x_locations, y_locations))
        batch_size = len(t0_datetimes)

        with futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_examples = []
            for coords in zipped[: self.n_timesteps_per_batch]:
                t0_datetime, x_location, y_location = coords
                future_example = executor.submit(
                    self.get_example, t0_datetime, x_location, y_location
                )
                future_examples.append(future_example)
            examples = [future_example.result() for future_example in future_examples]

        # Load the remaining examples.  This should hit the DataSource caches.
        for coords in zipped[self.n_timesteps_per_batch :]:
            t0_datetime, x_location, y_location = coords
            example = self.get_example(t0_datetime, x_location, y_location)
            examples.append(example)

        output = join_list_data_array_to_batch_dataset(examples)

        self._cache = {}

        return Satellite(output)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        try:
            return self._cache[t0_dt]
        except KeyError:
            start_dt = self._get_start_dt(t0_dt)
            end_dt = self._get_end_dt(t0_dt)
            data = self.data.sel(time=slice(start_dt, end_dt))
            data = data.load()
            self._cache[t0_dt] = data
            return data

    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:

        selected_data.data = selected_data.data.astype(np.float32)

        return selected_data

    def datetime_index(self, remove_night: bool = True) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes

        Args:
            remove_night: If True then remove datetimes at night.
        """
        if self._data is None:
            sat_data = self._open_data()
        else:
            sat_data = self._data

        datetime_index = pd.DatetimeIndex(sat_data.time.values)

        if remove_night:
            border_locations = self.geospatial_border()
            datetime_index = nd_time.select_daylight_datetimes(
                datetimes=datetime_index, locations=border_locations
            )

        return datetime_index


def open_sat_data(filename: str, consolidated: bool) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Adds 1 minute to the 'time' coordinates, so the timestamps
    are at 00, 05, ..., 55 past the hour.

    Args:
      filename: Cloud URL or local path.  If GCP URL, must start with 'gs://'
      consolidated: Whether or not the Zarr metadata is consolidated.
    """
    _LOG.debug("Opening satellite data: %s", filename)

    # We load using chunks=None so xarray *doesn't* use Dask to
    # load the Zarr chunks from disk.  Using Dask to load the data
    # seems to slow things down a lot if the Zarr store has more than
    # about a million chunks.
    # See https://github.com/openclimatefix/nowcasting_dataset/issues/23
    dataset = xr.open_dataset(
        filename, engine="zarr", consolidated=consolidated, mode="r", chunks=None
    )

    data_array = dataset["stacked_eumetsat_data"]
    del dataset

    # The 'time' dimension is at 04, 09, ..., 59 minutes past the hour.
    # To make it easier to align the satellite data with other data sources
    # (which are at 00, 05, ..., 55 minutes past the hour) we add 1 minute to
    # the time dimension.
    data_array["time"] = data_array.time + pd.Timedelta("1 minute")
    return data_array
