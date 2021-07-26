from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.example import Example, to_numpy
from nowcasting_dataset import utils
from typing import Iterable, Optional, List
from numbers import Number
import numpy as np
import xarray as xr
import pandas as pd
import logging
from dataclasses import dataclass, InitVar
from concurrent import futures

_LOG = logging.getLogger('nowcasting_dataset')


SAT_VARIABLE_NAMES = (
    'HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
    'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073')
# Means computed with
# nwp_ds = NWPDataSource(...)
# nwp_ds.open()
# mean = nwp_ds.data.isel(init_time=slice(0, 10)).mean(
#     dim=['step', 'x', 'init_time', 'y']).compute()
SAT_MEAN = xr.DataArray(
    data=[
        93.23458, 131.71373, 843.7779, 736.6148, 771.1189, 589.66034,
        862.29816, 927.69586,  90.70885, 107.58985, 618.4583, 532.47394],
    dims=['variable'],
    coords={'variable': list(SAT_VARIABLE_NAMES)}).astype(np.float32)

SAT_STD = xr.DataArray(
    data=[
        115.34247, 139.92636,  36.99538,  57.366386,  30.346825,
        149.68007,  51.70631,  35.872967, 115.77212, 120.997154,
        98.57828,  99.76469],
    dims=['variable'],
    coords={'variable': list(SAT_VARIABLE_NAMES)}).astype(np.float32)


@dataclass
class SatelliteDataSource(ZarrDataSource):
    """
    Args:
        filename: Must start with 'gs://' if on GCP.
    """
    filename: str = None
    channels: Optional[Iterable[str]] = SAT_VARIABLE_NAMES
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000
    normalise: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._cache = {}
        n_channels = len(self.channels)
        self._shape_of_example = (
            self._total_seq_len, image_size_pixels,
            image_size_pixels, n_channels)

    def open(self) -> None:
        # We don't want to open_sat_data in __init__.
        # If we did that, then we couldn't copy SatelliteDataSource
        # instances into separate processes.  Instead,
        # call open() _after_ creating separate processes.
        self._data = self._open_data()
        self._data = self._data.sel(variable=list(self.channels))

    def _open_data(self) -> xr.DataArray:
        return open_sat_data(
            filename=self.filename, consolidated=self.consolidated)

    def get_batch(
            self,
            t0_datetimes: pd.DatetimeIndex,
            x_locations: Iterable[Number],
            y_locations: Iterable[Number]) -> List[Example]:
        # Load the first _n_timesteps_per_batch concurrently.  This
        # loads the timesteps from disk concurrently, and fills the
        # cache.  If we try loading all examples
        # concurrently, then SatelliteDataSource will try reading from
        # empty caches, and things are much slower!
        zipped = list(zip(t0_datetimes, x_locations, y_locations))
        batch_size = len(t0_datetimes)

        with futures.ThreadPoolExecutor(
                max_workers=batch_size) as executor:
            future_examples = []
            for coords in zipped[:self.n_timesteps_per_batch]:
                t0_datetime, x_location, y_location = coords
                future_example = executor.submit(
                    self.get_example, t0_datetime, x_location, y_location)
                future_examples.append(future_example)
            examples = [
                future_example.result() for future_example in future_examples]

        # Load the remaining examples.  This should hit the DataSource caches.
        for coords in zipped[self.n_timesteps_per_batch:]:
            t0_datetime, x_location, y_location = coords
            example = self.get_example(t0_datetime, x_location, y_location)
            examples.append(example)

        if self.convert_to_numpy:
            examples = [to_numpy(example) for example in examples]
        self._cache = {}
        return examples

    def _put_data_into_example(self, selected_data: xr.DataArray) -> Example:
        return Example(
            sat_data=selected_data,
            sat_x_coords=selected_data.x,
            sat_y_coords=selected_data.y,
            sat_datetime_index=selected_data.time)

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
            self,
            selected_data: xr.DataArray,
            t0_dt: pd.Timestamp) -> xr.DataArray:
        if self.normalise:
            selected_data = selected_data - SAT_MEAN
            selected_data = selected_data / SAT_STD
        return selected_data

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes"""
        if self._data is None:
            sat_data = self._open_data()
        else:
            sat_data = self._data
        return pd.DatetimeIndex(sat_data.time.values)


def open_sat_data(filename: str, consolidated: bool) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Adds 1 minute to the 'time' coordinates, so the timestamps
    are at 00, 05, ..., 55 past the hour.

    Args:
      filename: Cloud URL or local path.  If GCP URL, must start with 'gs://'
      consolidated: Whether or not the Zarr metadata is consolidated.
    """
    _LOG.debug('Opening satellite data: %s', filename)
    utils.set_fsspec_for_multiprocess()

    # We load using chunks=None so xarray *doesn't* use Dask to
    # load the Zarr chunks from disk.  Using Dask to load the data
    # seems to slow things down a lot if the Zarr store has more than
    # about a million chunks.
    # See https://github.com/openclimatefix/nowcasting_dataset/issues/23
    dataset = xr.open_dataset(
        filename, engine='zarr', consolidated=consolidated, mode='r',
        chunks=None)

    data_array = dataset['stacked_eumetsat_data']
    del dataset

    # The 'time' dimension is at 04, 09, ..., 59 minutes past the hour.
    # To make it easier to align the satellite data with other data sources
    # (which are at 00, 05, ..., 55 minutes past the hour) we add 1 minute to
    # the time dimension.
    data_array['time'] = data_array.time + pd.Timedelta('1 minute')
    return data_array
