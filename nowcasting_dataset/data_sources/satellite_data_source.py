from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.example import Example
from nowcasting_dataset import utils
from typing import Union, Iterable, Optional
import xarray as xr
from pathlib import Path
import pandas as pd
import logging
from dataclasses import dataclass, InitVar

_LOG = logging.getLogger('nowcasting_dataset')


@dataclass
class SatelliteDataSource(ZarrDataSource):
    """
    Args:
        filename: Must start with 'gs://' if on GCP.
    """
    channels: Optional[Iterable[str]] = (
        'HRV', 'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
        'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073')
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)

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

    def _put_data_into_example(self, selected_data: xr.DataArray) -> Example:
        return Example(sat_data=selected_data)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        data = self.data.sel(time=slice(start_dt, end_dt))
        return data.load()

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes"""
        sat_data = self._open_data()
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
        chunks=None
    )
    data_array = dataset['stacked_eumetsat_data']
    del dataset

    # The 'time' dimension is at 04, 09, ..., 59 minutes past the hour.
    # To make it easier to align the satellite data with other data sources
    # (which are at 00, 05, ..., 55 minutes past the hour) we add 1 minute to
    # the time dimension.
    data_array['time'] = data_array.time + pd.Timedelta('1 minute')
    return data_array
