from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.example import Example
from nowcasting_dataset import consts
from typing import Union, Iterable, Optional
from numbers import Number
import xarray as xr
from pathlib import Path
import pandas as pd
import datetime
import logging
from dataclasses import dataclass

_LOG = logging.getLogger('nowcasting_dataset')


@dataclass
class SatelliteDataSource(DataSource):
    """
    Attributes:
      sat_data: xr.DataArray of satellite data, opened by open().
        x is left-to-right.
        y is top-to-bottom.
      filename: Filename of the satellite data Zarr.
      channels: List of satellite channels to load.
      image_size: Instance of Square, which defines the size of each sample.
        (Inherited from DataSource super-class).
    """
    filename: Union[str, Path] = consts.SAT_DATA_ZARR
    channels: Iterable[str] = ('HRV', )

    def open(self) -> None:
        sat_data = open_sat_data(filename=self.filename)
        self.sat_data = sat_data.sel(variable=list(self.channels))

    def get_sample(
            self,
            start: datetime.datetime,
            end: datetime.datetime,
            x_meters: Number,
            y_meters: Number,
            t0: Optional[datetime.datetime] = None) -> Example:
        del t0  # t0 is not used in this method!
        bounding_box = self.image_size.bounding_box_centered_on(
            x_meters=x_meters, y_meters=y_meters)
        selected_sat_data = self.sat_data.sel(
            time=slice(start, end),
            x=slice(bounding_box.left, bounding_box.right),
            y=slice(bounding_box.top, bounding_box.bottom))

        # selected_sat_data is likely to have 1 too many pixels in x and y
        # because sel(x=slice(a, b)) is [a, b], not [a, b).  So trim:
        selected_sat_data = selected_sat_data.isel(
            x=slice(0, self.image_size.size_pixels),
            y=slice(0, self.image_size.size_pixels))

        return Example(sat_data=selected_sat_data)

    def available_timestamps(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available timestamps"""
        sat_data = open_sat_data(filename=self.filename)
        return pd.DatetimeIndex(sat_data.time.values)


def open_sat_data(
        filename: Union[str, Path],
        consolidated: bool = True) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Adds 1 minute to the 'time' coordinates, so the timestamps
    are at 00, 05, ..., 55 past the hour.

    Args:
      filename: Cloud URL or local path.
      consolidated: Whether or not the Zarr metadata is consolidated.
    """
    _LOG.debug('Opening satellite data: %s', filename)
    dataset = xr.open_zarr(filename, consolidated=consolidated)
    data_array = dataset['stacked_eumetsat_data']

    # The 'time' dimension is at 04, 09, ..., 59 minutes past the hour.
    # To make it easier to align the satellite data with other data sources
    # (which are at 00, 05, ..., 55 minutes past the hour) we add 1 minute to
    # the time dimension.
    data_array['time'] = data_array.time + pd.Timedelta('1 minute')
    return data_array
