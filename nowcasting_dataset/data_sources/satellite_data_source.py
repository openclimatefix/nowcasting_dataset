from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.example import Example
from nowcasting_dataset import consts
from nowcasting_dataset import utils
from typing import Union, Iterable
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
      sat_data: xr.DataArray of satellite data.
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
            t0: datetime.datetime,
            x_meters: Number,
            y_meters: Number) -> Example:
        del t0  # t0 is not used in this method!
        bounding_box = self._image_size.bounding_box_centered_on(
            x_meters=x_meters, y_meters=y_meters)
        selected_sat_data = self.sat_data.sel(
            time=slice(start, end),
            x=slice(bounding_box.left, bounding_box.right),
            y=slice(bounding_box.top, bounding_box.bottom))
        return Example(sat_data=selected_sat_data)

    def available_timestamps(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available timestamps"""
        sat_data = open_sat_data(filename=self.filename)
        return pd.DatetimeIndex(sat_data.time.values)


def open_sat_data(
        filename: Union[str, Path],
        consolidated: bool=True) -> xr.DataArray:
    """Lazily opens the Zarr store on Google Cloud Storage (GCS).

    Selects the High Resolution Visible (HRV) satellite channel.
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
