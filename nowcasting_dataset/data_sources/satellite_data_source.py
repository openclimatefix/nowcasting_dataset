from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.example import Example
from typing import Union
import xarray as xr
from pathlib import Path
from satellite_dataloader import consts
from satellite_dataloader import utils
import pandas as pd
import datetime
import logging
from dataclasses import dataclass

_LOG = logging.get_logger('nowcasting_dataset')


@dataclass
class SatelliteDataSource(DataSource):
    """
    Attributes:
      sat_data: xr.DataArray of satellite data.
    """
    filename: Union[str, Path] = consts.SAT_DATA_ZARR

    def open(self) -> None:
        self.sat_data = open_sat_data(self.filename)

    def get_sample(
            self,
            start: datetime.datetime,
            end: datetime.datetime,
            t0: datetime.datetime,
            x: float,
            y: float
    ) -> Example:
        del t0  # Not used!
        # TODO: Use x and y!
        # TODO: Pre-compute image_size_meters, perhaps
        # in DataSource superclass.
        selected_sat_data = self.sat_data.sel(time=slice(start, end))
        return Example(sat_data=selected_sat_data)

    def available_timestamps(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available timesteps"""
        sat_data = open_sat_data(self.filename)
        return sat_data.time.values


def open_sat_data(
        filename: Union[str, Path] = consts.SAT_DATA_ZARR) -> xr.DataArray:
    """Lazily opens the Zarr store on Google Cloud Storage (GCS).

    Selects the High Resolution Visible (HRV) satellite channel.
    """
    _LOG.debug('Opening satellite data: %s', filename)
    dataset = utils.open_zarr_on_gcp(filename)
    data_array = dataset['stacked_eumetsat_data'].sel(variable='HRV')

    # The 'time' dimension is at 04, 09, ..., 59 minutes past the hour.
    # To make it easier to align the satellite data with other data sources
    # (which are at 00, 05, ..., 55 minutes past the hour) we add 1 minute to
    # the time dimension.
    data_array['time'] = data_array.time + pd.Timedelta('1 minute')
    return data_array
