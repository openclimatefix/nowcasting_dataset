from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.example import Example
from nowcasting_dataset import consts, utils, square
from typing import Union, Iterable, Optional, List, Tuple
from numbers import Number
import xarray as xr
from pathlib import Path
import pandas as pd
import logging
from dataclasses import dataclass, InitVar
import itertools

_LOG = logging.getLogger('nowcasting_dataset')


@dataclass
class SatelliteDataSource(DataSource):
    """
    Attributes:
      _sat_data: xr.DataArray of satellite data, opened by open().
        x is left-to-right.
        y is top-to-bottom.
        Access using public sat_data property.
      filename: Filename of the satellite data Zarr.
      consolidated: Whether or not the Zarr store is consolidated.
      channels: List of satellite channels to load. If None then don't filter by channels.
      image_size_pixels: Size of the width and height of the image crop returned by get_sample().
    """
    filename: Union[str, Path] = consts.SAT_FILENAME
    consolidated: bool = True
    channels: Optional[Iterable[str]] = None
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__()
        self._sat_data = None
        self._square = square.Square(
            size_pixels=image_size_pixels, 
            meters_per_pixel=meters_per_pixel)

    @property
    def sat_data(self):
        if self._sat_data is None:
            raise RuntimeError(
                'Please run `open()` before accessing sat_data!')
        return self._sat_data

    def open(self) -> None:
        # We don't want to open_sat_data in __init__.
        # If we did that, then we couldn't copy SatelliteDataSource
        # instances into separate processes.  Instead,
        # call open() _after_ creating separate processes.
        self._sat_data = self._open_sat_data()
        if self.channels is not None:
            self._sat_data = self._sat_data.sel(variable=list(self.channels))

    def get_sample(
            self,
            x_meters_center: Number,
            y_meters_center: Number,
            t0_dt: pd.Timestamp
    ) -> Example:
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        del t0_dt  # t0 is not used in this method!
        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center)
        selected_sat_data = self.sat_data.sel(
            time=slice(start_dt, end_dt),
            x=slice(bounding_box.left, bounding_box.right),
            y=slice(bounding_box.top, bounding_box.bottom))

        # selected_sat_data is likely to have 1 too many pixels in x and y
        # because sel(x=slice(a, b)) is [a, b], not [a, b).  So trim:
        selected_sat_data = selected_sat_data.isel(
            x=slice(0, self._square.size_pixels),
            y=slice(0, self._square.size_pixels))

        return Example(sat_data=selected_sat_data)

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes"""
        sat_data = self._open_sat_data()
        return pd.DatetimeIndex(sat_data.time.values)

    def geospatial_border(self) -> List[Tuple[Number, Number]]:
        """Get 'corner' coordinates for a rectangle within the boundary of the
        satellite imagery.

        Returns List of 2-tuples of the x and y coordinates of each corner,
        in OSGB projection.
        """
        GEO_BORDER: int = 64  #: In same geo projection and units as sat_data.
        sat_data = self._open_sat_data()
        return [
            (sat_data.x.values[x], sat_data.y.values[y])
            for x, y in itertools.product(
                [GEO_BORDER, -GEO_BORDER],
                [GEO_BORDER, -GEO_BORDER])]

    def _open_sat_data(self):
        return open_sat_data(filename=self.filename, consolidated=self.consolidated)


def open_sat_data(
        filename: Union[str, Path],
        consolidated: bool = False) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Adds 1 minute to the 'time' coordinates, so the timestamps
    are at 00, 05, ..., 55 past the hour.

    Args:
      filename: Cloud URL or local path.
      consolidated: Whether or not the Zarr metadata is consolidated.
    """
    _LOG.debug('Opening satellite data: %s', filename)
    utils.set_fsspec_for_multiprocess()
    dataset = xr.open_zarr(filename, consolidated=consolidated)
    data_array = dataset['stacked_eumetsat_data']

    # The 'time' dimension is at 04, 09, ..., 59 minutes past the hour.
    # To make it easier to align the satellite data with other data sources
    # (which are at 00, 05, ..., 55 minutes past the hour) we add 1 minute to
    # the time dimension.
    data_array['time'] = data_array.time + pd.Timedelta('1 minute')
    return data_array
