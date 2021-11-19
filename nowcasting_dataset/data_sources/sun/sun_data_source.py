""" Loading Raw data """
import logging
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.data_sources.sun.raw_data_load_save import load_from_zarr, x_y_to_name
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.geospatial import calculate_azimuth_and_elevation_angle

logger = logging.getLogger(__name__)


@dataclass
class SunDataSource(DataSource):
    """Add azimuth and elevation angles of the sun."""

    zarr_path: Union[str, Path]

    def __post_init__(self):
        """Post Init"""
        super().__post_init__()
        self._load()

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return Sun

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.  If not, raise a FileNotFoundError."""
        nd_fs_utils.check_path_exists(self.zarr_path)

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> xr.Dataset:
        """
        Get example data from t0_dt and x and y xoordinates

        Args:
            t0_dt: the timestamp to get the sun data for
            x_meters_center: the x coordinate (OSGB)
            y_meters_center: the y coordinate (OSGB)

        Returns: Dictionary of azimuth and elevation data
        """
        # all sun data is from 2019, analaysis showed over the timescale we are interested in the
        # elevation and azimuth angles change by < 1 degree, so to save data, we just use data
        # from 2019.
        t0_dt = t0_dt.replace(year=2019)

        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        # The names of the columns get truncated when saving, therefore we need to look for the
        # name of the columns near the location we are looking for
        locations = np.array(
            [[float(z.split(",")[0]), float(z.split(",")[1])] for z in self.azimuth.columns]
        )
        location = locations[
            np.isclose(locations[:, 0], x_meters_center)
            & np.isclose(locations[:, 1], y_meters_center)
        ]
        # lets make sure there is atleast one
        assert len(location) > 0
        # Take the first location, and x and y coordinates are the first and center entries in
        # this array.
        location = location[0]
        # make name of column to pull data from. The columns name will be about
        # something like '22222.555,3333.6666'
        name = x_y_to_name(x=location[0], y=location[1])

        del x_meters_center, y_meters_center
        azimuth = self.azimuth.loc[start_dt:end_dt][name]
        elevation = self.elevation.loc[start_dt:end_dt][name]

        azimuth = azimuth.to_xarray().rename({"index": "time"})
        elevation = elevation.to_xarray().rename({"index": "time"})

        sun = azimuth.to_dataset(name="azimuth")
        sun["elevation"] = elevation

        return sun

    def _load(self):

        logger.info(f"Loading Sun data from {self.zarr_path}")

        self.azimuth, self.elevation = load_from_zarr(zarr_path=self.zarr_path)

    def get_locations(self, t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """Sun data should not be used to get batch locations"""
        raise NotImplementedError("Sun data should not be used to get batch locations")

    def datetime_index(self) -> pd.DatetimeIndex:
        """Get datetimes where elevation >= 10"""

        # get the lat and lon from london
        latitude = 51
        longitude = 0

        # get elevation for all datetimes
        azimuth_elevation = calculate_azimuth_and_elevation_angle(
            latitude=latitude, longitude=longitude, datestamps=self.elevation.index
        )

        # only select elevations > 10
        mask = azimuth_elevation["elevation"] >= 10

        # create warnings, so we know how many datetimes will be dropped.
        # Should be slightly more than half as its night time 50% of the time
        n_dropping = len(azimuth_elevation) - sum(mask)
        logger.debug(
            f"Will be dropping {n_dropping} datetimes "
            f"out of {len(azimuth_elevation)} as elevation is < 10"
        )

        datetimes = self.elevation[mask].index

        # Sun data is only for 2019, so to expand on these by
        # repeating data from 2014 to 2023
        all_datetimes = pd.DatetimeIndex([])
        for delta_years in range(-5, 5, 1):
            on_year = datetimes + pd.offsets.DateOffset(months=12 * delta_years)
            all_datetimes = all_datetimes.append(on_year)

        return all_datetimes
