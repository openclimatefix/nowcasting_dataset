""" Loading Raw data """
import logging
from dataclasses import dataclass
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.data_sources.sun.raw_data_load_save import load_from_zarr, x_y_to_name
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.geospatial import calculate_azimuth_and_elevation_angle, osgb_to_lat_lon

logger = logging.getLogger(__name__)


@dataclass
class SunDataSource(DataSource):
    """Add azimuth and elevation angles of the sun."""

    zarr_path: Union[str, Path]
    load_live: bool = False
    elevation_limit: int = 10

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
        if not self.load_live:
            nd_fs_utils.check_path_exists(self.zarr_path)

    def get_example(self, location: SpaceTimeLocation) -> xr.Dataset:
        """
        Get example data from t0_dt and x and y xoordinates

        Args:
            location: A location object of the example which contains
                - a timestamp of the example (t0_datetime_utc),
                - the x center location of the example (x_location_osgb)
                - the y center location of the example(y_location_osgb)

        Returns: Dictionary of azimuth and elevation data
        """
        # all sun data is from 2019, analaysis showed over the timescale we are interested in the
        # elevation and azimuth angles change by < 1 degree, so to save data, we just use data
        # from 2019.

        t0_datetime_utc = location.t0_datetime_utc
        x_center_osgb = location.x_center_osgb
        y_center_osgb = location.y_center_osgb

        t0_datetime_utc = t0_datetime_utc.replace(year=2019)

        start_dt = self._get_start_dt(t0_datetime_utc)
        end_dt = self._get_end_dt(t0_datetime_utc)

        if not self.load_live:

            # The names of the columns get truncated when saving, therefore we need to look for the
            # name of the columns near the location we are looking for
            locations = np.array(
                [[float(z.split(",")[0]), float(z.split(",")[1])] for z in self.azimuth.columns]
            )
            location = locations[
                np.isclose(locations[:, 0], x_center_osgb)
                & np.isclose(locations[:, 1], y_center_osgb)
            ]
            # lets make sure there is atleast one
            assert len(location) > 0, (
                f"Could not find any locations for {location}. "
                f"The sun data source locations are {locations}"
            )
            # Take the first location, and x and y coordinates are the first and center entries in
            # this array.
            location = location[0]
            # make name of column to pull data from. The columns name will be about
            # something like '22222.555,3333.6666'
            name = x_y_to_name(x=location[0], y=location[1])

            del x_center_osgb, y_center_osgb
            azimuth = self.azimuth.loc[start_dt:end_dt][name]
            elevation = self.elevation.loc[start_dt:end_dt][name]

        else:

            latitude, longitude = osgb_to_lat_lon(x=x_center_osgb, y=y_center_osgb)

            datestamps = pd.date_range(start=start_dt, end=end_dt, freq="5T").tolist()
            azimuth_elevation = calculate_azimuth_and_elevation_angle(
                latitude=latitude, longitude=longitude, datestamps=datestamps
            )
            azimuth = azimuth_elevation["azimuth"]
            elevation = azimuth_elevation["elevation"]

        azimuth = azimuth.to_xarray().rename({"index": "time"})
        elevation = elevation.to_xarray().rename({"index": "time"})

        sun = azimuth.to_dataset(name="azimuth")
        sun["elevation"] = elevation

        return sun

    def _load(self):

        logger.info(f"Loading Sun data from {self.zarr_path}")

        if not self.load_live:
            self.azimuth, self.elevation = load_from_zarr(zarr_path=self.zarr_path)

    def get_locations(
        self, t0_datetimes_utc: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        """Sun data should not be used to get batch locations"""
        raise NotImplementedError("Sun data should not be used to get batch locations")

    def datetime_index(self) -> pd.DatetimeIndex:
        """Get datetimes where elevation >= 10"""

        # get the lat and lon from london
        latitude = 51
        longitude = 0

        if not self.load_live:
            datestamps = self.elevation.index
        else:
            datestamps = pd.date_range(
                datetime(2019, 1, 1), datetime(2019, 12, 31, 23, 55), freq="5T"
            )

        # get elevation for all datetimes
        azimuth_elevation = calculate_azimuth_and_elevation_angle(
            latitude=latitude, longitude=longitude, datestamps=datestamps
        )

        # only select elevations > 10
        mask = azimuth_elevation["elevation"] >= self.elevation_limit

        # create warnings, so we know how many datetimes will be dropped.
        # Should be slightly more than half as its night time 50% of the time
        n_dropping = len(azimuth_elevation) - sum(mask)
        logger.debug(
            f"Will be dropping {n_dropping} datetimes "
            f"out of {len(azimuth_elevation)} as elevation is < 10"
        )

        datetimes = datestamps[mask]

        # Sun data is only for 2019, so to expand on these by
        # repeating data from 2014 to 2023
        all_datetimes = pd.DatetimeIndex([])
        for delta_years in range(-5, 5, 1):
            on_year = datetimes + pd.offsets.DateOffset(months=12 * delta_years)
            all_datetimes = all_datetimes.append(on_year)

        return all_datetimes
