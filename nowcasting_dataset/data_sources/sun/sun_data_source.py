""" Loading Raw data """
from dataclasses import dataclass
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.data_sources.sun.raw_data_load_save import load_from_zarr, x_y_to_name
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from nowcasting_dataset.dataset.xr_utils import convert_data_array_to_dataset


@dataclass
class SunDataSource(DataSource):
    """Add azimuth and elevation angles of the sun."""

    filename: Union[str, Path]
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    def __post_init__(self):
        """ Post Init """
        super().__post_init__()
        self._load()

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Sun:
        """
        Get example data from t0_dt and x and y xoordinates

        Args:
            t0_dt: the timestamp to get the sun data for
            x_meters_center: the x coordinate (OSGB)
            y_meters_center: the y coordinate (OSGB)

        Returns: Dictionary of azimuth and elevation data
        """
        # all sun data is from 2019, analaysis showed over the timescale we are interested in the
        # elevation and azimuth angles change by < 1 degree, so to save data, we just use data form 2019
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
        # Take the first location, and x and y coordinates are the first and center entries in this array
        location = location[0]
        # make name of column to pull data from. The columns name will be about something like '22222.555,3333.6666'
        name = x_y_to_name(x=location[0], y=location[1])

        del x_meters_center, y_meters_center
        azimuth = self.azimuth.loc[start_dt:end_dt][name]
        elevation = self.elevation.loc[start_dt:end_dt][name]

        azimuth = azimuth.to_xarray().rename({"index": "time"})
        elevation = elevation.to_xarray().rename({"index": "time"})

        sun = convert_data_array_to_dataset(azimuth).rename({"data": "azimuth"})
        elevation = convert_data_array_to_dataset(elevation)
        sun["elevation"] = elevation.data

        return Sun(sun)

    def _load(self):
        self.azimuth, self.elevation = load_from_zarr(
            filename=self.filename, start_dt=self.start_dt, end_dt=self.end_dt
        )

    def get_locations(self, t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """ Sun data should not be used to get batch locations """
        raise NotImplementedError("Sun data should not be used to get batch locations")

    def datetime_index(self) -> pd.DatetimeIndex:
        """ The datetime index of this datasource """
        return self.azimuth.index
