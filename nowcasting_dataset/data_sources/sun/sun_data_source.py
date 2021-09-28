from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset import time as nd_time
from dataclasses import dataclass
import pandas as pd
from numbers import Number
from typing import List, Tuple, Union, Optional
from pathlib import Path
import io
import gcsfs
import xarray as xr
from datetime import datetime
from nowcasting_dataset.consts import SUN_AZIMUTH_ANGLE, SUN_ELEVATION_ANGLE

from nowcasting_dataset.data_sources.sun.raw_data_load_save import load_from_zarr, x_y_to_name


@dataclass
class SunDataSource(DataSource):
    """Add azimuth and elevation angles."""

    filename: Union[str, Path]
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    def __post_init__(self):
        super().__post_init__()
        self._load()

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:

        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)

        name = x_y_to_name(x_meters_center, y_meters_center)
        del x_meters_center, y_meters_center

        azimuth = self.azimuth.loc[start_dt:end_dt][name]
        elevation = self.elevation.loc[start_dt:end_dt][name]

        example = Example()
        example[SUN_AZIMUTH_ANGLE] = azimuth.values
        example[SUN_ELEVATION_ANGLE] = elevation.values

        return example

    def _load(self):

        self.azimuth, self.elevation = load_from_zarr(
            filename=self.filename, start_dt=self.start_dt, end_dt=self.end_dt
        )

    def get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        raise NotImplementedError()

    def datetime_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError()
