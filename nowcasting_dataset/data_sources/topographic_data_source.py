from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset import time as nd_time
from dataclasses import dataclass
import pandas as pd
from numbers import Number
from typing import List, Tuple
import xarray as xr


@dataclass
class TopographicDataSource(DataSource):
    """Add topographic/elevation map features."""

    def __post_init__(self):
        super().__post_init__()

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:
        del x_meters_center, y_meters_center
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        index = pd.date_range(start_dt, end_dt, freq="5T")
        return nd_time.datetime_features_in_example(index)

    def get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        raise NotImplementedError()

    def _put_data_into_example(self, selected_data: xr.DataArray) -> Example:
        return Example(
            sat_data=selected_data,
            sat_x_coords=selected_data.x,
            sat_y_coords=selected_data.y,
            sat_datetime_index=selected_data.time,
        )

    def datetime_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError()
