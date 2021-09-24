from nowcasting_dataset.data_sources.data_source import DataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset import time as nd_time
from dataclasses import dataclass, InitVar
import pandas as pd
from numbers import Number
from typing import List, Tuple
import xarray as xr
import numpy as np

# Means computed with
# out_fp = "europe_dem_1km.tif"
# out = rasterio.open(out_fp)
# data = out.read(masked=True)
# print(np.mean(data))
# print(np.std(data))
TOPO_MEAN = xr.DataArray(
    data=[
        365.486887,
    ],
    dims=["variable"],
    coords={"variable": "topo_data"},
).astype(np.float32)

TOPO_STD = xr.DataArray(
    data=[
        478.841369,
    ],
    dims=["variable"],
    coords={"variable": "topo_data"},
).astype(np.float32)


@dataclass
class TopographicDataSource(DataSource):
    """Add topographic/elevation map features."""

    filename: str = None
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000
    normalize: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._cache = {}
        self._shape_of_example = (
            image_size_pixels,
            image_size_pixels,
            1,  # Topographic data is just the height, so single channel
        )

    def open(self) -> None:
        # We don't want to open_sat_data in __init__.
        # If we did that, then we couldn't copy SatelliteDataSource
        # instances into separate processes.  Instead,
        # call open() _after_ creating separate processes.
        self._data = self._open_data()
        self._data = self._data.sel(variable=list(self.channels))

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:
        del t0_dt

        return NotImplementedError

    def get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        raise NotImplementedError()

    def _put_data_into_example(self, selected_data: xr.DataArray) -> Example:
        return Example(
            topo_data=selected_data,
            topo_x_coords=selected_data.x,
            topo_y_coords=selected_data.y,
        )

    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:
        if self.normalize:
            selected_data = selected_data - TOPO_MEAN
            selected_data = selected_data / TOPO_STD
        return selected_data

    def datetime_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError()
