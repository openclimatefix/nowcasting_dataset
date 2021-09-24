from nowcasting_dataset.data_sources.data_source import ImageDataSource, ZarrDataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from nowcasting_dataset.geospatial import lat_lon_to_osgb
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
    coords={"variable": TOPOGRAPHIC_DATA},
).astype(np.float32)

TOPO_STD = xr.DataArray(
    data=[
        478.841369,
    ],
    dims=["variable"],
    coords={"variable": TOPOGRAPHIC_DATA},
).astype(np.float32)


@dataclass
class TopographicDataSource(ZarrDataSource):
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
        self._data = xr.open_dataset(filename_or_obj=self.filename).to_array(dim=TOPOGRAPHIC_DATA)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        return self._data

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
