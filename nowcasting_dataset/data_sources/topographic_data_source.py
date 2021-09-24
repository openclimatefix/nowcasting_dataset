from nowcasting_dataset.data_sources.data_source import ImageDataSource, ZarrDataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from dataclasses import dataclass, InitVar
from numbers import Number
import pandas as pd
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
    coords={"variable": [TOPOGRAPHIC_DATA]},
).astype(np.float32)

TOPO_STD = xr.DataArray(
    data=[
        478.841369,
    ],
    dims=["variable"],
    coords={"variable": [TOPOGRAPHIC_DATA]},
).astype(np.float32)


@dataclass
class TopographicDataSource(ImageDataSource):
    """Add topographic/elevation map features."""

    # TODO Have to resample on fly, if meters_per_pixel is greater than 1km, than resample area of interest
    # to match that new resolution, gives artifacts, but matches size and shape

    filename: str = None
    normalize: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._shape_of_example = (
            image_size_pixels,
            image_size_pixels,
            1,  # Topographic data is just the height, so single channel
        )
        self._data = xr.open_dataset(filename_or_obj=self.filename).to_array(dim=TOPOGRAPHIC_DATA)
        self._stored_pixel_size = self._data.attrs["scale_km"]
        print(self._data)

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:
        selected_data = self._get_time_slice(t0_dt)
        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )
        print(bounding_box)
        selected_data = selected_data.sel(
            x=slice(bounding_box.left, bounding_box.right),
            y=slice(bounding_box.top, bounding_box.bottom),
        )
        print(selected_data.shape)

        # selected_sat_data is likely to have 1 too many pixels in x and y
        # because sel(x=slice(a, b)) is [a, b], not [a, b).  So trim:
        selected_data = selected_data.isel(
            x=slice(0, self._square.size_pixels), y=slice(0, self._square.size_pixels)
        )

        selected_data = self._post_process_example(selected_data, t0_dt)
        if selected_data.shape != self._shape_of_example:
            raise RuntimeError(
                "Example is wrong shape! "
                f"x_meters_center={x_meters_center}\n"
                f"y_meters_center={y_meters_center}\n"
                f"t0_dt={t0_dt}\n"
                f"expected shape={self._shape_of_example}\n"
                f"actual shape {selected_data.shape}"
            )

        return self._put_data_into_example(selected_data)

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
        # Shrink extra dims
        selected_data = selected_data.squeeze(axis=[0, 1])
        return selected_data
