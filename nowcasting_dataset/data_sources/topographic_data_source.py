from nowcasting_dataset.data_sources.data_source import ImageDataSource, ZarrDataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from dataclasses import dataclass, InitVar
from numbers import Number
import pandas as pd
import xarray as xr
import numpy as np
import rioxarray

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
        self._data = rioxarray.open_rasterio(filename=self.filename, parse_coordinates=True)
        # Scale factor can be computed from distance between pixels, that would be it in meters
        self._stored_pixel_size_meters = abs(self._data.coords["x"][1] - self._data.coords["x"][0])
        print(
            self._stored_pixel_size_meters
        )  # Gives values of 892.053ish, not the 1km that is expected
        print(self._data)
        exit()
        self._scale_factor = int(meters_per_pixel / self._stored_pixel_size_meters)
        self._meters_per_pixel = meters_per_pixel
        self._data = self._data.to_array(dim=TOPOGRAPHIC_DATA)
        assert self._stored_pixel_size_meters <= meters_per_pixel, AttributeError(
            "The stored topographical map has a lower resolution than requested"
        )

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:

        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )
        selected_data = self._data.sel(
            x=slice(bounding_box.left, bounding_box.right),
            y=slice(bounding_box.top, bounding_box.bottom),
        )
        # Rescale here to the exact size, assumes that the above is good slice

        print(selected_data)
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
