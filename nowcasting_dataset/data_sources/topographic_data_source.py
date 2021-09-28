from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.consts import TOPOGRAPHIC_DATA
from nowcasting_dataset.geospatial import OSGB
from rasterio.warp import Resampling
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

    filename: str = None
    normalize: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._shape_of_example = (
            image_size_pixels,
            image_size_pixels,
        )
        self._data = rioxarray.open_rasterio(
            filename=self.filename, parse_coordinates=True, masked=True
        )
        self._data = self._data.fillna(0)  # Set nodata values to 0 (mostly should be ocean)
        # Add CRS for later, topo maps are assumed to be in OSGB
        self._data.attrs["crs"] = OSGB
        # Distance between pixels, giving their spatial extant, in meters
        self._stored_pixel_size_meters = abs(self._data.coords["x"][1] - self._data.coords["x"][0])
        self._meters_per_pixel = meters_per_pixel

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> Example:
        """
        Get a single example

        Args:
            t0_dt: Current datetime for the example, unused
            x_meters_center: Center of the example in meters in the x direction in OSGB coordinates
            y_meters_center: Center of the example in meters in the y direction in OSGB coordinates

        Returns:
            Example containing topographic data for the selected area
        """

        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )
        selected_data = self._data.sel(
            x=slice(bounding_box.left, bounding_box.right),
            y=slice(bounding_box.top, bounding_box.bottom),
        )
        if self._stored_pixel_size_meters != self._meters_per_pixel:
            # Rescale here to the exact size, assumes that the above is good slice
            # Useful if using different spatially sized grids
            selected_data = selected_data.rio.reproject(
                dst_crs=selected_data.attrs["crs"],
                shape=(self._square.size_pixels, self._square.size_pixels),
                resampling=Resampling.bilinear,
            )

        # selected_data is likely to have 1 too many pixels in x and y
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
        """
        Insert the data and coordinates into an Example

        Args:
            selected_data: DataArray containing the data to insert

        Returns:
            Example containing the Topographic data
        """
        return Example(
            topo_data=selected_data,
            topo_x_coords=selected_data.x,
            topo_y_coords=selected_data.y,
        )

    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:
        """
        Post process the topographical data, removing an extra dim and optionally
        normalizing

        Args:
            selected_data: DataArray containing the topographic data
            t0_dt: Unused

        Returns:
            DataArray with optionally normalized data, and removed first dimension
        """
        if self.normalize:
            selected_data = selected_data - TOPO_MEAN
            selected_data = selected_data / TOPO_STD
        # Shrink extra dims
        selected_data = selected_data.squeeze()
        return selected_data
