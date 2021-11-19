""" Topological DataSource """
import logging
from dataclasses import dataclass
from numbers import Number

import pandas as pd
import rioxarray
import xarray as xr
from rasterio.warp import Resampling

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.data_sources.data_source import ImageDataSource
from nowcasting_dataset.data_sources.topographic.topographic_model import Topographic
from nowcasting_dataset.geospatial import OSGB
from nowcasting_dataset.utils import OpenData

logger = logging.getLogger(__name__)


@dataclass
class TopographicDataSource(ImageDataSource):
    """Add topographic/elevation map features."""

    filename: str = None

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """Post init"""
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._shape_of_example = (
            image_size_pixels,
            image_size_pixels,
        )

        logger.info(f"Loading Topological data {self.filename}")

        with OpenData(file_name=self.filename) as filename:
            self._data = rioxarray.open_rasterio(
                filename=filename, parse_coordinates=True, masked=True
            )

        self._data = self._data.fillna(0)  # Set nodata values to 0 (mostly should be ocean)
        # Add CRS for later, topo maps are assumed to be in OSGB
        self._data.attrs["crs"] = OSGB
        # Distance between pixels, giving their spatial extant, in meters
        self._stored_pixel_size_meters = abs(self._data.coords["x"][1] - self._data.coords["x"][0])
        self._meters_per_pixel = meters_per_pixel

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return Topographic

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.  If not, raise a FileNotFoundError."""
        nd_fs_utils.check_path_exists(self.filename)

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> xr.Dataset:
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

        # change to dataset
        topo_xd = selected_data.to_dataset(name="data")

        return topo_xd

    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:
        """
        Post process the topographical data, removing an extra dim

        Args:
            selected_data: DataArray containing the topographic data
            t0_dt: Unused

        Returns:
            DataArray  removed first dimension
        """
        # Shrink extra dims
        selected_data = selected_data.squeeze()
        return selected_data
