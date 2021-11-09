""" Optical Flow Data Source """
import logging
from concurrent import futures
from dataclasses import InitVar, dataclass
from numbers import Number
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.time as nd_time
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow
from nowcasting_dataset.dataset.xr_utils import join_list_data_array_to_batch_dataset

_LOG = logging.getLogger("nowcasting_dataset")


@dataclass
class OpticalFlowDataSource(ZarrDataSource):
    """
    Optical Flow Data Source, computing flow between Satellite data

    zarr_path: Must start with 'gs://' if on GCP.
    """

    zarr_path: str = None
    previous_timestep_for_flow: int = 1
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """Post Init"""
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._cache = {}
        self._shape_of_example = (
            self._total_seq_length,
            image_size_pixels,
            image_size_pixels,
            2,
        )

    def open(self) -> None:
        """
        Open Satellite data

        We don't want to open_sat_data in __init__.
        If we did that, then we couldn't copy SatelliteDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        self._data = self._open_data()
        self._data = self._data.sel(variable=list(self.channels))

    def _open_data(self) -> xr.DataArray:
        return open_sat_data(zarr_path=self.zarr_path, consolidated=self.consolidated)

    def get_batch(
        self,
        t0_datetimes: pd.DatetimeIndex,
        x_locations: Iterable[Number],
        y_locations: Iterable[Number],
    ) -> OpticalFlow:
        """
        Get batch data

        Load the first _n_timesteps_per_batch concurrently.  This
        loads the timesteps from disk concurrently, and fills the
        cache.  If we try loading all examples
        concurrently, then SatelliteDataSource will try reading from
        empty caches, and things are much slower!

        Args:
            t0_datetimes: list of timestamps for the datetime of the batches. The batch will also
                include data for historic and future depending on `history_minutes` and
                `future_minutes`.
            x_locations: x center batch locations
            y_locations: y center batch locations

        Returns: Batch data

        """
        # Load the first _n_timesteps_per_batch concurrently.  This
        # loads the timesteps from disk concurrently, and fills the
        # cache.  If we try loading all examples
        # concurrently, then SatelliteDataSource will try reading from
        # empty caches, and things are much slower!
        zipped = list(zip(t0_datetimes, x_locations, y_locations))
        batch_size = len(t0_datetimes)

        with futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_examples = []
            for coords in zipped[: self.n_timesteps_per_batch]:
                t0_datetime, x_location, y_location = coords
                future_example = executor.submit(
                    self.get_example, t0_datetime, x_location, y_location
                )
                future_examples.append(future_example)
            examples = [future_example.result() for future_example in future_examples]

        # Load the remaining examples.  This should hit the DataSource caches.
        for coords in zipped[self.n_timesteps_per_batch :]:
            t0_datetime, x_location, y_location = coords
            example = self.get_example(t0_datetime, x_location, y_location)
            examples.append(example)

        output = join_list_data_array_to_batch_dataset(examples)

        self._cache = {}

        return OpticalFlow(output)

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> DataSourceOutput:
        """
        Get Optical Flow Example data

        Args:
            t0_dt: list of timestamps for the datetime of the batches. The batch will also include
                data for historic and future depending on `history_minutes` and `future_minutes`.
            x_meters_center: x center batch locations
            y_meters_center: y center batch locations

        Returns: Example Data

        """
        selected_data = self._get_time_slice(t0_dt)
        bounding_box = self._square.bounding_box_centered_on(
            x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )
        selected_data = selected_data.sel(
            x=slice(bounding_box.left, bounding_box.right),
            y=slice(bounding_box.top, bounding_box.bottom),
        )

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
                f"times are {selected_data.time}\n"
                f"expected shape={self._shape_of_example}\n"
                f"actual shape {selected_data.shape}"
            )

        # rename 'variable' to 'channels'
        selected_data = selected_data.rename({"variable": "channels"})

        # Compute optical flow for the timesteps
        # Get Optical Flow for the pre-t0 time, and applying the t0-previous_timesteps_per_flow to
        # t0 optical flow for forecast steps in the future
        # Creates a pyramid of optical flows for all timesteps up to t0, and apply predictions
        # for all future timesteps for each of them
        # Compute optical flow per channel, as it might be different
        selected_data = self._compute_and_return_optical_flow(selected_data, t0_dt=t0_dt)

        return selected_data

    def _update_dataarray_with_predictions(
        self,
        satellite_data: xr.DataArray,
        predictions: np.ndarray,
        t0_dt: pd.Timestamp,
        final_image_size_pixels: int,
    ) -> xr.DataArray:
        """
        Updates the dataarray with predictions

         Additionally, changes the temporal size to t0+1 to forecast horizon

        Args:
            satellite_data: Satellite data
            predictions: Predictions from the optical flow

        Returns:
            The Xarray DataArray with the optical flow predictions
        """

        # Combine all channels for a single timestep
        satellite_data = satellite_data.where(satellite_data.time > t0_dt, drop=True)
        # Make sure its the correct size
        buffer = satellite_data.sizes["x"] - final_image_size_pixels // 2
        satellite_data = satellite_data.isel(
            x=slice(buffer, satellite_data.sizes["x"] - buffer),
            y=slice(buffer, satellite_data.sizes["y"] - buffer),
        )
        dataarray = xr.DataArray(
            data=predictions,
            dims=satellite_data.dims,
            coords=satellite_data.coords,
        )

        return dataarray

    def _get_previous_timesteps(
        self,
        satellite_data: xr.DataArray,
        t0_dt: pd.Timestamp,
    ) -> xr.DataArray:
        """
        Get timestamp of previous

        Args:
            satellite_data: Satellite data to use
            t0_dt: Timestamp

        Returns:
            The previous timesteps
        """
        satellite_data = satellite_data.where(satellite_data.time <= t0_dt, drop=True)
        return satellite_data

    def _get_number_future_timesteps(
        self, satellite_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> int:
        """
        Get number of future timestamps

        Args:
            satellite_data: Satellite data to use
            t0_dt: The timestamp of the t0 image

        Returns:
            The number of future timesteps
        """
        satellite_data = satellite_data.where(satellite_data.time > t0_dt, drop=True)
        return len(satellite_data.coords["time_index"])

    def _compute_and_return_optical_flow(
        self,
        satellite_data: xr.DataArray,
        t0_dt: pd.Timestamp,
        final_image_size_pixels: int,
    ) -> xr.DataArray:
        """
        Compute and return optical flow predictions for the example

        Args:
            satellite_data: Satellite DataArray
            t0_dt: t0 timestamp

        Returns:
            The Tensor with the optical flow predictions for t0 to forecast horizon
        """

        # Get the previous timestamp
        future_timesteps = _get_number_future_timesteps(satellite_data, t0_dt)
        satellite_data: xr.DataArray = _get_previous_timesteps(
            satellite_data,
            t0_dt=t0_dt,
        )
        prediction_block = np.zeros(
            (
                future_timesteps,
                final_image_size_pixels,
                final_image_size_pixels,
                satellite_data.sizes["channels_index"],
            )
        )
        for prediction_timestep in range(future_timesteps):
            for channel in range(0, len(satellite_data.coords["channels_index"]), 4):
                # Optical Flow works with RGB images, so chunking channels for it to be faster
                channel_images = satellite_data.sel(channels_index=slice(channel, channel + 3))
                # Extra 1 in shape from time dimension, so removing that dimension
                t0_image = channel_images.isel(
                    time_index=len(satellite_data.time_index) - 1
                ).data.values
                previous_image = channel_images.isel(
                    time_index=len(satellite_data.time_index) - 2
                ).data.values
                optical_flow = _compute_optical_flow(t0_image, previous_image)
                # Do predictions now
                flow = (
                    optical_flow * prediction_timestep + 1
                )  # Otherwise first prediction would be 0
                warped_image = _remap_image(t0_image, flow)
                warped_image = crop_center(
                    warped_image,
                    final_image_size_pixels,
                    final_image_size_pixels,
                )
                prediction_block[prediction_timestep, :, :, channel : channel + 4] = warped_image
        # Convert to correct C, T, H, W order
        prediction_block = np.permute(prediction_block, [3, 0, 1, 2])
        dataarray = _update_dataarray_with_predictions(
            satellite_data=satellite_data, predictions=prediction_block, t0_dt=t0_dt
        )
        return dataarray

    def _compute_optical_flow(self, t0_image: np.ndarray, previous_image: np.ndarray) -> np.ndarray:
        """
        Compute the optical flow for a set of images

        Args:
            t0_image: t0 image
            previous_image: previous image to compute optical flow with

        Returns:
            Optical Flow field
        """
        # Input images have to be single channel and between 0 and 1
        image_min = np.min([t0_image, previous_image])
        image_max = np.max([t0_image, previous_image])
        t0_image -= image_min
        t0_image /= image_max
        previous_image -= image_min
        previous_image /= image_max
        t0_image = cv2.cvtColor(t0_image.astype(np.float32), cv2.COLOR_RGBA2GRAY)
        previous_image = cv2.cvtColor(previous_image.astype(np.float32), cv2.COLOR_RGBA2GRAY)
        return cv2.calcOpticalFlowFarneback(
            prev=previous_image,
            next=t0_image,
            flow=None,
            pyr_scale=0.5,
            levels=2,
            winsize=40,
            iterations=3,
            poly_n=5,
            poly_sigma=0.7,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )

    def _remap_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Takes an image and warps it forwards in time according to the flow field.

        Args:
            image: The grayscale image to warp.
            flow: A 3D array.  The first two dimensions must be the same size as the first two
                dimensions of the image.  The third dimension represented the x and y displacement.

        Returns:  Warped image.  The border has values np.NaN.
        """
        # Adapted from https://github.com/opencv/opencv/issues/11068
        height, width = flow.shape[:2]
        remap = -flow.copy()
        remap[..., 0] += np.arange(width)  # map_x
        remap[..., 1] += np.arange(height)[:, np.newaxis]  # map_y
        return cv2.remap(
            src=image,
            map1=remap,
            map2=None,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.NaN,
        )

    def crop_center(self, image, x_size, y_size):
        """
        Crop center of numpy image

        Args:
            image: Image to crop
            x_size: Size in x direction
            y_size: Size in y direction

        Returns:
            The cropped image
        """
        y, x, channels = image.shape
        startx = x // 2 - (x_size // 2)
        starty = y // 2 - (y_size // 2)
        return image[starty : starty + y_size, startx : startx + x_size]

    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:

        selected_data.data = selected_data.data.astype(np.float32)

        return selected_data
