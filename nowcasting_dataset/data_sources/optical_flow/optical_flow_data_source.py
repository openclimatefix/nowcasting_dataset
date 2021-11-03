""" Optical Flow Data Source """
import logging
from dataclasses import InitVar, dataclass
from numbers import Number
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SatelliteDataSource

_LOG = logging.getLogger("nowcasting_dataset")

IMAGE_BUFFER_SIZE = 16


@dataclass
class OpticalFlowDataSource(SatelliteDataSource):
    """
    Optical Flow Data Source, computing flow between Satellite data

    Pads image size to allow for cropping out NaN values
    """

    channels: Optional[Iterable[str]] = SAT_VARIABLE_NAMES
    previous_timestep_for_flow: int = 1
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """ Post Init  Add 16 pixels to each side of the image"""
        super().__post_init__(image_size_pixels + (2 * IMAGE_BUFFER_SIZE), meters_per_pixel)
        n_channels = len(self.channels)
        self._cache = {}
        self._shape_of_example = (
            self.forecast_length,
            image_size_pixels,
            image_size_pixels,
            n_channels,
        )

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

        # rename 'variable' to 'channels'
        selected_data = selected_data.rename({"variable": "channels"})

        # Compute optical flow for the timesteps
        # Get Optical Flow for the pre-t0 time, and applying the t0-previous_timesteps_per_flow to
        # t0 optical flow for forecast steps in the future
        # Creates a pyramid of optical flows for all timesteps up to t0, and apply predictions
        # for all future timesteps for each of them
        # Compute optical flow per channel, as it might be different
        selected_data: xr.DataArray = self._compute_and_return_optical_flow(
            selected_data, t0_dt=t0_dt
        )

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

        return selected_data

    def _compute_previous_timestep(
        self, satellite_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> pd.Timestamp:
        """
        Get timestamp of previous

        Args:
            satellite_data: Satellite data to use
            t0_dt: Timestamp

        Returns:
            The previous timesteps
        """
        satellite_data = satellite_data.where(satellite_data.time <= t0_dt, drop=True)
        return satellite_data.isel(
            time=len(satellite_data.time) - self.previous_timestep_for_flow
        ).time.values

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
        return len(satellite_data.coords["time"])

    def _compute_and_return_optical_flow(
        self, satellite_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:
        """
        Compute and return optical flow predictions for the example

        Args:
            satellite_data: Satellite DataArray
            t0_dt: t0 timestamp

        Returns:
            The xr.DataArray with the optical flow predictions for t0 to forecast horizon
        """

        prediction_dictionary = {}
        # Get the previous timestamp
        previous_timestamp = self._compute_previous_timestep(satellite_data, t0_dt=t0_dt)
        for prediction_timestep in range(self._get_number_future_timesteps(satellite_data, t0_dt)):
            predictions = []
            for channel in satellite_data.coords["channels"]:
                channel_images = satellite_data.sel(channels=channel)
                t0_image = channel_images.sel(time=t0_dt).values
                previous_image = channel_images.sel(time=previous_timestamp).values
                optical_flow = self._compute_optical_flow(t0_image, previous_image)
                # Do predictions now
                flow = optical_flow * prediction_timestep
                warped_image = self._remap_image(t0_image, flow)
                warped_image = crop_center(
                    warped_image,
                    self._square.size_pixels - (2 * IMAGE_BUFFER_SIZE),
                    self._square.size_pixels - (2 * IMAGE_BUFFER_SIZE),
                )
                predictions.append(warped_image)
            # Add the block of predictions for all channels
            prediction_dictionary[prediction_timestep] = np.concatenate(predictions, axis=-1)
        # Make a block of T, H, W, C ordering
        prediction = np.stack(
            [prediction_dictionary[k] for k in prediction_dictionary.keys()], axis=0
        )
        if len(self.channels) == 1:  # Only case where another channel needs to be added
            prediction = np.expand_dims(prediction, axis=-1)
        # Swap out data for the future part of the dataarray
        dataarray = self._update_dataarray_with_predictions(
            satellite_data, predictions=prediction, t0_dt=t0_dt
        )
        return dataarray

    def _update_dataarray_with_predictions(
        self, satellite_data: xr.DataArray, predictions: np.ndarray, t0_dt: pd.Timestamp
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
        satellite_data = satellite_data.isel(
            x=slice(IMAGE_BUFFER_SIZE, self._square.size_pixels - IMAGE_BUFFER_SIZE),
            y=slice(IMAGE_BUFFER_SIZE, self._square.size_pixels - IMAGE_BUFFER_SIZE),
        )
        dataarray = xr.DataArray(
            data=predictions,
            dims=satellite_data.dims,
            coords=satellite_data.coords,
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

    def _dataset_to_data_source_output(output: xr.Dataset) -> OpticalFlow:
        return OpticalFlow(output)


def crop_center(img, cropx, cropy):
    """
    Crop center of numpy image

    Args:
        img: Image to crop
        cropx: Size in x direction
        cropy: Size in y direction

    Returns:
        The cropped image
    """
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]
