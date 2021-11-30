""" Optical Flow Data Source """
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.data_sources.data_source import DerivedDataSource
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow

_LOG = logging.getLogger(__name__)


@dataclass
class OpticalFlowDataSource(DerivedDataSource):
    """
    Optical Flow Data Source, computing flow between Satellite data

    number_previous_timesteps_to_use: Number of previous timesteps to use, i.e. if 1, only uses the
        flow between t-1 and t0 images, if 3, computes the flow between (t-3,t-2),(t-2,t-1),
        and (t-1,t0) image pairs and uses the mean optical flow for future timesteps.
    """

    number_previous_timesteps_to_use: int = 1
    image_size_pixels: Optional[int] = None

    def get_example(
        self,
        batch,  # Of type nowcasting_dataset.dataset.batch.Batch.  But we can't use
        # an "actual" type hint here otherwise we get a circular import error!
        example_idx: int,
        t0_dt: pd.Timestamp,
        **kwargs
    ) -> DataSourceOutput:
        """
        Get Optical Flow Example data

        Args:
            batch: nowcasting_dataset.dataset.batch.Batch containing satellite and metadata at least
            example_idx: The example to load and use
            t0_dt: t0 datetime for the example

        Returns: Example Data

        """

        if self.image_size_pixels is None:
            self.image_size_pixels = len(batch.satellite.x_index)

        # Only do optical flow for satellite data
        # TODO: Enable this to work with hrvsatellite too.
        satellite_data: xr.DataArray = batch.satellite.sel(example=example_idx)
        return self._compute_and_return_optical_flow(satellite_data, t0_datetime_utc=t0_dt)

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return OpticalFlow

    def _put_predictions_into_data_array(
        self,
        satellite_data: xr.DataArray,
        predictions: np.ndarray,
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
        # Select the timesteps for the optical flow predictions.
        satellite_data = satellite_data.isel(
            time_index=slice(
                satellite_data.sizes["time_index"] - predictions.shape[0],
                satellite_data.sizes["time_index"],
            )
        )
        # Select the center crop.
        border = (satellite_data.sizes["x_index"] - self.image_size_pixels) // 2
        satellite_data = satellite_data.isel(
            x_index=slice(border, satellite_data.sizes["x_index"] - border),
            y_index=slice(border, satellite_data.sizes["y_index"] - border),
        )
        return xr.DataArray(
            data=predictions,
            coords=(
                ("time_index", satellite_data.coords["time_index"].values),
                ("x_index", satellite_data.coords["x_index"].values),
                ("y_index", satellite_data.coords["y_index"].values),
                ("channels_index", satellite_data.coords["channels_index"].values),
            ),
            name="data",
        )

    def _get_previous_timesteps(
        self,
        satellite_data: xr.DataArray,
        t0_datetime_utc: pd.Timestamp,
    ) -> xr.DataArray:
        """
        Get timestamp of previous

        Args:
            satellite_data: Satellite data to use
            t0_datetime_utc: Timestamp

        Returns:
            The previous timesteps
        """
        satellite_data = satellite_data.where(satellite_data.time <= t0_datetime_utc, drop=True)
        return satellite_data

    def _get_number_future_timesteps(
        self, satellite_data: xr.DataArray, t0_datetime_utc: pd.Timestamp
    ) -> int:
        """
        Get number of future timestamps

        Args:
            satellite_data: Satellite data to use
            t0_datetime_utc: The timestamp of the t0 image

        Returns:
            The number of future timesteps
        """
        satellite_data = satellite_data.where(satellite_data.time > t0_datetime_utc, drop=True)
        return len(satellite_data.coords["time_index"])

    def _compute_and_return_optical_flow(
        self,
        satellite_data: xr.DataArray,
        t0_datetime_utc: pd.Timestamp,
    ) -> xr.DataArray:
        """
        Compute and return optical flow predictions for the example

        Args:
            satellite_data: Satellite DataArray
            t0_datetime_utc: t0 timestamp

        Returns:
            The Tensor with the optical flow predictions for t0 to forecast horizon
        """

        # Get the previous timestamp
        future_timesteps = self._get_number_future_timesteps(satellite_data, t0_datetime_utc)
        historical_satellite_data: xr.DataArray = self._get_previous_timesteps(
            satellite_data,
            t0_datetime_utc=t0_datetime_utc,
        )
        assert (
            len(historical_satellite_data.coords["time_index"])
            - self.number_previous_timesteps_to_use
            - 1
        ) >= 0, "Trying to compute flow further back than the number of historical timesteps"

        # TODO: Use the correct dtype.
        n_channels = satellite_data.sizes["channels_index"]
        prediction_block = np.full(
            shape=(
                future_timesteps,
                self.image_size_pixels,
                self.image_size_pixels,
                n_channels,
            ),
            fill_value=np.NaN,
        )

        for channel in range(n_channels):
            # Compute optical flow field:
            historical_sat_data_for_chan = historical_satellite_data.isel(channels_index=channel)

            # Loop through pairs of historical images to compute optical flow fields:
            optical_flows = []
            n_historical_timesteps = len(historical_satellite_data.coords["time_index"])
            end_time_i = n_historical_timesteps
            start_time_i = end_time_i - self.number_previous_timesteps_to_use
            for time_i in range(start_time_i, end_time_i):
                prev_image = historical_sat_data_for_chan.isel(time_index=time_i - 1).data.values
                next_image = historical_sat_data_for_chan.isel(time_index=time_i).data.values
                optical_flow = compute_optical_flow(prev_image, next_image)
                optical_flows.append(optical_flow)
            # Average predictions
            optical_flow = np.mean(optical_flows, axis=0)

            # Compute predicted images.
            t0_image = historical_sat_data_for_chan.isel(time_index=-1).data.values
            for prediction_timestep in range(future_timesteps):
                flow = optical_flow * (prediction_timestep + 1)
                warped_image = remap_image(image=t0_image, flow=flow)
                warped_image = crop_center(
                    warped_image,
                    self.image_size_pixels,
                    self.image_size_pixels,
                )
                prediction_block[prediction_timestep, :, :, channel] = warped_image

        data_array = self._put_predictions_into_data_array(
            satellite_data=satellite_data, predictions=prediction_block
        )
        return data_array


def compute_optical_flow(prev_image: np.ndarray, next_image: np.ndarray) -> np.ndarray:
    """
    Compute the optical flow for a set of images

    Args:
        t0_image: t0 image
        previous_image: previous image to compute optical flow with

    Returns:
        Optical Flow field
    """
    # Input images have to be single channel and uint8.
    # TODO: Refactor this!
    image_min = np.min([prev_image, next_image])
    image_max = np.max([prev_image, next_image])
    prev_image = prev_image - image_min
    prev_image = prev_image / (image_max - image_min)
    prev_image = prev_image * 255
    prev_image = prev_image.astype(np.uint8)
    next_image = next_image - image_min
    next_image = next_image / (image_max - image_min)
    next_image = next_image * 255
    next_image = next_image.astype(np.uint8)

    # Docs: https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af  # nopa
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_image,
        next=next_image,
        flow=None,
        pyr_scale=0.5,
        levels=2,
        winsize=40,
        iterations=3,
        poly_n=5,
        poly_sigma=0.7,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    return flow


def remap_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
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
    # remap docs: https://docs.opencv.org/4.5.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4  # noqa
    # TODO: Maybe use integer remap: docs say that might be faster?
    remapped_image = cv2.remap(
        src=image,
        map1=remap,
        map2=None,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.NaN,
    )
    return remapped_image


def crop_center(image: np.ndarray, x_size: int, y_size: int) -> np.ndarray:
    """
    Crop center of numpy image

    Args:
        image: Image to crop
        x_size: Size in x direction
        y_size: Size in y direction

    Returns:
        The cropped image
    """
    y, x = image.shape
    startx = x // 2 - (x_size // 2)
    starty = y // 2 - (y_size // 2)
    return image[starty : starty + y_size, startx : startx + x_size]
