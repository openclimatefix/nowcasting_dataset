"""Functions for computing the optical flow on the fly for satellite images"""
import logging
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.data_sources.transforms.transform import Transform
from nowcasting_dataset.dataset.batch import Batch

_LOG = logging.getLogger("nowcasting_dataset")


class OpticalFlowTransform(Transform):
    """
    Optical Flow Transform that adds optical flow images

    """

    final_image_size_pixels: Optional[int] = None

    def apply_transform(self, batch: Batch) -> Batch:
        """
        Calculate optical flow for the batch, and add to Batch

        Args:
            batch: Batch containing satellite data for optical flow

        Returns:
            Batch with optical flow added
        """
        batch.optical_flow = compute_optical_flow_for_batch(batch)
        return batch


def compute_optical_flow_for_batch(
    batch: Batch, final_image_size_pixels: Optional[int] = None
) -> xr.DataArray:
    """
    Computes the optical flow for satellite images in the batch

    Assumes metadata is also in Batch, for getting t0

    Args:
        batch: Batch containing at least metadata and satellite data

    Returns:
        Tensor containing the Optical Flow predictions
    """

    assert (
        batch.satellite is not None
    ), "Satellite data does not exist in batch, required for optical flow"
    assert batch.metadata is not None, "Metadata does not exist in batch, required for optical flow"

    if final_image_size_pixels is None:
        final_image_size_pixels = len(batch.satellite.x_index)

    # Only do optical flow for satellite data
    optical_flow_predictions = []
    for i in range(batch.batch_size):
        satellite_data: xr.DataArray = batch.satellite.sel(example=i)
        t0_dt = batch.metadata.t0_dt.values[i]
        optical_flow_predictions.append(
            _compute_and_return_optical_flow(
                satellite_data, t0_dt=t0_dt, final_image_size_pixels=final_image_size_pixels
            )
        )
    # Concatenate all the DataArrays
    dataarray = xr.concat(optical_flow_predictions, dim="example")
    return dataarray


def _update_dataarray_with_predictions(
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


def _get_number_future_timesteps(satellite_data: xr.DataArray, t0_dt: pd.Timestamp) -> int:
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
            flow = optical_flow * prediction_timestep + 1  # Otherwise first prediction would be 0
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


def _compute_optical_flow(t0_image: np.ndarray, previous_image: np.ndarray) -> np.ndarray:
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


def _remap_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
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


def crop_center(image, x_size, y_size):
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
