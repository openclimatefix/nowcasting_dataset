""" Optical Flow Data Source """
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.dataset.batch
from nowcasting_dataset.data_sources.data_source import DerivedDataSource
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput

_LOG = logging.getLogger("nowcasting_dataset")


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
        batch: nowcasting_dataset.dataset.batch.Batch,
        example_idx: int,
        t0_datetime: pd.Timestamp,
        **kwargs
    ) -> DataSourceOutput:
        """
        Get Optical Flow Example data

        Args:
            batch: Batch containing satellite and metadata at least
            example_idx: The example to load and use
            t0_datetime: t0 datetime for the example

        Returns: Example Data

        """

        if self.image_size_pixels is None:
            self.image_size_pixels = len(batch.satellite.x_index)

        # Only do optical flow for satellite data
        self._data: xr.DataArray = batch.satellite.sel(example=example_idx)

        selected_data = self._compute_and_return_optical_flow(self._data, t0_dt=t0_datetime)

        return selected_data

    def _update_dataarray_with_predictions(
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
        # Combine all channels for a single timestep
        satellite_data = satellite_data.isel(
            time_index=slice(
                satellite_data.sizes["time_index"] - predictions.shape[0],
                satellite_data.sizes["time_index"],
            )
        )
        # Make sure its the correct size
        buffer = (satellite_data.sizes["x_index"] - self.image_size_pixels) // 2
        satellite_data = satellite_data.isel(
            x_index=slice(buffer, satellite_data.sizes["x_index"] - buffer),
            y_index=slice(buffer, satellite_data.sizes["y_index"] - buffer),
        )
        dataarray = xr.DataArray(
            data=predictions,
            dims={
                "time_index": satellite_data.dims["time_index"],
                "x_index": satellite_data.dims["x_index"],
                "y_index": satellite_data.dims["y_index"],
                "channels_index": satellite_data.dims["channels_index"],
            },
            coords={
                "time_index": satellite_data.coords["time_index"],
                "x_index": satellite_data.coords["x_index"],
                "y_index": satellite_data.coords["y_index"],
                "channels_index": satellite_data.coords["channels_index"],
            },
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
        future_timesteps = self._get_number_future_timesteps(satellite_data, t0_dt)
        historical_satellite_data: xr.DataArray = self._get_previous_timesteps(
            satellite_data,
            t0_dt=t0_dt,
        )
        assert (
            len(historical_satellite_data.coords["time_index"])
            - self.number_previous_timesteps_to_use
            - 1
        ) >= 0, "Trying to compute flow further back than the number of historical timesteps"
        prediction_block = np.zeros(
            (
                future_timesteps,
                self.image_size_pixels,
                self.image_size_pixels,
                satellite_data.sizes["channels_index"],
            )
        )
        for prediction_timestep in range(future_timesteps):
            for channel in range(0, len(historical_satellite_data.coords["channels_index"])):
                t0 = historical_satellite_data.sel(channels_index=channel)
                previous = historical_satellite_data.sel(channels_index=channel)
                optical_flows = []
                for i in range(
                    len(historical_satellite_data.coords["time_index"]) - 1,
                    len(historical_satellite_data.coords["time_index"])
                    - self.number_previous_timesteps_to_use
                    - 1,
                    -1,
                ):
                    t0_image = t0.isel(time_index=i).data.values
                    previous_image = previous.isel(time_index=i - 1).data.values
                    optical_flow = compute_optical_flow(t0_image, previous_image)
                    optical_flows.append(optical_flow)
                # Average predictions
                optical_flow = np.mean(optical_flows, axis=0)
                # Do predictions now
                t0_image = t0.isel(time_index=-1).data.values
                flow = optical_flow * (prediction_timestep + 1)
                warped_image = remap_image(t0_image, flow)
                warped_image = crop_center(
                    warped_image,
                    self.image_size_pixels,
                    self.image_size_pixels,
                )
                prediction_block[prediction_timestep, :, :, channel] = warped_image
        dataarray = self._update_dataarray_with_predictions(
            satellite_data=self._data, predictions=prediction_block
        )
        return dataarray


def compute_optical_flow(t0_image: np.ndarray, previous_image: np.ndarray) -> np.ndarray:
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
    return cv2.remap(
        src=image,
        map1=remap,
        map2=None,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.NaN,
    )


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
