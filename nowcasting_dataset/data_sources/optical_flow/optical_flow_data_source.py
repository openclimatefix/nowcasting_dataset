""" Optical Flow Data Source """
import logging
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_dataset.data_sources import DataSource
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow

_LOG = logging.getLogger(__name__)


@dataclass
class OpticalFlowDataSource(DataSource):
    """
    Optical Flow Data Source, computing flow between Satellite data

    history_minutes: Duration of historical data to use when computing the optical flow field.
        For example, set to 5 to use just two images: the t-1 and t0 images.  Set to 10 to compute
        the optical flow field separately for the image pairs (t-2, t-1), and (t-1, t0) and to
        use the mean optical flow field.
    forecast_minutes: Duration of the optical flow predictions.
    zarr_path: The location of the intermediate satellite data to compute optical flows with.
    input_image_size_pixels: The *input* image size (i.e. the image size to load off disk).
        This should be larger than output_image_size_pixels to provide sufficient border to mean
        that, even after the image has been "flowed", all edges of the output image are
        "real" pixels values, and not NaNs.
    output_image_size_pixels: The size of the output image.  The output image is a center-crop of
        the input image, after it has been "flowed".
    source_data_source_class: Either HRVSatelliteDataSource or SatelliteDataSource.
    channels: The satellite channels to compute optical flow for.
    """

    zarr_path: Union[Path, str]
    channels: Iterable[str]
    input_image_size_pixels: int = 64
    meters_per_pixel: int = 2000
    output_image_size_pixels: int = 32
    source_data_source_class_name: str = "SatelliteDataSource"

    def __post_init__(self):
        super().__post_init__()

        # Get round circular import problem
        from nowcasting_dataset.data_sources import HRVSatelliteDataSource, SatelliteDataSource

        _MAP_SATELLITE_DATA_SOURCE_NAME_TO_CLASS = {
            "HRVSatelliteDataSource": HRVSatelliteDataSource,
            "SatelliteDataSource": SatelliteDataSource,
        }

        source_data_source_class = _MAP_SATELLITE_DATA_SOURCE_NAME_TO_CLASS[
            self.source_data_source_class_name
        ]
        self.source_data_source = source_data_source_class(
            zarr_path=self.zarr_path,
            image_size_pixels=self.input_image_size_pixels,
            history_minutes=self.history_minutes,
            forecast_minutes=0,
            channels=self.channels,
            meters_per_pixel=self.meters_per_pixel,
        )

    def open(self):
        self.source_data_source.open()

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
        satellite_data: xr.Dataset = self.source_data_source.get_example(
            t0_dt=t0_dt, x_meters_center=x_meters_center, y_meters_center=y_meters_center
        )
        satellite_data = satellite_data["data"]
        return self._compute_and_return_optical_flow(satellite_data)

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
        t0_datetime_utc = satellite_data.isel(time=-1)["time"].values
        datetime_index_of_predictions = pd.date_range(
            t0_datetime_utc, periods=self.forecast_length, freq=self.sample_period_duration
        )

        # Select the center crop.
        # TODO: Generalise crop_center and use again here:
        border = (satellite_data.sizes["x"] - self.output_image_size_pixels) // 2
        satellite_data = satellite_data.isel(
            x=slice(border, satellite_data.sizes["x"] - border),
            y=slice(border, satellite_data.sizes["y"] - border),
        )
        return xr.DataArray(
            data=predictions,
            coords=(
                ("time", datetime_index_of_predictions),
                ("x", satellite_data.coords["x"].values),
                ("y", satellite_data.coords["y"].values),
                ("channels", satellite_data.coords["channels"].values),
            ),
            name="data",
        )

    def _compute_and_return_optical_flow(self, satellite_data: xr.DataArray) -> xr.DataArray:
        """
        Compute and return optical flow predictions for the example

        Args:
            satellite_data: Satellite DataArray of historical satellite images.

        Returns:
            The Tensor with the optical flow predictions for t0 to forecast horizon
        """
        n_channels = satellite_data.sizes["channels"]

        # Sanity check
        assert (
            len(satellite_data.coords["time"]) == self.history_length + 1
        ), f"{len(satellite_data.coords['time'])=} != {self.history_length+1=}"
        assert n_channels == len(self.channels), f"{n_channels=} != {len(self.channels)=}"

        # TODO: Use the correct dtype.
        prediction_block = np.full(
            shape=(
                self.forecast_length,
                self.output_image_size_pixels,
                self.output_image_size_pixels,
                n_channels,
            ),
            fill_value=np.NaN,
        )

        for channel_i in range(n_channels):
            # Compute optical flow field:
            sat_data_for_chan = satellite_data.isel(channels=channel_i)

            # Loop through pairs of historical images to compute optical flow fields:
            optical_flows = []
            # self.history_length does not include t0.
            for history_timestep in range(self.history_length):
                prev_image = sat_data_for_chan.isel(time=history_timestep).data
                next_image = sat_data_for_chan.isel(time=history_timestep + 1).data
                optical_flow = compute_optical_flow(prev_image, next_image)
                optical_flows.append(optical_flow)
            # Average predictions
            optical_flow = np.mean(optical_flows, axis=0)

            # Compute predicted images.
            t0_image = sat_data_for_chan.isel(time=-1).data
            for prediction_timestep in range(self.forecast_length):
                flow = optical_flow * (prediction_timestep + 1)
                warped_image = remap_image(image=t0_image, flow=flow)
                warped_image = crop_center(
                    warped_image,
                    self.output_image_size_pixels,
                    self.output_image_size_pixels,
                )
                prediction_block[prediction_timestep, :, :, channel_i] = warped_image

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

    # Docs: https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af  # noqa
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
