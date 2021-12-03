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
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow

_LOG = logging.getLogger(__name__)


@dataclass
class OpticalFlowDataSource(DataSource):
    """
    Optical Flow Data Source.

    Predicts future satellite imagery by computing the "flow" between consecutive pairs of
    satellite images and using that flow to "warp" the most recent satellite image (the "t0 image")
    to predict future satellite images.

    Optical flow is surprisingly effective at predicting future satellite images over time horizons
    out to about 2 hours.  After 2 hours the predictions start to go a bit crazy.  There are some
    notable problems with optical flow predictions:

    1) Optical flow doesn't understand that clouds grow, shrink, appear from "nothing", and disappear
       into "nothing".  Optical flow just moves pixels around.
    2) Optical flow doesn't understand that satellite images tend to get brighter as the sun rises
       and darker as the sun sets.

    Arguments for the OpticalFlowDataSource constructor:

    history_minutes: Duration of historical data to use when computing the optical flow field.
        For example, set to 5 to use just two images: the t-1 and t0 images.  Set to 10 to compute
        the optical flow field separately for the image pairs (t-2, t-1) and (t-1, t0) and to
        use the mean optical flow field.
    forecast_minutes: Duration of the optical flow predictions.
    zarr_path: The location of the intermediate satellite data to compute optical flows with.
    input_image_size_pixels: The *input* image size (i.e. the image size to load off disk).
        This should be significantly larger than output_image_size_pixels to provide sufficient
        border so that, even after the image has been "flowed", all edges of the output image are
        "real" pixels values, and not NaNs.  For a forecast horizon of 120 minutes, and an output
        image size of 24 pixels, we have found that the input image size needs to be at least
        128 pixels.
    output_image_size_pixels: The size of the output image.  The output image is a center-crop of
        the input image after it has been "flowed".
    source_data_source_class_name: Either HRVSatelliteDataSource or SatelliteDataSource.
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
    ) -> xr.Dataset:
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
        optical_flow_data_array = self._compute_and_return_optical_flow(satellite_data)
        return optical_flow_data_array.to_dataset()

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
        Puts optical flow predictions into an xr.DataArray.

        Args:
            satellite_data: Satellite data
            predictions: Predictions from the optical flow

        Returns:
            The Xarray DataArray with the optical flow predictions
        """
        # Generate a pd.DatetimeIndex for the optical flow predictions.
        t0_datetime_utc = satellite_data.isel(time=-1)["time"].values
        t1_datetime_utc = t0_datetime_utc + self.sample_period_duration
        datetime_index_of_predictions = pd.date_range(
            t1_datetime_utc, periods=self.forecast_length, freq=self.sample_period_duration
        )

        # Select the center crop.
        satellite_data_cropped = satellite_data.isel(time_index=0, channels_index=0)
        satellite_data_cropped = crop_center(satellite_data_cropped, self.output_image_size_pixels)

        # Put into DataArray
        return xr.DataArray(
            data=predictions,
            coords=(
                ("time", datetime_index_of_predictions),
                ("x", satellite_data_cropped.coords["x"].values),
                ("y", satellite_data_cropped.coords["y"].values),
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
            fill_value=-1,
            dtype=np.int16,
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
                warped_image = crop_center(warped_image, self.output_image_size_pixels)
                prediction_block[prediction_timestep, :, :, channel_i] = warped_image

        data_array = self._put_predictions_into_data_array(
            satellite_data=satellite_data, predictions=prediction_block
        )
        return data_array


def _convert_arrays_to_uint8(*arrays: tuple[np.ndarray]) -> tuple[np.ndarray]:
    """Convert multiple arrays to uint8, using the same min and max to scale all arrays."""
    # First, stack into a single numpy array so we can work on all images at the same time:
    stacked = np.stack(arrays)

    # Rescale pixel values to be in the range [0, 1]:
    stacked -= stacked.min()
    stacked /= stacked.max()

    # Convert to uint8 (uint8 can represent integers in the range [0, 255]):
    stacked *= 255
    stacked = stacked.astype(np.uint8)

    return tuple(stacked)


def compute_optical_flow(prev_image: np.ndarray, next_image: np.ndarray) -> np.ndarray:
    """
    Compute the optical flow for a set of images

    Args:
        prev_image, next_image: A pair of images representing two timesteps.  This algorithm
            will estimate the "movement" across these two timesteps.  Both images must be the
            same dtype.

    Returns:
        Dense optical flow field: A 3D array.  The first two dimension are the same size as the
            input images.  The third dimension is of size 2 and represents the
            displacement in x and y.
    """
    assert prev_image.dtype == next_image.dtype

    # cv2.calcOpticalFlowFarneback expects images to be uint8:
    prev_image, next_image = _convert_arrays_to_uint8(prev_image, next_image)

    # Docs for cv2.calcOpticalFlowFarneback:
    # https://docs.opencv.org/4.5.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
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


def remap_image(
    image: np.ndarray,
    flow: np.ndarray,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """
    Takes an image and warps it forwards in time according to the flow field.

    Args:
        image: The grayscale image to warp.
        flow: A 3D array.  The first two dimensions must be the same size as the first two
            dimensions of the image.  The third dimension represented the x and y displacement.
        border_mode: One of cv2's BorderTypes such as cv2.BORDER_CONSTANT or cv2.BORDER_REPLICATE.
            If border_mode=cv2.BORDER_CONSTANT then the border will be set to -1.
            For details of other border_mode settings, see the Open CV docs here:
            docs.opencv.org/4.5.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5

    Returns:  Warped image.
    """
    # Adapted from https://github.com/opencv/opencv/issues/11068
    height, width = flow.shape[:2]
    remap = -flow.copy()
    remap[..., 0] += np.arange(width)  # map_x
    remap[..., 1] += np.arange(height)[:, np.newaxis]  # map_y
    # remap docs:
    # docs.opencv.org/4.5.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
    # TODO: Maybe use integer remap: docs say that might be faster?
    remapped_image = cv2.remap(
        src=image,
        map1=remap,
        map2=None,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=-1,
    )
    return remapped_image


def crop_center(image: np.ndarray, output_image_size_pixels: int) -> np.ndarray:
    """
    Crop center of a 2D numpy image.

    Args:
        image: The input image to crop.
        output_image_size_pixels: The requested size of the output image.
    Returns:
        The cropped image, of size output_image_size_pixels x output_image_size_pixels
    """
    input_size_y, input_size_x = image.shape
    assert input_size_x >= output_image_size_pixels
    assert input_size_y >= output_image_size_pixels
    half_output_image_size_pixels = output_image_size_pixels // 2
    start_x = (input_size_x // 2) - half_output_image_size_pixels
    start_y = (input_size_y // 2) - half_output_image_size_pixels
    end_x = start_x + output_image_size_pixels
    end_y = start_y + output_image_size_pixels
    return image[start_y:end_y, start_x:end_x]
