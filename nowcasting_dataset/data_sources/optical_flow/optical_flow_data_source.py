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

import nowcasting_dataset.filesystem.utils as nd_fs_utils
from nowcasting_dataset.data_sources import DataSource
from nowcasting_dataset.data_sources.optical_flow.format_images import crop_center, remap_image
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow
from nowcasting_dataset.dataset.xr_utils import convert_arrays_to_uint8

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

    1) Optical flow doesn't understand that clouds grow, shrink, appear from "nothing", and
       disappear into "nothing".  Optical flow just moves pixels around.
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

    def __post_init__(self):  # noqa
        assert self.output_image_size_pixels <= self.input_image_size_pixels, (
            "output_image_size_pixels must be equal to or smaller than input_image_size_pixels"
            f" {self.output_image_size_pixels=}, {self.input_image_size_pixels=}"
        )

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
        """Open the underlying self.source_data_source."""
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
        satellite_data_cropped = satellite_data.isel(time=0, channels=0)
        satellite_data_cropped = crop_center(satellite_data_cropped, self.output_image_size_pixels)

        # Put into DataArray:
        return xr.DataArray(
            data=predictions,
            coords=(
                ("time", datetime_index_of_predictions),
                ("x_osgb", satellite_data_cropped.coords["x_osgb"].values),
                ("y_osgb", satellite_data_cropped.coords["y_osgb"].values),
                ("channels", satellite_data.coords["channels"].values),
            ),
            name="data",
        )

    def _compute_and_return_optical_flow(self, satellite_data: xr.DataArray) -> xr.DataArray:
        """
        Compute and return optical flow predictions for the example

        Args:
            satellite_data: Satellite DataArray of historical satellite images, up to and include t0

        Returns:
            DataArray with the optical flow predictions from t1 to the forecast horizon.
        """
        n_channels = satellite_data.sizes["channels"]

        # Sanity check
        assert (
            len(satellite_data.coords["time"]) == self.history_length + 1
        ), f"{len(satellite_data.coords['time'])=} != {self.history_length+1=}"
        assert n_channels == len(self.channels), f"{n_channels=} != {len(self.channels)=}"

        # Pre-allocate an array, into which our optical flow prediction will be placed.
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

        # Compute flow fields and optical flow predictions separately for each satellite channel
        # because the different channels represent different physical phenomena and so,
        # in principle, could move in different directions (e.g. water vapour vs high clouds).
        for channel_i in range(n_channels):
            # Compute optical flow field:
            sat_data_for_chan = satellite_data.isel(channels=channel_i)

            # Loop through pairs of historical images to compute optical flow fields for each
            # pair of consecutive satellite images, and then compute the mean of those flow fields.
            optical_flows = []
            # self.history_length does not include t0.
            for history_timestep in range(self.history_length):
                prev_image = sat_data_for_chan.isel(time=history_timestep).data
                next_image = sat_data_for_chan.isel(time=history_timestep + 1).data
                optical_flow = compute_optical_flow(prev_image, next_image)
                optical_flows.append(optical_flow)
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

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.  If not, raise a FileNotFoundError."""
        nd_fs_utils.check_path_exists(self.zarr_path)


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
    assert prev_image.dtype == next_image.dtype, "Images must be the same dtype!"

    # cv2.calcOpticalFlowFarneback expects images to be uint8:
    prev_image, next_image = convert_arrays_to_uint8(prev_image, next_image)

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
