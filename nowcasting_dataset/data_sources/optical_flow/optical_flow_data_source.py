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
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.data_sources.optical_flow.optical_flow_model import OpticalFlow
from nowcasting_dataset.dataset.xr_utils import join_list_data_array_to_batch_dataset

_LOG = logging.getLogger("nowcasting_dataset")


@dataclass
class OpticalFlowDataSource(ZarrDataSource):
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
        super().__post_init__(image_size_pixels+32, meters_per_pixel)
        n_channels = len(self.channels)
        self._cache = {}
        self._shape_of_example = (
            self._total_seq_length,
            image_size_pixels,
            image_size_pixels,
            n_channels,
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

        # Compute optical flow for the timesteps
        # Get Optical Flow for the pre-t0 time, and applying the t0-previous_timesteps_per_flow to
        # t0 optical flow for forecast steps in the future
        # Creates a pyramid of optical flows for all timesteps up to t0, and apply predictions
        # for all future timesteps for each of them
        # Compute optical flow per channel, as it might be different
        selected_data = self._compute_and_return_optical_flow(selected_data, t0_dt = t0_dt)

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

        return selected_data

    def _compute_previous_timestep(self, satellite_data: xr.DataArray, t0_dt: pd.Timestamp) -> pd.Timestamp:
        """
        Get timestamp of previous

        Args:
            satellite_data:
            t0_dt:

        Returns:

        """
        satellite_data = satellite_data.where(satellite_data.time < t0_dt, drop = True)
        return satellite_data.isel(time=-self.previous_timestep_for_flow).values

    def _get_number_future_timesteps(self, satellite_data: xr.DataArray, t0_dt: pd.Timestamp) -> \
            int:
        """
        Get number of future timestamps

        Args:
            satellite_data:
            t0_dt:

        Returns:

        """
        satellite_data = satellite_data.where(satellite_data.time > t0_dt, drop = True)
        return len(satellite_data.coords['time'])

    def _compute_and_return_optical_flow(self, satellite_data: xr.DataArray, t0_dt: pd.Timestamp):
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
        previous_timestamp = self._compute_previous_timestep(satellite_data, t0_dt = t0_dt)
        for channel in satellite_data.coords["channels"]:
            channel_images = satellite_data.sel(channel=channel)
            t0_image = channel_images.sel(time=t0_dt).values
            previous_image = channel_images.sel(time=previous_timestamp).values
            optical_flow = self._compute_optical_flow(t0_image, previous_image)
            # Do predictions now
            predictions = []
            # Number of timesteps before t0
            for prediction_timestep in range(self._get_number_future_timesteps(satellite_data, t0_dt)):
                flow = optical_flow * prediction_timestep
                warped_image = self._remap_image(t0_image, flow)
                warped_image = crop_center(warped_image, self._square.size_pixels,
                                           self._square.size_pixels)
                predictions.append(warped_image)
            prediction_dictionary[channel] = predictions
        # TODO Convert to xr.DataArray
        return prediction_dictionary


    def _compute_optical_flow(self, t0_image: np.ndarray, previous_image: np.ndarray) -> np.ndarray:
        """
        Args:
            satellite_data: uint8 numpy array of shape (num_timesteps, height, width)

        Returns:
            optical flow field
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
        """Takes an image and warps it forwards in time according to the flow field.

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
        # cv.remap docs: https://docs.opencv.org/4.5.0/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
        return cv2.remap(
            src=image,
            map1=remap,
            map2=None,
            interpolation=cv2.INTER_LINEAR,
            # See BorderTypes: https://docs.opencv.org/4.5.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.NaN,
        )

    def _open_data(self) -> xr.DataArray:
        return open_sat_data(zarr_path=self.zarr_path, consolidated=self.consolidated)

    def _dataset_to_data_source_output(output: xr.Dataset) -> OpticalFlow:
        return OpticalFlow(output)

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        data = self.data.sel(time=slice(start_dt, end_dt))
        return data

    def datetime_index(self, remove_night: bool = True) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes

        Args:
            remove_night: If True then remove datetimes at night.
                We're interested in forecasting solar power generation, so we
                don't care about nighttime data :)

                In the UK in summer, the sun rises first in the north east, and
                sets last in the north west [1].  In summer, the north gets more
                hours of sunshine per day.

                In the UK in winter, the sun rises first in the south east, and
                sets last in the south west [2].  In winter, the south gets more
                hours of sunshine per day.

                |                        | Summer | Winter |
                |           ---:         |  :---: |  :---: |
                | Sun rises first in     | N.E.   | S.E.   |
                | Sun sets last in       | N.W.   | S.W.   |
                | Most hours of sunlight | North  | South  |

                Before training, we select timesteps which have at least some
                sunlight.  We do this by computing the clearsky global horizontal
                irradiance (GHI) for the four corners of the satellite imagery,
                and for all the timesteps in the dataset.  We only use timesteps
                where the maximum global horizontal irradiance across all four
                corners is above some threshold.

                The 'clearsky solar irradiance' is the amount of sunlight we'd
                expect on a clear day at a specific time and location. The SI unit
                of irradiance is watt per square meter.  The 'global horizontal
                irradiance' (GHI) is the total sunlight that would hit a
                horizontal surface on the surface of the Earth.  The GHI is the
                sum of the direct irradiance (sunlight which takes a direct path
                from the Sun to the Earth's surface) and the diffuse horizontal
                irradiance (the sunlight scattered from the atmosphere).  For more
                info, see: https://en.wikipedia.org/wiki/Solar_irradiance

        References:
          1. [Video of June 2019](https://www.youtube.com/watch?v=IOp-tj-IJpk)
          2. [Video of Jan 2019](https://www.youtube.com/watch?v=CJ4prUVa2nQ)
        """
        if self._data is None:
            sat_data = self._open_data()
        else:
            sat_data = self._data

        datetime_index = pd.DatetimeIndex(sat_data.time.values)

        if remove_night:
            border_locations = self.geospatial_border()
            datetime_index = nd_time.select_daylight_datetimes(
                datetimes=datetime_index, locations=border_locations
                )

        return datetime_index


def open_sat_data(zarr_path: str, consolidated: bool) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Adds 1 minute to the 'time' coordinates, so the timestamps
    are at 00, 05, ..., 55 past the hour.

    Args:
      zarr_path: Cloud URL or local path.  If GCP URL, must start with 'gs://'
      consolidated: Whether or not the Zarr metadata is consolidated.
    """
    _LOG.debug("Opening satellite data: %s", zarr_path)

    # We load using chunks=None so xarray *doesn't* use Dask to
    # load the Zarr chunks from disk.  Using Dask to load the data
    # seems to slow things down a lot if the Zarr store has more than
    # about a million chunks.
    # See https://github.com/openclimatefix/nowcasting_dataset/issues/23
    dataset = xr.open_dataset(
        zarr_path, engine="zarr", consolidated=consolidated, mode="r", chunks=None
    )

    data_array = dataset["stacked_eumetsat_data"]
    del dataset

    # The 'time' dimension is at 04, 09, ..., 59 minutes past the hour.
    # To make it easier to align the satellite data with other data sources
    # (which are at 00, 05, ..., 55 minutes past the hour) we add 1 minute to
    # the time dimension.
    # TODO Remove this as new Zarr already has the time fixed
    data_array["time"] = data_array.time + pd.Timedelta("1 minute")
    return data_array


def crop_center(img,cropx,cropy):
    """
    Crop center of numpy image

    Args:
        img:
        cropx:
        cropy:

    Returns:

    """
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]