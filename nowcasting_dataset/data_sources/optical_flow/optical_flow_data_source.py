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
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000
    previous_timestep_for_flow: int = 1

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """ Post Init """
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
        selected_data = self._compute_and_return_optical_flow(selected_data, t0_dt = t0_dt)

        return selected_data

    def _compute_previous_timestep(self, satellite_data: xr.DataArray, t0_dt: pd.Timestamp) -> pd.Timestamp:
        """
        Get timestamp of previous

        Args:
            satellite_data:
            t0_dt:

        Returns:

        """
        satellite_data = satellite_data.where(satellite_data.time <= t0_dt, drop = True)
        return satellite_data.isel(time=-self.previous_timestep_for_flow).values


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
            # TODO Fix this, number of future steps
            for prediction_timestep in range(9):
                flow = optical_flow * prediction_timestep
                warped_image = self._remap_image(t0_image, flow)
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

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        try:
            return self._cache[t0_dt]
        except KeyError:
            start_dt = self._get_start_dt(t0_dt)
            end_dt = self._get_end_dt(t0_dt)
            data = self.data.sel(time=slice(start_dt, end_dt))
            data = data.load()
            self._cache[t0_dt] = data
            return data

    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:

        selected_data.data = selected_data.data.astype(np.float32)

        return selected_data

    def datetime_index(self, remove_night: bool = True) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes

        Args:
            remove_night: If True then remove datetimes at night.
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
