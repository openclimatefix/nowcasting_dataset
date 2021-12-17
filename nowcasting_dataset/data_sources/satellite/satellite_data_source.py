""" Satellite Data Source """
import logging
from dataclasses import InitVar, dataclass
from functools import partial
from numbers import Number
from typing import Iterable, Optional

import dask
import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.time as nd_time
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite

_LOG = logging.getLogger(__name__)
_LOG_HRV = logging.getLogger(__name__.replace("satellite", "hrvsatellite"))


@dataclass
class SatelliteDataSource(ZarrDataSource):
    """Satellite Data Source."""

    channels: Optional[Iterable[str]] = SAT_VARIABLE_NAMES[1:]
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000
    logger = _LOG

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """Post Init"""
        assert len(self.channels) > 0, "channels cannot be empty!"
        assert image_size_pixels > 0, "image_size_pixels cannot be <= 0!"
        assert meters_per_pixel > 0, "meters_per_pixel cannot be <= 0!"
        super().__post_init__(image_size_pixels, meters_per_pixel)
        n_channels = len(self.channels)
        self._shape_of_example = (
            self.total_seq_length,
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
        if "variable" in self._data.dims:
            self._data = self._data.rename({"variable": "channels"})
        if not set(self.channels).issubset(self._data.channels.values):
            raise RuntimeError(
                f"One or more requested channels are not available in {self.zarr_path}!"
                f"  Requested channels={self.channels}."
                f"  Available channels={self._data.channels.values}"
            )
        self._data = self._data.sel(channels=list(self.channels))

    def _open_data(self) -> xr.DataArray:
        return open_sat_data(
            zarr_path=self.zarr_path, consolidated=self.consolidated, logger=self.logger
        )

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return Satellite

    def _get_time_slice(self, t0_dt: pd.Timestamp) -> xr.DataArray:
        start_dt = self._get_start_dt(t0_dt)
        end_dt = self._get_end_dt(t0_dt)
        data = self.data.sel(time=slice(start_dt, end_dt))
        assert type(data) == xr.DataArray

        return data

    def get_spatial_region_of_interest(
        self, data_array: xr.DataArray, x_center_osgb: Number, y_center_osgb: Number
    ) -> xr.DataArray:
        """
        Gets the satellite image as a square around the center

        Ignores x and y coordinates as for the original satellite projection each pixel varies in
        both its x and y distance from other pixels. See Issue 401 for more details.

        This results, in 'real' spatial terms, each image covering about 2x as much distance in the
        x direction as in the y direction.

        Args:
            data_array: DataArray to subselect from
            x_center_osgb: Center of the image x coordinate in OSGB coordinates
            y_center_osgb: Center of image y coordinate in OSGB coordinates

        Returns:
            The selected data around the center
        """
        # Get the index into x and y nearest to x_center_osgb and y_center_osgb:
        x_index_at_center = np.searchsorted(data_array.x.values, x_center_osgb) - 1
        y_index_at_center = np.searchsorted(data_array.y.values, y_center_osgb) - 1
        # Put x_index_at_center and y_index_at_center into a pd.Series so we can operate
        # on them both in a single line of code.
        x_and_y_index_at_center = pd.Series({"x": x_index_at_center, "y": y_index_at_center})
        half_image_size_pixels = self._square.size_pixels // 2
        min_x_and_y_index = x_and_y_index_at_center - half_image_size_pixels
        max_x_and_y_index = x_and_y_index_at_center + half_image_size_pixels

        # Check whether the requested region of interest steps outside of the available data:
        suggested_reduction_of_image_size_pixels = (
            max(
                (-min_x_and_y_index.min() if (min_x_and_y_index < 0).any() else 0),
                (max_x_and_y_index.x - len(data_array.x)),
                (max_x_and_y_index.y - len(data_array.y)),
            )
            * 2
        )
        # If the requested region does step outside the available data then raise an exception
        # with a helpful message:
        if suggested_reduction_of_image_size_pixels > 0:
            new_suggested_image_size_pixels = (
                self._square.size_pixels - suggested_reduction_of_image_size_pixels
            )
            raise RuntimeError(
                "Requested region of interest of satellite data steps outside of the available"
                " geographical extent of the Zarr data.  The requested region of interest extends"
                f" from pixel indicies"
                f" x={min_x_and_y_index.x} to x={max_x_and_y_index.x},"
                f" y={min_x_and_y_index.y} to y={max_x_and_y_index.y}.  In the Zarr data,"
                f" len(x)={len(data_array.x)}, len(y)={len(data_array.y)}. Try reducing"
                f" image_size_pixels from {self._square.size_pixels} to"
                f" {new_suggested_image_size_pixels} pixels."
            )

        # Select the geographical region of interest.
        # Note that isel is *exclusive* of the end of the slice.
        # e.g. isel(x=slice(0, 3)) will return the first, second, and third values.
        data_array = data_array.isel(
            x=slice(min_x_and_y_index.x, max_x_and_y_index.x),
            y=slice(min_x_and_y_index.y, max_x_and_y_index.y),
        )
        return data_array

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> xr.Dataset:
        """
        Get Example data

        Args:
            t0_dt: list of timestamps for the datetime of the batches. The batch will also include
                data for historic and future depending on `history_minutes` and `future_minutes`.
            x_meters_center: x center batch locations
            y_meters_center: y center batch locations

        Returns: Example Data

        """
        selected_data = self._get_time_slice(t0_dt)
        selected_data = self.get_spatial_region_of_interest(
            data_array=selected_data,
            x_center_osgb=x_meters_center,
            y_center_osgb=y_meters_center,
        )

        if "variable" in list(selected_data.dims):
            selected_data = selected_data.rename({"variable": "channels"})

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

        return selected_data.load().to_dataset(name="data")

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


class HRVSatelliteDataSource(SatelliteDataSource):
    """Satellite Data Source for HRV data."""

    channels: Optional[Iterable[str]] = SAT_VARIABLE_NAMES[:1]
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000
    logger = _LOG_HRV


def remove_acq_time_from_dataset_and_fix_time_coords(
    dataset: xr.Dataset, logger: logging.Logger
) -> xr.Dataset:
    """
    Preprocess datasets by dropping `acq_time`, which causes problems otherwise

    Args:
        dataset: xr.Dataset to preprocess
        logger: logger object to write to

    Returns:
        dataset with acq_time dropped
    """
    dataset = dataset.drop_vars("acq_time", errors="ignore")

    # If there are any duplicated init_times then drop the duplicated init_times:
    data_array = dataset["stacked_eumetsat_data"]
    times = pd.DatetimeIndex(data_array["time"])
    if not times.is_unique:
        n_duplicates = times.duplicated().sum()
        logger.warning(f"Satellite Zarr has {n_duplicates:,d} duplicated times.  Fixing...")
        data_array = data_array.drop_duplicates(dim="time")
        times = pd.DatetimeIndex(data_array["time"])
        dataset = data_array.to_dataset(name="stacked_eumetsat_data")

    # If any init_times are not monotonic_increasing then drop the out-of-order init_times:
    if not times.is_monotonic_increasing:
        total_n_out_of_order_times = 0
        logger.warning("Satellite Zarr times is not monotonic_increasing.  Fixing...")
        while not times.is_monotonic_increasing:
            diff = np.diff(times.view(int))
            out_of_order = np.where(diff < 0)[0]
            total_n_out_of_order_times += len(out_of_order)
            out_of_order = times[out_of_order]
            data_array = data_array.drop_sel(time=out_of_order)
            times = pd.DatetimeIndex(data_array["init_time"])
        logger.info(f"Fixed {total_n_out_of_order_times:,d} out of order init_times.")
        dataset = data_array.to_dataset(name="stacked_eumetsat_data")

    return dataset


def open_sat_data(zarr_path: str, consolidated: bool, logger: logging.Logger) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Adds 1 minute to the 'time' coordinates, so the timestamps
    are at 00, 05, ..., 55 past the hour.

    Args:
      zarr_path: Cloud URL or local path pattern.  If GCP URL, must start with 'gs://'
      consolidated: Whether or not the Zarr metadata is consolidated.
      logger: logger object to write to
    """
    logger.debug("Opening satellite data: %s", zarr_path)

    # If we are opening multiple Zarr stores (i.e. one for each month of the year) we load them
    # together and create a single dataset from them.  open_mfdataset also works if zarr_path
    # points to a specific zarr directory (with no wildcards).

    # Silence the warning about large chunks.
    # Alternatively, we could set this to True, but that slows down loading a Satellite batch
    # from 8 seconds to 50 seconds!
    dask.config.set(**{"array.slicing.split_large_chunks": False})

    # add logger to preprocess function
    p_remove_acq_time_from_dataset_and_fix_time_coords = partial(
        remove_acq_time_from_dataset_and_fix_time_coords, logger=logger
    )

    # Open datasets.
    dataset = xr.open_mfdataset(
        zarr_path,
        chunks="auto",  # See issue #456 for why we use "auto".
        mode="r",
        engine="zarr",
        concat_dim="time",
        preprocess=p_remove_acq_time_from_dataset_and_fix_time_coords,
        consolidated=consolidated,
        combine="nested",
    )

    data_array = dataset["stacked_eumetsat_data"]
    if "stacked_eumetsat_data" == data_array.name:
        data_array.name = "data"
    del dataset

    # Flip coordinates to top-left first
    data_array = data_array.reindex(x=data_array.x[::-1])

    # Sanity check!
    times = pd.DatetimeIndex(data_array["time"])
    assert times.is_unique
    assert times.is_monotonic_increasing

    return data_array
