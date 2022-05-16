""" Satellite Data Source """
import itertools
import logging
from dataclasses import InitVar, dataclass
from datetime import timedelta
from functools import partial
from numbers import Number
from typing import Callable, Iterable, Optional

import dask
import numpy as np
import pandas as pd
import pyproj
import pyresample
import xarray as xr

import nowcasting_dataset.time as nd_time
from nowcasting_dataset.consts import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.data_source import ZarrDataSource
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.data_sources.satellite.satellite_model import Satellite
from nowcasting_dataset.geospatial import OSGB
from nowcasting_dataset.utils import drop_duplicate_times, drop_non_monotonic_increasing, is_sorted

_LOG = logging.getLogger(__name__)
_LOG_HRV = logging.getLogger(__name__.replace("satellite", "hrvsatellite"))


@dataclass
class SatelliteDataSource(ZarrDataSource):
    """Satellite Data Source."""

    channels: Optional[Iterable[str]] = SAT_VARIABLE_NAMES[1:]
    image_size_pixels_height: InitVar[int] = 128
    image_size_pixels_width: InitVar[int] = 256
    meters_per_pixel: InitVar[int] = 2_000
    logger = _LOG
    time_resolution_minutes: int = 5
    keep_dawn_dusk_hours: int = 0
    is_live: bool = False
    live_delay_minutes: int = 30

    def __post_init__(
        self, image_size_pixels_height: int, image_size_pixels_width: int, meters_per_pixel: int
    ):
        """Post Init"""
        assert len(self.channels) > 0, "channels cannot be empty!"
        assert image_size_pixels_height > 0, "image_size_pixels_height cannot be <= 0!"
        assert image_size_pixels_width > 0, "image_size_pixels_width cannot be <= 0!"
        assert meters_per_pixel > 0, "meters_per_pixel cannot be <= 0!"
        super().__post_init__(image_size_pixels_height, image_size_pixels_width, meters_per_pixel)
        n_channels = len(self.channels)

        if self.is_live:
            # This is to account for the delay in satellite data
            self.total_seq_length = (
                self.history_length
                - int(self.live_delay_minutes / self.time_resolution_minutes)
                + 1
            )

        self._shape_of_example = (
            self.total_seq_length,
            image_size_pixels_height,
            image_size_pixels_width,
            n_channels,
        )

        self._osgb_to_geostationary: Callable = None

    @property
    def sample_period_minutes(self) -> int:
        """Override the default sample minutes"""
        return self.time_resolution_minutes

    def open(self) -> None:
        """
        Open Satellite data

        We don't want to open_sat_data in __init__.
        If we did that, then we couldn't copy SatelliteDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        self._data = self._open_data()
        if not set(self.channels).issubset(self._data.channels.values):
            raise RuntimeError(
                f"One or more requested channels are not available in {self.zarr_path}!"
                f"  Requested channels={self.channels}."
                f"  Available channels={self._data.channels.values}"
            )
        self._data = self._data.sel(channels=list(self.channels))
        self._load_geostationary_area_definition_and_transform()

        # Check the x and y coords are ascending. If they are not then searchsorted won't work!
        assert is_sorted(self._data.x_geostationary)
        assert is_sorted(self._data.y_geostationary)

    def _load_geostationary_area_definition_and_transform(self) -> None:
        area_definition_yaml = self._data.attrs["area"]
        geostationary_area_definition = pyresample.area_config.load_area_from_string(
            area_definition_yaml
        )
        geostationary_crs = geostationary_area_definition.crs
        self._osgb_to_geostationary = pyproj.Transformer.from_crs(
            crs_from=OSGB, crs_to=geostationary_crs
        ).transform

    def _open_data(self) -> xr.DataArray:
        data_array = open_sat_data(
            zarr_path=self.zarr_path,
            consolidated=self.consolidated,
            logger=self.logger,
            sample_period_minutes=self.sample_period_minutes,
        )
        # If live, then load into memory so the data
        # doesn't disappear while its being used
        if self.is_live:
            data_array = data_array.load()
        return data_array

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
        return Satellite

    def _get_time_slice(self, t0_datetime_utc: pd.Timestamp) -> xr.DataArray:
        start_dt = self._get_start_dt(t0_datetime_utc)
        end_dt = self._get_end_dt(t0_datetime_utc)

        # if live data, take 30 ins from the end time,
        # so that we account for delay in data from satellite
        if self.is_live:
            end_dt = end_dt - timedelta(minutes=self.live_delay_minutes)

        # floor to 15 mins
        start_floor = start_dt.floor(f"{self.sample_period_minutes}T")
        end_floor = end_dt.floor(f"{self.sample_period_minutes}T")

        data = self.data.sel(time=slice(start_floor, end_floor))
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
        x_center_geostationary, y_center_geostationary = self._osgb_to_geostationary(
            x_center_osgb, y_center_osgb
        )
        # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
        x_index_at_center = (
            np.searchsorted(data_array.x_geostationary.values, x_center_geostationary) - 1
        )
        y_index_at_center = (
            np.searchsorted(data_array.y_geostationary.values, y_center_geostationary) - 1
        )
        # Put x_index_at_center and y_index_at_center into a pd.Series so we can operate
        # on them both in a single line of code.
        x_and_y_index_at_center = pd.Series(
            {"x_index_at_center": x_index_at_center, "y_index_at_center": y_index_at_center}
        )
        half_image_size_pixels_height = self._rectangle.size_pixels_height // 2
        half_image_size_pixels_width = self._rectangle.size_pixels_width // 2
        min_x_and_y_index_width = x_and_y_index_at_center - half_image_size_pixels_width
        max_x_and_y_index_width = x_and_y_index_at_center + half_image_size_pixels_width
        min_x_and_y_index_height = x_and_y_index_at_center - half_image_size_pixels_height
        max_x_and_y_index_height = x_and_y_index_at_center + half_image_size_pixels_height

        # Check whether the requested region of interest steps outside of the available data:
        # Need to know how much to pad the outputs, so can do that here
        left_width_padding = max(-min_x_and_y_index_width.x_index_at_center, 0)
        right_width_padding = max(
            max_x_and_y_index_width.x_index_at_center - len(data_array.x_geostationary), 0
        )
        bottom_height_padding = max(-min_x_and_y_index_height.y_index_at_center, 0)
        top_height_padding = max(
            max_x_and_y_index_height.y_index_at_center - len(data_array.y_geostationary), 0
        )

        # Select the geographical region of interest.
        # Note that isel is *exclusive* of the end of the slice.
        # e.g. isel(x=slice(0, 3)) will return the first, second, and third values.
        data_array = data_array.isel(
            x_geostationary=slice(
                max(min_x_and_y_index_width.x_index_at_center, 0),
                min(max_x_and_y_index_width.x_index_at_center, len(data_array.x_geostationary)),
            ),
            y_geostationary=slice(
                max(min_x_and_y_index_height.y_index_at_center, 0),
                min(max_x_and_y_index_height.y_index_at_center, len(data_array.y_geostationary)),
            ),
        )

        # isel is wrong if padding before, so add padding after to be the correct size
        # Get the difference for each direction and use that
        if (
            bottom_height_padding > 0
            or left_width_padding > 0
            or top_height_padding > 0
            or right_width_padding > 0
        ):
            data_array = data_array.pad(
                pad_width={
                    "x_geostationary": (left_width_padding, right_width_padding),
                    "y_geostationary": (bottom_height_padding, top_height_padding),
                },
                mode="constant",
                constant_values=0,
            )

        return data_array

    def get_example(self, location: SpaceTimeLocation) -> xr.Dataset:
        """
        Get Example data

        Args:
            location: A location object of the example which contains
                - a timestamp of the example (t0_datetime_utc),
                - the x center location of the example (x_location_osgb)
                - the y center location of the example(y_location_osgb)

        Returns: Example Data

        """
        t0_datetime_utc = location.t0_datetime_utc
        x_center_osgb = location.x_center_osgb
        y_center_osgb = location.y_center_osgb

        selected_data = self._get_time_slice(t0_datetime_utc)
        selected_data = self.get_spatial_region_of_interest(
            data_array=selected_data,
            x_center_osgb=x_center_osgb,
            y_center_osgb=y_center_osgb,
        )

        selected_data = self._post_process_example(selected_data, t0_datetime_utc)

        if selected_data.shape != self._shape_of_example:
            raise RuntimeError(
                "Example is wrong shape! "
                f"x_center_osgb={x_center_osgb}\n"
                f"y_center_osgb={y_center_osgb}\n"
                f"t0_dt={t0_datetime_utc}\n"
                f"times are {selected_data.time}\n"
                f"expected shape={self._shape_of_example}\n"
                f"actual shape {selected_data.shape}"
                f" {self.forecast_length=}"
                f" {self.history_length=}"
            )

        # Delete the attributes because they fail to save to HDF5.
        # HDF complains with the exception `TypeError: No conversion path for dtype: dtype('<U32')`
        # (If the downstream code wants to use attributes then we can almost certainly find a way
        # to encode the attributes in an HDF5-friendly fashion. The error was
        # See https://docs.h5py.org/en/latest/strings.html#what-about-numpy-s-u-type
        selected_data.attrs = {}

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
                datetimes=datetime_index,
                locations=border_locations,
                keep_dawn_dusk_hours=self.keep_dawn_dusk_hours,
            )

        return datetime_index

    def geospatial_border(self) -> list[tuple[Number, Number]]:
        """
        Get 'corner' coordinates for a rectangle within the boundary of the data.

        Returns List of 2-tuples of the x and y coordinates of each corner,
        in OSGB projection.
        """
        GEO_BORDER: int = 64  #: In same geo projection and units as sat_data.
        data = self._open_data()
        return [
            (
                data.x_osgb.isel(x_geostationary=x, y_geostationary=y).values,
                data.y_osgb.isel(x_geostationary=x, y_geostationary=y).values,
            )
            for x, y in itertools.product([GEO_BORDER, -GEO_BORDER], [GEO_BORDER, -GEO_BORDER])
        ]


class HRVSatelliteDataSource(SatelliteDataSource):
    """Satellite Data Source for HRV data."""

    channels: Optional[Iterable[str]] = SAT_VARIABLE_NAMES[:1]
    image_size_pixels: InitVar[int] = 128
    meters_per_pixel: InitVar[int] = 2_000
    logger = _LOG_HRV


def dedupe_time_coords(dataset: xr.Dataset, logger: logging.Logger) -> xr.Dataset:
    """
    Preprocess datasets by de-duplicating the time coordinates.

    Args:
        dataset: xr.Dataset to preprocess
        logger: logger object to write to

    Returns:
        dataset with time coords de-duped.
    """
    # If there are any duplicated init_times then drop the duplicated time:
    data = drop_duplicate_times(data_array=dataset["data"], class_name="Satellite", time_dim="time")

    # If any init_times are not monotonic_increasing then drop the out-of-order init_times:
    data = drop_non_monotonic_increasing(data_array=data, class_name="Satellite", time_dim="time")
    dataset = data.to_dataset(name="data")

    assert pd.DatetimeIndex(data["time"]).is_unique
    assert pd.DatetimeIndex(data["time"]).is_monotonic_increasing

    return dataset


def open_sat_data(
    zarr_path: str, consolidated: bool, logger: logging.Logger, sample_period_minutes: int = 15
) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Adds 1 minute to the 'time' coordinates, so the timestamps
    are at 00, 05, ..., 55 past the hour.

    Args:
      zarr_path: Cloud URL or local path pattern.  If GCP URL, must start with 'gs://'
      consolidated: Whether or not the Zarr metadata is consolidated.
      logger: logger object to write to
      sample_period_minutes: The sample period minutes that the data should be reduced to.
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
    p_dedupe_time_coords = partial(dedupe_time_coords, logger=logger)
    if str(zarr_path).split(".")[-1] == "zip":
        if "s3://" or "gs://" in str(zarr_path):
            data_path = "zip::" + str(zarr_path)
        else:
            data_path = "zip://" + str(zarr_path)
        dataset = xr.open_dataset(data_path, engine="zarr")

    else:
        # Open datasets.
        dataset = xr.open_mfdataset(
            zarr_path,
            chunks="auto",  # See issue #456 for why we use "auto".
            mode="r",
            engine="zarr",
            concat_dim="time",
            preprocess=p_dedupe_time_coords,
            consolidated=consolidated,
            combine="nested",
        )

    # Rename
    # These renamings will no longer be necessary when the Zarr uses the 'correct' names,
    # see https://github.com/openclimatefix/Satip/issues/66
    if "x" in dataset:
        dataset = dataset.rename({"x": "x_geostationary", "y": "y_geostationary"})
    if "variable" in dataset:
        dataset = dataset.rename({"variable": "channels"})
    elif "channels" not in dataset:
        # This is HRV version 3, which doesn't have a channels dim.  So add one.
        dataset = dataset.expand_dims(dim={"channels": ["HRV"]}, axis=-1)

    data_array = dataset["data"]
    if "stacked_eumetsat_data" == data_array.name:
        data_array.name = "data"
    del dataset

    # Flip coordinates to top-left first
    data_array = data_array.reindex(x_geostationary=data_array.x_geostationary[::-1])

    # reindex satellite to 15 mins data
    datetime_index = pd.DatetimeIndex(data_array["time"])
    if sample_period_minutes != 5:
        time_mask = datetime_index.minute % sample_period_minutes == 0
        data_array = data_array.sel(time=time_mask)

    # Sanity check!
    assert datetime_index.is_unique
    assert datetime_index.is_monotonic_increasing

    return data_array
