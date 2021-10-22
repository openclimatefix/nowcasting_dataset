"""  General Data Source Class """
import itertools
import logging
from dataclasses import dataclass, InitVar
from numbers import Number
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
import xarray as xr

import nowcasting_dataset.time as nd_time
from nowcasting_dataset import square
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.dataset.xr_utils import join_dataset_to_batch_dataset

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Abstract base class.

    Attributes:
      history_minutes: Number of minutes of history to include in each example.
        Does NOT include t0.  That is, if history_minutes = 0 then the example
        will start at t0.
      forecast_minutes: Number of minutes of forecast to include in each example.
        Does NOT include t0.  If forecast_minutes = 0 then the example will end
        at t0.  If both history_minutes and forecast_minutes are 0, then the example
        will consist of a single timestep at t0.
      sample_period_minutes: The time delta between each data point.  Note that this is set
        using the sample_period_minutes property, so it can be overridden by child classes.

    Attributes ending in `_length` are sequence lengths represented as integer numbers of timesteps.
    Attributes ending in `_duration` are sequence durations represented as pd.Timedeltas.
    """

    history_minutes: int
    forecast_minutes: int

    def __post_init__(self):
        """ Post Init """
        self.sample_period_duration = pd.Timedelta(self.sample_period_minutes, unit="minutes")

        # TODO: Do we still need all these different representations of sequence lengths?
        # See GitHub issue #219 for more details, and to track this TODO task.
        self.history_length = self.history_minutes // self.sample_period_minutes
        self.forecast_length = self.forecast_minutes // self.sample_period_minutes

        assert self.history_length >= 0
        assert self.forecast_length >= 0
        assert self.history_minutes % self.sample_period_minutes == 0, (
            f"sample period ({self.sample_period_minutes}) minutes "
            f"does not fit into historic minutes ({self.forecast_minutes})"
        )
        assert self.forecast_minutes % self.sample_period_minutes == 0, (
            f"sample period ({self.sample_period_minutes}) minutes "
            f"does not fit into forecast minutes ({self.forecast_minutes})"
        )

        # Plus 1 because neither history_length nor forecast_length include t0.
        self._total_seq_length = self.history_length + self.forecast_length + 1

        self._history_duration = pd.Timedelta(self.history_minutes, unit="minutes")
        self._forecast_duration = pd.Timedelta(self.forecast_minutes, unit="minutes")
        # Add sample_period_duration because neither history_duration not forecast_duration
        # include t0.
        self._total_seq_duration = (
            self._history_duration + self._forecast_duration + self.sample_period_duration
        )

    def _get_start_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt - self._history_duration

    def _get_end_dt(self, t0_dt: pd.Timestamp) -> pd.Timestamp:
        return t0_dt + self._forecast_duration

    def get_contiguous_t0_time_periods(self) -> pd.DataFrame:
        """Get all time periods which contain valid t0 datetimes.

        `t0` is the datetime of the most recent observation.

        Returns:
          pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

        Raises:
          NotImplementedError if this DataSource has no concept of a datetime index.
        """
        contiguous_time_periods = self.get_contiguous_time_periods()
        contiguous_time_periods["start_dt"] += self._history_duration
        contiguous_time_periods["end_dt"] -= self._forecast_duration
        assert (contiguous_time_periods["start_dt"] <= contiguous_time_periods["end_dt"]).all()
        return contiguous_time_periods

    # ************* METHODS THAT CAN BE OVERRIDDEN ****************************
    @property
    def sample_period_minutes(self) -> int:
        """
        This is the default sample period in minutes.

        This functions may be overwritten if
        the sample period of the data source is not 5 minutes.
        """
        logging.debug(
            "Getting sample_period_minutes default of 5 minutes. "
            "This means the data is spaced 5 minutes apart"
        )
        return 5

    def open(self):
        """Open the data source, if necessary.

        Called from each worker process.  Useful for data sources where the
        underlying data source cannot be forked (like Zarr on GCP!).

        Data sources which can be forked safely should call open()
        from __init__().
        """
        pass

    def get_batch(
        self,
        t0_datetimes: pd.DatetimeIndex,
        x_locations: Iterable[Number],
        y_locations: Iterable[Number],
    ) -> DataSourceOutput:
        """
        Get Batch Data

        Args:
            t0_datetimes: list of timestamps for the datetime of the batches. The batch will also
                include data for historic and future depending on `history_minutes` and
                `future_minutes`.
            x_locations: x center batch locations
            y_locations: y center batch locations

        Returns: Batch data.
        """
        examples = []
        zipped = zip(t0_datetimes, x_locations, y_locations)
        for t0_datetime, x_location, y_location in zipped:
            output: xr.Dataset = self.get_example(t0_datetime, x_location, y_location)

            examples.append(output)

        # could add option here, to save each data source using
        # 1. # DataSourceOutput.to_xr_dataset() to make it a dataset
        # 2. DataSourceOutput.save_netcdf(), save to netcdf

        # get the name of the cls, this could be one of the data sources like Sun
        cls = examples[0].__class__

        # join the examples together, and cast them to the cls, so that validation can occur
        return cls(join_dataset_to_batch_dataset(examples))

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes."""
        # Leave this NotImplemented if this DataSource has no concept
        # of a list of datetimes (e.g. for DatetimeDataSource).
        raise NotImplementedError()

    def get_contiguous_time_periods(self) -> pd.DataFrame:
        """Get all the time periods for which this DataSource has contiguous data.

        Optionally filter out any time periods which don't make sense for this DataSource,
        e.g. remove nighttime.

        Returns:
          pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

        Raises:
          NotImplementedError if this DataSource has no concept of a datetime index.
        """
        datetimes = self.datetime_index()
        return nd_time.get_contiguous_time_periods(
            datetimes=datetimes,
            min_seq_length=self._total_seq_length,
            max_gap_duration=self.sample_period_duration,
        )

    def get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        """Find a valid geographical locations for each t0_datetime.

        Should be overridden by DataSources which may be used to define the locations
        for each batch.

        Returns:  x_locations, y_locations. Each has one entry per t0_datetime.
            Locations are in OSGB coordinates.
        """
        raise NotImplementedError()

    # ****************** METHODS THAT MUST BE OVERRIDDEN **********************
    def _get_time_slice(self, t0_dt: pd.Timestamp):
        """Get a single timestep of data.  Must be overridden."""
        raise NotImplementedError()

    def get_example(
        self,
        t0_dt: pd.Timestamp,  #: Datetime of "now": The most recent obs.
        x_meters_center: Number,  #: Centre, in OSGB coordinates.
        y_meters_center: Number,  #: Centre, in OSGB coordinates.
    ) -> xr.Dataset:
        """Must be overridden by child classes."""
        raise NotImplementedError()


@dataclass
class ImageDataSource(DataSource):
    """
    Image Data source

    Args:
      image_size_pixels: Size of the width and height of the image crop
        returned by get_sample().
    """

    image_size_pixels: InitVar[int]
    meters_per_pixel: InitVar[int]

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """ Post Init """
        super().__post_init__()
        self._square = square.Square(
            size_pixels=image_size_pixels, meters_per_pixel=meters_per_pixel
        )


@dataclass
class ZarrDataSource(ImageDataSource):
    """
    A General Zarr Data source

    Attributes:
      _data: xr.DataArray data, opened by open().
        x is left-to-right.
        y is top-to-bottom.
        Access using public data property.
      consolidated: Whether or not the Zarr store is consolidated.
      channels: The Zarr parameters to load.
    """

    channels: Iterable[str]
    #: Mustn't be None, but cannot have a non-default arg in this position :)
    n_timesteps_per_batch: int = None
    consolidated: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """ Post init """
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._data = None
        if self.n_timesteps_per_batch is None:
            raise ValueError("n_timesteps_per_batch must be set!")

    @property
    def data(self):
        """ Data property """
        if self._data is None:
            raise RuntimeError("Please run `open()` before accessing data!")
        return self._data

    def get_example(
        self, t0_dt: pd.Timestamp, x_meters_center: Number, y_meters_center: Number
    ) -> DataSourceOutput:
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

        return selected_data

    def geospatial_border(self) -> List[Tuple[Number, Number]]:
        """
        Get 'corner' coordinates for a rectangle within the boundary of the data.

        Returns List of 2-tuples of the x and y coordinates of each corner,
        in OSGB projection.
        """
        GEO_BORDER: int = 64  #: In same geo projection and units as sat_data.
        data = self._open_data()
        return [
            (data.x.values[x], data.y.values[y])
            for x, y in itertools.product([GEO_BORDER, -GEO_BORDER], [GEO_BORDER, -GEO_BORDER])
        ]

    # ****************** METHODS THAT CAN BE OVERRIDDEN **********************
    def _post_process_example(
        self, selected_data: xr.DataArray, t0_dt: pd.Timestamp
    ) -> xr.DataArray:
        return selected_data

    # ****************** METHODS THAT MUST BE OVERRIDDEN **********************
    # (in addition to the DataSource methods that must be overridden)
    def open(self) -> None:
        """
        Open the data

        We don't want to _open_data() in __init__.
        If we did that, then we couldn't copy ZarrDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        raise NotImplementedError()

    def _open_data(self) -> xr.DataArray:
        raise NotImplementedError()
