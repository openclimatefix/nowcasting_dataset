"""  General Data Source Class """
import itertools
import logging
from concurrent import futures
from dataclasses import InitVar, dataclass
from numbers import Number
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import pandas as pd
import xarray as xr

import nowcasting_dataset.filesystem.utils as nd_fs_utils
import nowcasting_dataset.time as nd_time
import nowcasting_dataset.utils as nd_utils
from nowcasting_dataset import square
from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_COLUMN_NAMES
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.dataset.xr_utils import join_list_dataset_to_batch_dataset, make_dim_index

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
        """Post Init"""
        self.check_input_paths_exist()
        self.sample_period_duration = pd.Timedelta(self.sample_period_minutes, unit="minutes")

        # TODO: Do we still need all these different representations of sequence lengths?
        # See GitHub issue #219 for more details, and to track this TODO task.
        self.history_length = self.history_minutes // self.sample_period_minutes
        self.forecast_length = self.forecast_minutes // self.sample_period_minutes

        assert self.history_length >= 0
        assert self.forecast_length >= 0
        assert self.history_minutes % self.sample_period_minutes == 0, (
            f"sample period ({self.sample_period_minutes}) minutes "
            f"does not fit into historic minutes ({self.history_minutes})"
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

    def _get_start_dt(
        self, t0_dt: Union[pd.Timestamp, pd.DatetimeIndex]
    ) -> Union[pd.Timestamp, pd.DatetimeIndex]:
        return t0_dt - self._history_duration

    def _get_end_dt(
        self, t0_dt: Union[pd.Timestamp, pd.DatetimeIndex]
    ) -> Union[pd.Timestamp, pd.DatetimeIndex]:
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

        This functions may be overwritten if the sample period of the data source is not 5 minutes.
        """
        logging.debug(
            "Getting sample_period_minutes default of 5 minutes. "
            "This means the data is spaced 5 minutes apart"
        )
        return 5

    def open(self):
        """Open the data source, if necessary.

        Called from each worker process.  Useful for data sources where the
        underlying data source cannot be forked (like Zarr).

        Data sources which can be forked safely should call open() from __init__().
        """
        pass

    def check_input_paths_exist(self) -> None:
        """Check any input paths exist.  Raise FileNotFoundError if not.

        Can be overridden by child classes.
        """
        pass

    # TODO: Issue #319: Standardise parameter names.
    def create_batches(
        self,
        spatial_and_temporal_locations_of_each_example: pd.DataFrame,
        idx_of_first_batch: int,
        batch_size: int,
        dst_path: Path,
        local_temp_path: Path,
        upload_every_n_batches: int,
    ) -> None:
        """Create multiple batches and save them to disk.

        Safe to call from worker processes.

        Args:
          spatial_and_temporal_locations_of_each_example: A DataFrame where each row specifies
            the spatial and temporal location of an example.  The number of rows must be
            an exact multiple of `batch_size`.
            Columns are: t0_datetime_UTC, x_center_OSGB, y_center_OSGB.
          idx_of_first_batch: The batch number of the first batch to create.
          batch_size: The number of examples per batch.
          dst_path: The final destination path for the batches.  Must exist.
          local_temp_path: The local temporary path.  This is only required when dst_path is a
            cloud storage bucket, so files must first be created on the VM's local disk in temp_path
            and then uploaded to dst_path every upload_every_n_batches. Must exist. Will be emptied.
          upload_every_n_batches: Upload the contents of temp_path to dst_path after this number
            of batches have been created.  If 0 then will write directly to dst_path.
        """
        # Sanity checks:
        assert idx_of_first_batch >= 0
        assert batch_size > 0
        assert len(spatial_and_temporal_locations_of_each_example) % batch_size == 0
        assert upload_every_n_batches >= 0
        assert spatial_and_temporal_locations_of_each_example.columns.to_list() == list(
            SPATIAL_AND_TEMPORAL_LOCATIONS_COLUMN_NAMES
        )

        self.open()

        # Figure out where to write batches to:
        save_batches_locally_and_upload = upload_every_n_batches > 0
        if save_batches_locally_and_upload:
            nd_fs_utils.delete_all_files_in_temp_path(local_temp_path)
        path_to_write_to = local_temp_path if save_batches_locally_and_upload else dst_path

        # Split locations per example into batches:
        n_batches = len(spatial_and_temporal_locations_of_each_example) // batch_size
        locations_for_batches = []
        for batch_idx in range(n_batches):
            start_example_idx = batch_idx * batch_size
            end_example_idx = (batch_idx + 1) * batch_size
            locations_for_batch = spatial_and_temporal_locations_of_each_example.iloc[
                start_example_idx:end_example_idx
            ]
            locations_for_batches.append(locations_for_batch)

        # Loop round each batch:
        for n_batches_processed, locations_for_batch in enumerate(locations_for_batches):
            batch_idx = idx_of_first_batch + n_batches_processed
            logger.debug(f"{self.__class__.__name__} creating batch {batch_idx}!")

            # Generate batch.
            batch = self.get_batch(
                t0_datetimes=locations_for_batch.t0_datetime_UTC,
                x_locations=locations_for_batch.x_center_OSGB,
                y_locations=locations_for_batch.y_center_OSGB,
            )

            # Save batch to disk.
            netcdf_filename = path_to_write_to / nd_utils.get_netcdf_filename(batch_idx)
            batch.to_netcdf(netcdf_filename)

            # Upload if necessary.
            if (
                save_batches_locally_and_upload
                and n_batches_processed > 0
                and n_batches_processed % upload_every_n_batches == 0
            ):
                nd_fs_utils.upload_and_delete_local_files(dst_path, path_to_write_to)

        # Upload last few batches, if necessary:
        if save_batches_locally_and_upload:
            nd_fs_utils.upload_and_delete_local_files(dst_path, path_to_write_to)

    # TODO: Issue #319: Standardise parameter names.
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
                `future_minutes`.  The batch size is given by the length of the t0_datetimes.
            x_locations: x center batch locations
            y_locations: y center batch locations

        Returns: Batch data.
        """
        assert len(t0_datetimes) == len(
            x_locations
        ), f"len(t0_datetimes) != len(x_locations): {len(t0_datetimes)} != {len(x_locations)}"
        assert len(t0_datetimes) == len(
            y_locations
        ), f"len(t0_datetimes) != len(y_locations): {len(t0_datetimes)} != {len(y_locations)}"
        zipped = list(zip(t0_datetimes, x_locations, y_locations))
        batch_size = len(t0_datetimes)

        with futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_examples = []
            for coords in zipped:
                t0_datetime, x_location, y_location = coords
                future_example = executor.submit(
                    self.get_example, t0_datetime, x_location, y_location
                )
                future_examples.append(future_example)
            examples = [future_example.result() for future_example in future_examples]

        # Get the DataSource class, this could be one of the data sources like Sun
        cls = examples[0].__class__

        # Set the coords to be indices before joining into a batch
        examples = [make_dim_index(example) for example in examples]

        # join the examples together, and cast them to the cls, so that validation can occur
        return cls(join_list_dataset_to_batch_dataset(examples))

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

    # TODO: Issue #319: Standardise parameter names.
    def get_locations(self, t0_datetimes: pd.DatetimeIndex) -> Tuple[List[Number], List[Number]]:
        """Find a valid geographical locations for each t0_datetime.

        Should be overridden by DataSources which may be used to define the locations.

        Returns:  x_locations, y_locations. Each has one entry per t0_datetime.
            Locations are in OSGB coordinates.
        """
        raise NotImplementedError()

    # ****************** METHODS THAT MUST BE OVERRIDDEN **********************
    # TODO: Issue #319: Standardise parameter names.
    def _get_time_slice(self, t0_dt: pd.Timestamp):
        """Get a single timestep of data.  Must be overridden."""
        raise NotImplementedError()

    # TODO: Issue #319: Standardise parameter names.
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
        """Post Init"""
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

    # zarr_path and channels must be set.  But dataclasses complains about defining a non-default
    # argument after a default argument if we remove the ` = None`.
    zarr_path: Union[Path, str] = None
    channels: Iterable[str] = None
    consolidated: bool = True

    def __post_init__(self, image_size_pixels: int, meters_per_pixel: int):
        """Post init"""
        super().__post_init__(image_size_pixels, meters_per_pixel)
        self._data = None

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.  If not, raise a FileNotFoundError."""
        nd_fs_utils.check_path_exists(self.zarr_path)

    @property
    def data(self):
        """Data property"""
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

        return selected_data.load()

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
