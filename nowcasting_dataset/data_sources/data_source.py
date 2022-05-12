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
from nowcasting_dataset import square, utils
from nowcasting_dataset.data_sources.datasource_output import DataSourceOutput
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.dataset.xr_utils import (
    convert_coordinates_to_indexes_for_list_datasets,
    join_list_dataset_to_batch_dataset,
)
from nowcasting_dataset.utils import get_start_and_end_example_index

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

    Attributes ending in `_length` are sequence lengths represented
        as integer numbers of timesteps.
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
        self.total_seq_length = self.history_length + self.forecast_length + 1

        self.history_duration = pd.Timedelta(self.history_minutes, unit="minutes")
        self.forecast_duration = pd.Timedelta(self.forecast_minutes, unit="minutes")
        # Add sample_period_duration because neither history_duration not forecast_duration
        # include t0.
        self.total_seq_duration = (
            self.history_duration + self.forecast_duration + self.sample_period_duration
        )

    def _get_start_dt(
        self, t0_datetime_utc: Union[pd.Timestamp, pd.DatetimeIndex]
    ) -> Union[pd.Timestamp, pd.DatetimeIndex]:

        return t0_datetime_utc - self.history_duration

    def _get_end_dt(
        self, t0_datetime_utc: Union[pd.Timestamp, pd.DatetimeIndex]
    ) -> Union[pd.Timestamp, pd.DatetimeIndex]:
        return t0_datetime_utc + self.forecast_duration

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
        contiguous_time_periods["start_dt"] += self.history_duration
        contiguous_time_periods["end_dt"] -= self.forecast_duration
        assert (contiguous_time_periods["start_dt"] <= contiguous_time_periods["end_dt"]).all()
        return contiguous_time_periods

    # ************* METHODS THAT CAN BE OVERRIDDEN ****************************
    @property
    def sample_period_minutes(self) -> int:
        """
        This is the default sample period in minutes.

        This functions may be overwritten if the sample period of the data source is not 5 minutes.
        """
        logger.debug(
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

    # TODO: Issue #319: Standardise parameter names.
    # TODO: Issue #367: Reduce duplication.
    @utils.exception_logger
    def create_batches(
        self,
        spatial_and_temporal_locations_of_each_example: List[SpaceTimeLocation],
        idx_of_first_batch: int,
        batch_size: int,
        dst_path: Path,
        local_temp_path: Path,
        upload_every_n_batches: int,
    ) -> None:
        """Create multiple batches and save them to disk.

        Safe to call from worker processes.

        Args:
            spatial_and_temporal_locations_of_each_example (pd.DataFrame): A DataFrame where each
                row specifies the spatial and temporal location of an example. The number of rows
                must be an exact multiple of `batch_size`.
                Columns are: t0_datetime_UTC, x_center_OSGB, y_center_OSGB.
            idx_of_first_batch (int): The batch number of the first batch to create.
            batch_size (int): The number of examples per batch.
            dst_path (Path): The final destination path for the batches.  Must exist.
            local_temp_path (Path): The local temporary path.  This is only required when dst_path
                is a cloud storage bucket, so files must first be created on the VM's local disk in
                temp_path and then uploaded to dst_path every `upload_every_n_batches`. Must exist.
                Will be emptied.
            upload_every_n_batches (int): Upload the contents of temp_path to dst_path after this
                number of batches have been created.  If 0 then will write directly to `dst_path`.
        """
        # Sanity checks:
        assert (
            idx_of_first_batch >= 0
        ), "The batch number of the first batch to create should be greater than 0"
        assert batch_size > 0, "The batch size should be strictly greater than 0."
        assert len(spatial_and_temporal_locations_of_each_example) % batch_size == 0, (
            f"{len(spatial_and_temporal_locations_of_each_example)=} must be"
            f" exactly divisible by {batch_size=}"
        )
        assert upload_every_n_batches >= 0, "`upload_every_n_batches` must be >= 0"

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
            start_example_idx, end_example_idx = get_start_and_end_example_index(
                batch_idx=batch_idx, batch_size=batch_size
            )

            locations_for_batch = spatial_and_temporal_locations_of_each_example[
                start_example_idx:end_example_idx
            ]
            locations_for_batches.append(locations_for_batch)

        # Loop round each batch:
        for n_batches_processed, locations_for_batch in enumerate(locations_for_batches):
            batch_idx = idx_of_first_batch + n_batches_processed
            logger.debug(f"{self.__class__.__name__} creating batch {batch_idx}!")

            # Generate batch.
            batch = self.get_batch(locations=locations_for_batch)

            # Save batch to disk.
            batch.save_netcdf(
                batch_i=batch_idx, path=path_to_write_to, add_data_source_name_to_path=False
            )

            # Upload if necessary.
            if (
                save_batches_locally_and_upload
                and n_batches_processed > 0
                and n_batches_processed % upload_every_n_batches == 0
            ):
                nd_fs_utils.upload_and_delete_local_files(
                    dst_path=dst_path, local_path=path_to_write_to
                )

        # Upload last few batches, if necessary:

        if save_batches_locally_and_upload:
            nd_fs_utils.upload_and_delete_local_files(
                dst_path=dst_path, local_path=path_to_write_to
            )

    def get_batch(self, locations: List[SpaceTimeLocation]) -> DataSourceOutput:
        """
        Get Batch Data

        Args:
            locations: List of locations object
                A location object contains
                - a timestamp of the example (t0_datetime_utc),
                - the x center location of the example (x_location_osgb)
                - the y center location of the example(y_location_osgb)

        Returns: Batch data.
        """

        batch_size = len(locations)

        with futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_examples = []
            for location in locations:
                future_example = executor.submit(self.get_example, location)
                future_examples.append(future_example)

            # Get the examples back.  Loop round each future so we can log a helpful error.
            # If the worker thread raised an exception then the exception won't "bubble up"
            # until we call future_example.result().
            examples = []
            for example_i, future_example in enumerate(future_examples):
                try:
                    result = future_example.result()
                except Exception:
                    logger.error(f"Exception when processing {example_i=}!")
                    raise
                else:
                    examples.append(result)

        # Get the DataSource class, this could be one of the data sources like Sun
        cls = self.get_data_model_for_batch()

        # Set the coords to be indices before joining into a batch
        examples = convert_coordinates_to_indexes_for_list_datasets(examples)

        # join the examples together, and cast them to the cls, so that validation can occur
        batch_one_datasource = cls(join_list_dataset_to_batch_dataset(examples))

        # lets validate
        cls.validate(batch_one_datasource)

        return batch_one_datasource

    def datetime_index(self) -> pd.DatetimeIndex:
        """Returns a complete list of all available datetimes."""
        # Leave this NotImplemented if this DataSource has no concept
        # of a list of datetimes (e.g. for DatetimeDataSource).
        raise NotImplementedError(f"Datetime not implemented for class {self.__class__}")

    @staticmethod
    def get_data_model_for_batch():
        """Get the model that is used in the batch"""
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
            min_seq_length=self.total_seq_length,
            max_gap_duration=self.sample_period_duration,
        )

    def get_locations(
        self, t0_datetimes_utc: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        """Find a valid geographical locations for each t0_datetime.

        Should be overridden by DataSources which may be used to define the locations.

        Returns:  x_locations, y_locations. Each has one entry per t0_datetime.
            Locations are in OSGB coordinates.
        """
        raise NotImplementedError()

    def get_all_locations(
        self, t0_datetimes_utc: pd.DatetimeIndex
    ) -> Tuple[pd.DatetimeIndex, List[Number], List[Number]]:
        """Find all valid geographical locations for each t0_datetime.

        Should be overridden by DataSources which may be used to define the locations.

        Returns:  all_t0_datetimes_utc, x_locations, y_locations.
            For each single t0_datetime, there are all possible locations.
            Each return has length len(t0_datetimes_utc) * number_of_location
            Locations are in OSGB coordinates.
        """
        raise NotImplementedError()

    def get_number_locations(self) -> int:
        """
        Get the number of locations of a data source

        For GSP, this is the number of GSP
        For PV, this is the number of PV locations
        """
        raise NotImplementedError()

    # ****************** METHODS THAT MUST BE OVERRIDDEN **********************
    def _get_time_slice(self, t0_datetime_utc: pd.Timestamp):
        """Get a single timestep of data.  Must be overridden if get_example is not overridden."""
        raise NotImplementedError()

    def get_example(
        self,
        location: SpaceTimeLocation,  #: Location object of the most recent observation
    ) -> xr.Dataset:
        """Must be overridden by child classes."""
        raise NotImplementedError()

    def check_input_paths_exist(self) -> None:
        """Check any input paths exist.  Raise FileNotFoundError if not.

        Must be overridden by child classes.
        """
        raise NotImplementedError()


@dataclass
class ImageDataSource(DataSource):
    """Abstract base class for image Data source."""

    image_size_pixels_height: InitVar[int]
    image_size_pixels_width: InitVar[int]
    meters_per_pixel: InitVar[int]

    def __post_init__(
        self, image_size_pixels_height: int, image_size_pixels_width: int, meters_per_pixel: int
    ):
        """Post Init"""
        super().__post_init__()
        self._rectangle = square.Rectangle(
            size_pixels_height=image_size_pixels_height,
            size_pixels_width=image_size_pixels_width,
            meters_per_pixel=meters_per_pixel,
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

    def __post_init__(
        self, image_size_pixels_height: int, image_size_pixels_width: int, meters_per_pixel: int
    ):
        """Post init"""
        super().__post_init__(image_size_pixels_height, image_size_pixels_width, meters_per_pixel)
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

        logger.debug(
            f"Getting example for {t0_datetime_utc=},  " f"{x_center_osgb=} and  {y_center_osgb=}"
        )

        selected_data = self._get_time_slice(t0_datetime_utc)
        bounding_box = self._rectangle.bounding_box_centered_on(
            x_center_osgb=x_center_osgb, y_center_osgb=y_center_osgb
        )
        selected_data = selected_data.sel(
            x_osgb=slice(bounding_box.left, bounding_box.right),
            y_osgb=slice(bounding_box.top, bounding_box.bottom),
        )

        # selected_sat_data is likely to have 1 too many pixels in x and y
        # because sel(x=slice(a, b)) is [a, b], not [a, b).  So trim:
        selected_data = selected_data.isel(
            x_osgb=slice(0, self._rectangle.size_pixels_width),
            y_osgb=slice(0, self._rectangle.size_pixels_height),
        )

        selected_data = self._post_process_example(selected_data, t0_datetime_utc)

        if selected_data.shape != self._shape_of_example:
            m = (
                "Example is wrong shape! "
                f"x_center_osgb={x_center_osgb}\n"
                f"y_center_osgb={y_center_osgb}\n"
                f"t0_datetime_utc={t0_datetime_utc}\n"
                f"times are {selected_data.time}\n"
                f"expected shape={self._shape_of_example}\n"
                f"actual shape {selected_data.shape}"
            )
            logger.error(m)
            raise RuntimeError(m)

        return selected_data.load().to_dataset(name="data")

    def geospatial_border(self) -> list[tuple[Number, Number]]:
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
