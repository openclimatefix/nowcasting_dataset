"""Manager class."""

import logging
from concurrent import futures
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# nowcasting_dataset imports
import nowcasting_dataset.time as nd_time
import nowcasting_dataset.utils as nd_utils
from nowcasting_dataset import config
from nowcasting_dataset.consts import (
    SPATIAL_AND_TEMPORAL_LOCATIONS_COLUMN_NAMES,
    SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME,
)
from nowcasting_dataset.data_sources import ALL_DATA_SOURCE_NAMES, MAP_DATA_SOURCE_NAME_TO_CLASS
from nowcasting_dataset.dataset.split import split
from nowcasting_dataset.filesystem import utils as nd_fs_utils

logger = logging.getLogger(__name__)


class Manager:
    """The Manager initialises and manage a dict of DataSource objects.

    Attrs:
      config: Configuration object.
      data_sources: dict[str, DataSource]
      data_source_which_defines_geospatial_locations: DataSource: The DataSource used to compute the
        geospatial locations of each example.
      save_batches_locally_and_upload: bool: Set to True by `load_yaml_configuration()` if
        `config.process.upload_every_n_batches > 0`.
      local_temp_path: Path: `config.process.local_temp_path` with `~` expanded.
    """

    def __init__(self) -> None:  # noqa: D107
        self.config = None
        self.data_sources = {}
        self.data_source_which_defines_geospatial_locations = None

    def load_yaml_configuration(self, filename: str) -> None:
        """Load YAML config from `filename`."""
        logger.debug(f"Loading YAML configuration file {filename}")
        self.config = config.load_yaml_configuration(filename)
        self.config = config.set_git_commit(self.config)
        self.save_batches_locally_and_upload = self.config.process.upload_every_n_batches > 0

        # TODO: Issue #320: This could be done in the Pydantic model?
        self.local_temp_path = Path(self.config.process.local_temp_path).expanduser()
        logger.debug(f"config={self.config}")

    def save_yaml_configuration(self):
        """Save configuration to the 'output_data' location"""
        config.save_yaml_configuration(configuration=self.config)

    def initialise_data_sources(
        self, names_of_selected_data_sources: Optional[list[str]] = ALL_DATA_SOURCE_NAMES
    ) -> None:
        """Initialise DataSources specified in the InputData configuration.

        For each key in each DataSource's configuration object, the string `<data_source_name>_`
        is removed from the key before passing to the DataSource constructor.  This allows us to
        have verbose field names in the configuration YAML files, whilst also using standard
        constructor arguments for DataSources.
        """
        for data_source_name in names_of_selected_data_sources:
            logger.debug(f"Creating {data_source_name} DataSource object.")
            config_for_data_source = getattr(self.config.input_data, data_source_name)
            if config_for_data_source is None:
                logger.info(f"No configuration found for {data_source_name}.")
                continue
            config_for_data_source = config_for_data_source.dict()

            # Strip `<data_source_name>_` from the config option field names.
            config_for_data_source = nd_utils.remove_regex_pattern_from_keys(
                config_for_data_source, pattern_to_remove=f"^{data_source_name}_"
            )

            data_source_class = MAP_DATA_SOURCE_NAME_TO_CLASS[data_source_name]
            try:
                data_source = data_source_class(**config_for_data_source)
            except Exception:
                logger.exception(f"Exception whilst instantiating {data_source_name}!")
                raise
            self.data_sources[data_source_name] = data_source

        # Set data_source_which_defines_geospatial_locations:
        try:
            self.data_source_which_defines_geospatial_locations = self.data_sources[
                self.config.input_data.data_source_which_defines_geospatial_locations
            ]
        except KeyError:
            msg = (
                "input_data.data_source_which_defines_geospatial_locations="
                f"{self.config.input_data.data_source_which_defines_geospatial_locations}"
                " is not a member of the DataSources, so cannot set"
                " self.data_source_which_defines_geospatial_locations!"
                f" The available DataSources are: {list(self.data_sources.keys())}"
            )
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            logger.info(
                f"DataSource `{data_source_name}` set as"
                " data_source_which_defines_geospatial_locations."
            )

    def create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary(
        self,
    ) -> None:
        """Creates CSV files specifying the locations of each example if those files don't exist yet.

        Creates one file per split, in this location:

        `<output_data.filepath> / <split_name> / spatial_and_temporal_locations_of_each_example.csv`

        Creates the output directory if it does not exist.

        Works on any compute environment.
        """
        if self._locations_csv_file_exists():
            logger.info(
                f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME} already exists!"
            )
            return
        logger.info(
            f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME} does not exist so"
            " will create..."
        )
        t0_datetimes = self.get_t0_datetimes_across_all_data_sources(
            freq=self.config.process.t0_datetime_frequency
        )
        split_t0_datetimes = split.split_data(
            datetimes=t0_datetimes, method=self.config.process.split_method
        )
        for split_name, datetimes_for_split in split_t0_datetimes._asdict().items():
            n_batches = self._get_n_batches_for_split_name(split_name)
            n_examples = n_batches * self.config.process.batch_size
            logger.debug(
                f"Creating {n_batches:,d} batches x {self.config.process.batch_size:,d} examples"
                f" per batch = {n_examples:,d} examples for {split_name}."
            )
            df_of_locations = self.sample_spatial_and_temporal_locations_for_examples(
                t0_datetimes=datetimes_for_split, n_examples=n_examples
            )
            output_filename = self._filename_of_locations_csv_file(split_name)
            path_for_csv = self.config.output_data.filepath / split_name
            logger.info(f"Making {path_for_csv} if it does not exist.")
            nd_fs_utils.makedirs(path_for_csv, exist_ok=True)
            logger.debug(f"Writing {output_filename}")
            df_of_locations.to_csv(output_filename)

    def _get_n_batches_for_split_name(self, split_name: str) -> int:
        return getattr(self.config.process, f"n_{split_name}_batches")

    def _filename_of_locations_csv_file(self, split_name: str) -> Path:
        return (
            self.config.output_data.filepath
            / split_name
            / SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
        )

    def _locations_csv_file_exists(self) -> bool:
        """Check if filepath/train/spatial_and_temporal_locations_of_each_example.csv exists."""
        filename = self._filename_of_locations_csv_file(split_name=split.SplitName.TRAIN.value)
        try:
            nd_fs_utils.check_path_exists(filename)
        except FileNotFoundError:
            logging.info(f"{filename} does not exist!")
            return False
        else:
            logger.info(f"{filename} exists!")
            return True

    def get_t0_datetimes_across_all_data_sources(
        self, freq: Union[str, pd.Timedelta]
    ) -> pd.DatetimeIndex:
        """
        Compute the intersection of the t0 datetimes available across all DataSources.

        Args:
            freq: The Pandas frequency string. The returned DatetimeIndex will be at this frequency,
                and every datetime will be aligned to this frequency.  For example, if
                freq='5 minutes' then every datetime will be at 00, 05, ..., 55 minutes
                past the hour.

        Returns:  Valid t0 datetimes, taking into consideration all DataSources,
            filtered by daylight hours (SatelliteDataSource.datetime_index() removes the night
            datetimes).
        """
        logger.debug(
            f"Getting the intersection of time periods across all DataSources at freq={freq}..."
        )
        if set(self.data_sources.keys()) != set(ALL_DATA_SOURCE_NAMES):
            logger.warning(
                "Computing available t0 datetimes using less than all available DataSources!"
                " Are you sure you mean to do this?!"
            )

        # Get the intersection of t0 time periods from all data sources.
        t0_time_periods_for_all_data_sources = []
        for data_source_name, data_source in self.data_sources.items():
            logger.debug(f"Getting t0 time periods for {data_source_name}")
            try:
                t0_time_periods = data_source.get_contiguous_t0_time_periods()
            except NotImplementedError:
                # Skip data_sources with no concept of time.
                logger.debug(f"Skipping {data_source_name} because it has not concept of datetime.")
            else:
                t0_time_periods_for_all_data_sources.append(t0_time_periods)

        intersection_of_t0_time_periods = nd_time.intersection_of_multiple_dataframes_of_periods(
            t0_time_periods_for_all_data_sources
        )

        t0_datetimes = nd_time.time_periods_to_datetime_index(
            time_periods=intersection_of_t0_time_periods, freq=freq
        )
        logger.debug(
            f"Found {len(t0_datetimes):,d} datetimes at freq=`{freq}` across"
            f" DataSources={list(self.data_sources.keys())}."
            f"  From {t0_datetimes[0]} to {t0_datetimes[-1]}."
        )
        return t0_datetimes

    def sample_spatial_and_temporal_locations_for_examples(
        self, t0_datetimes: pd.DatetimeIndex, n_examples: int
    ) -> pd.DataFrame:
        """
        Computes the geospatial and temporal locations for each training example.

        The first data_source in this DataSourceList defines the geospatial locations of
        each example.

        Args:
            t0_datetimes: All available t0 datetimes.  Can be computed with
                `DataSourceList.get_t0_datetimes_across_all_data_sources()`
            n_examples: The number of examples requested.

        Returns:
            Each row of each the DataFrame specifies the position of each example, using
            columns: 't0_datetime_UTC', 'x_center_OSGB', 'y_center_OSGB'.
        """
        shuffled_t0_datetimes = np.random.choice(t0_datetimes, size=n_examples)
        # TODO: Issue #304. Speed this up by splitting the shuffled_t0_datetimes across
        # multiple processors.  Currently takes about half an hour for 25,000 batches.
        # But wait until we've implemented issue #305, as that is likely to be sufficient!
        (
            x_locations,
            y_locations,
        ) = self.data_source_which_defines_geospatial_locations.get_locations(shuffled_t0_datetimes)
        return pd.DataFrame(
            {
                "t0_datetime_UTC": shuffled_t0_datetimes,
                "x_center_OSGB": x_locations,
                "y_center_OSGB": y_locations,
            }
        )

    def _get_first_batches_to_create(
        self, overwrite_batches: bool
    ) -> dict[split.SplitName, dict[str, int]]:
        """For each SplitName & for each DataSource name, return the first batch ID to create.

        For example, the returned_dict[SplitName.TRAIN]['gsp'] tells us the first batch_idx to
        create for the training set for the GSPDataSource.
        """
        # Initialise to zero:
        first_batches_to_create: dict[split.SplitName, dict[str, int]] = {}
        for split_name in split.SplitName:
            first_batches_to_create[split_name] = {
                data_source_name: 0 for data_source_name in self.data_sources
            }

        if overwrite_batches:
            return first_batches_to_create

        # If we're not overwriting batches then find the last batch on disk.
        for split_name in split.SplitName:
            for data_source_name in self.data_sources:
                path = (
                    self.config.output_data.filepath / split_name.value / data_source_name / "*.nc"
                )
                try:
                    max_batch_id_on_disk = nd_fs_utils.get_maximum_batch_id(path)
                except FileNotFoundError:
                    max_batch_id_on_disk = -1
                first_batches_to_create[split_name][data_source_name] = max_batch_id_on_disk + 1

        return first_batches_to_create

    def _check_if_more_batches_are_required_for_split(
        self,
        split_name: split.SplitName,
        first_batches_to_create: dict[split.SplitName, dict[str, int]],
    ) -> bool:
        """Returns True if batches still need to be created for any DataSource."""
        n_batches_requested = self._get_n_batches_for_split_name(split_name.value)
        for data_source_name in self.data_sources:
            if first_batches_to_create[split_name][data_source_name] < n_batches_requested:
                return True
        return False

    def _find_splits_which_need_more_batches(
        self, first_batches_to_create: dict[split.SplitName, dict[str, int]]
    ) -> list[split.SplitName]:
        """Returns list of SplitNames which need more batches to be produced."""
        splits_which_need_more_batches = []
        for split_name in split.SplitName:
            if self._check_if_more_batches_are_required_for_split(
                split_name, first_batches_to_create
            ):
                splits_which_need_more_batches.append(split_name)
        return splits_which_need_more_batches

    def create_batches(self, overwrite_batches: bool) -> None:
        """Create batches (if necessary).

        Make dirs: `<output_data.filepath> / <split_name> / <data_source_name>`.

        Also make `local_temp_path` if necessary.

        Args:
          overwrite_batches: If True then start from batch 0, regardless of which batches have
            previously been written to disk. If False then check which batches have previously been
            written to disk, and only create any batches which have not yet been written to disk.
        """
        first_batches_to_create = self._get_first_batches_to_create(overwrite_batches)

        # Check if there's any work to do.
        if overwrite_batches:
            splits_which_need_more_batches = [split_name for split_name in split.SplitName]
        else:
            splits_which_need_more_batches = self._find_splits_which_need_more_batches(
                first_batches_to_create
            )
            if len(splits_which_need_more_batches) == 0:
                logger.info("All batches have already been created!  No work to do!")
                return

        # Load locations for each example off disk.
        locations_for_each_example_of_each_split: dict[split.SplitName, pd.DataFrame] = {}
        for split_name in splits_which_need_more_batches:
            filename = self._filename_of_locations_csv_file(split_name.value)
            logger.info(f"Loading {filename}.")
            locations_for_each_example = pd.read_csv(filename, index_col=0)
            assert locations_for_each_example.columns.to_list() == list(
                SPATIAL_AND_TEMPORAL_LOCATIONS_COLUMN_NAMES
            )
            # Converting to datetimes is much faster using `pd.to_datetime()` than
            # passing `parse_datetimes` into `pd.read_csv()`.
            locations_for_each_example["t0_datetime_UTC"] = pd.to_datetime(
                locations_for_each_example["t0_datetime_UTC"]
            )
            locations_for_each_example_of_each_split[split_name] = locations_for_each_example

        # Fire up a separate process for each DataSource, and pass it a list of batches to
        # create, and whether to utils.upload_and_delete_local_files().
        # TODO: Issue 321: Split this up into separate functions!!!
        n_data_sources = len(self.data_sources)
        nd_utils.set_fsspec_for_multiprocess()
        for split_name in splits_which_need_more_batches:
            locations_for_split = locations_for_each_example_of_each_split[split_name]
            with futures.ProcessPoolExecutor(max_workers=n_data_sources) as executor:
                future_create_batches_jobs = []
                for worker_id, (data_source_name, data_source) in enumerate(
                    self.data_sources.items()
                ):
                    # Get indexes of first batch and example. And subset locations_for_split.
                    idx_of_first_batch = first_batches_to_create[split_name][data_source_name]
                    idx_of_first_example = idx_of_first_batch * self.config.process.batch_size
                    locations = locations_for_split.loc[idx_of_first_example:]

                    # Get paths.
                    dst_path = (
                        self.config.output_data.filepath / split_name.value / data_source_name
                    )
                    local_temp_path = (
                        self.local_temp_path
                        / split_name.value
                        / data_source_name
                        / f"worker_{worker_id}"
                    )

                    # Make folders.
                    nd_fs_utils.makedirs(dst_path, exist_ok=True)
                    if self.save_batches_locally_and_upload:
                        nd_fs_utils.makedirs(local_temp_path, exist_ok=True)

                    # Submit data_source.create_batches task to the worker process.
                    future = executor.submit(
                        data_source.create_batches,
                        spatial_and_temporal_locations_of_each_example=locations,
                        idx_of_first_batch=idx_of_first_batch,
                        batch_size=self.config.process.batch_size,
                        dst_path=dst_path,
                        local_temp_path=local_temp_path,
                        upload_every_n_batches=self.config.process.upload_every_n_batches,
                    )
                    future_create_batches_jobs.append(future)

                # Wait for all futures to finish:
                for future, data_source_name in zip(
                    future_create_batches_jobs, self.data_sources.keys()
                ):
                    # Call exception() to propagate any exceptions raised by the worker process into
                    # the main process, and to wait for the worker to finish.
                    exception = future.exception()
                    if exception is not None:
                        logger.exception(f"Worker process {data_source_name} raised exception!")
                        raise exception
