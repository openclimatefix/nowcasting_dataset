"""Manager live class."""

import logging
import multiprocessing
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import pandas as pd

import nowcasting_dataset.utils as nd_utils
from nowcasting_dataset import config
from nowcasting_dataset.consts import (
    SPATIAL_AND_TEMPORAL_LOCATIONS_COLUMN_NAMES,
    SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME,
)
from nowcasting_dataset.data_sources import ALL_DATA_SOURCE_NAMES, MAP_DATA_SOURCE_NAME_TO_CLASS
from nowcasting_dataset.filesystem import utils as nd_fs_utils
from nowcasting_dataset.manager.utils import callback, error_callback

logger = logging.getLogger(__name__)


class ManagerLive:
    """The Manager initialises and manage a dict of DataSource objects.

    1. Load configuration file
    2. Initialize the data sources
    3. Make Locations
    4. Create batches.
        For GSP there will be 338 examples.
        For a batch size of 32, there will be 11 batches,
        and therefore we upload at the end of making all the batches

    Attrs:
      config: Configuration object.
      data_sources: dict[str, DataSource]
      data_source_which_defines_geospatial_locations: DataSource: The DataSource used to compute the
        geospatial locations of each example.
      save_batches_locally_and_upload: bool: Set to True by `load_yaml_configuration()` if
        `config.process.upload_every_n_batches > 0`.
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
        logger.debug(f"config={self.config}")

    def initialise_data_sources(
        self, names_of_selected_data_sources: Optional[list[str]] = ALL_DATA_SOURCE_NAMES
    ) -> None:
        """Initialize DataSources specified in the InputData configuration.

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
            config_for_data_source.pop("log_level")

            # save config to data source logger
            data_source_logger = logging.getLogger(
                f"nowcasting_dataset.data_sources.{data_source_name}"
            )
            data_source_logger.debug(
                f"The configuration for {data_source_name} is {config_for_data_source}"
            )

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

    def create_files_specifying_spatial_and_temporal_locations_of_each_example(
        self, t0_datetime: datetime
    ) -> None:
        """Creates CSV files specifying the locations of each example if those files don't exist yet.

        - Creates one file per split, in this location:
        `<output_data.filepath> / 'live' / spatial_and_temporal_locations_of_each_example.csv`
        - Creates the output directory if it does not exist.
        - This works on any compute environment.
        """

        logger.info(
            f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME} does not exist so"
            " will create..."
        )

        split_name = "live"
        datetimes_for_split = [t0_datetime]

        path_for_csv = self.config.output_data.filepath / split_name
        n_batches_requested = self._get_n_batches_requested_for_split_name(split_name)
        if (n_batches_requested == 0 and len(datetimes_for_split) != 0) or (
            len(datetimes_for_split) == 0 and n_batches_requested != 0
        ):
            # TODO: Issue #450: Test this scenario!
            msg = (
                f"For split {split_name}: n_{split_name}_batches={n_batches_requested} and"
                f" {len(datetimes_for_split)=}!  This is an error!"
                f"  If n_{split_name}_batches==0 then len(datetimes_for_split) must also"
                f" equal 0, and visa-versa!  Please check `n_{split_name}_batches` and"
                " `split_method` in the config YAML!"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if n_batches_requested == 0:
            logger.info(f"0 batches requested for {split_name} so won't create {path_for_csv}")
        else:
            n_examples = n_batches_requested * self.config.process.batch_size
            logger.debug(
                f"Creating {n_batches_requested:,d} batches x {self.config.process.batch_size:,d}"
                f" examples per batch = {n_examples:,d} examples for {split_name}."
            )

            # # for the test set, we want to get all locations for each datetime,
            # # for the train and validation set, we just want one location for each datetime
            # if split_name == split.SplitName.TEST.value:
            #     get_all_locations = True
            # else:
            #     get_all_locations = False

            df_of_locations = self.sample_spatial_and_temporal_locations_for_examples(
                t0_datetime=datetimes_for_split[0],
            )
            output_filename = self._filename_of_locations_csv_file()
            logger.info(f"Making {path_for_csv} if it does not exist.")
            nd_fs_utils.makedirs(path_for_csv, exist_ok=True)
            logger.debug(f"Writing {output_filename}")
            df_of_locations.to_csv(output_filename)

    def _get_n_batches_requested_for_split_name(self, split_name: str) -> int:
        # TODO make dynamic
        return 338

    def _filename_of_locations_csv_file(self) -> Path:
        return (
            self.config.output_data.filepath
            / "live"
            / SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
        )

    def sample_spatial_and_temporal_locations_for_examples(
        self, t0_datetime: datetime
    ) -> pd.DataFrame:
        """
        Computes the geospatial and temporal locations for each training example.

        The first data_source in this DataSourceList defines the geospatial locations of
        each example.

        Args:
            t0_datetimes: All available t0 datetimes.  Can be computed with
                `DataSourceList.get_t0_datetimes_across_all_data_sources()`
            n_examples: The number of examples requested.
            get_all_locations: optional to return all locations for each t0_datetime

        Returns:
            Each row of each the DataFrame specifies the position of each example, using
            columns: 't0_datetime_UTC', 'x_center_OSGB', 'y_center_OSGB'.
        """

        # note that the returned 'shuffled_t0_datetimes'
        # has duplicate datetimes for each location
        (
            shuffled_t0_datetimes,
            x_locations,
            y_locations,
        ) = self.data_source_which_defines_geospatial_locations.get_all_locations(
            t0_datetimes_utc=pd.DatetimeIndex([t0_datetime])
        )
        shuffled_t0_datetimes = list(shuffled_t0_datetimes)

        # find out the number of examples in the last batch,
        # we maybe need to duplicate the last example into order to get a full batch
        n_examples_last_batch = len(shuffled_t0_datetimes) % self.config.process.batch_size
        # Note 0 means the examples fit into the batches

        if n_examples_last_batch != 0:
            # add extra examples to make sure the number of examples fits into batches.
            # There could be a different way to do this,
            # but this keeps it pretty simple for the moment
            extra_examples_needed = self.config.process.batch_size - n_examples_last_batch
            for _ in range(0, extra_examples_needed):
                shuffled_t0_datetimes.append(shuffled_t0_datetimes[0])
                x_locations.append(x_locations[0])
                y_locations.append(y_locations[0])

        return pd.DataFrame(
            {
                "t0_datetime_UTC": shuffled_t0_datetimes,
                "x_center_OSGB": x_locations,
                "y_center_OSGB": y_locations,
            }
        )

    def create_batches(self, overwrite_batches: bool) -> None:
        """Create batches (if necessary).

        Make dirs: `<output_data.filepath> / <split_name> / <data_source_name>`.

        Also make `local_temp_path` if necessary.

        Args:
          overwrite_batches: If True then start from batch 0, regardless of which batches have
            previously been written to disk. If False then check which batches have previously been
            written to disk, and only create any batches which have not yet been written to disk.
        """
        logger.debug("Entering Manager.create_batches...")

        # Load locations for each example off disk.
        # locations_for_each_example_of_each_split: dict[split.SplitName, pd.DataFrame] = {}
        # for split_name in splits_which_need_more_batches:
        filename = self._filename_of_locations_csv_file()
        logger.info(f"Loading {filename}.")

        # TODO add pydantic model for this
        locations_for_each_example = pd.read_csv(filename, index_col=0)
        assert locations_for_each_example.columns.to_list() == list(
            SPATIAL_AND_TEMPORAL_LOCATIONS_COLUMN_NAMES
        )
        # Converting to datetimes is much faster using `pd.to_datetime()` than
        # passing `parse_datetimes` into `pd.read_csv()`.
        locations_for_each_example["t0_datetime_UTC"] = pd.to_datetime(
            locations_for_each_example["t0_datetime_UTC"]
        )

        # Fire up a separate process for each DataSource, and pass it a list of batches to
        # create, and whether to utils.upload_and_delete_local_files().
        # TODO: Issue 321: Split this up into separate functions!!!
        n_data_sources = len(self.data_sources)
        nd_utils.set_fsspec_for_multiprocess()

        split_name = "live"
        # TODO Need to think if this is the correct option for live.
        with multiprocessing.Pool(processes=n_data_sources) as pool:
            async_results_from_create_batches = []
            an_error_has_occured = multiprocessing.Event()
            for worker_id, (data_source_name, data_source) in enumerate(self.data_sources.items()):

                # Get indexes of first batch and example. And subset locations_for_split.
                idx_of_first_batch = 0
                locations = locations_for_each_example

                # Get paths.
                dst_path = self.config.output_data.filepath / split_name / data_source_name

                # TODO: Issue 455: Guarantee that local temp path is unique and empty.
                local_temp_path = (
                    self.config.process.local_temp_path
                    / split_name
                    / data_source_name
                    / f"worker_{worker_id}"
                )

                # Make folders.
                nd_fs_utils.makedirs(dst_path, exist_ok=True)
                if dst_path != local_temp_path:
                    nd_fs_utils.makedirs(local_temp_path, exist_ok=True)

                # Key word arguments to be passed into data_source.create_batches():
                kwargs_for_create_batches = dict(
                    spatial_and_temporal_locations_of_each_example=locations,
                    idx_of_first_batch=idx_of_first_batch,
                    batch_size=self.config.process.batch_size,
                    dst_path=dst_path,
                    local_temp_path=local_temp_path,
                    upload_every_n_batches=self.config.process.upload_every_n_batches,
                )

                # Submit data_source.create_batches task to the worker process.
                logger.debug(
                    f"About to submit create_batches task for {data_source_name}, {split_name}"
                )
                async_result = pool.apply_async(
                    data_source.create_batches,
                    kwds=kwargs_for_create_batches,
                    callback=partial(
                        callback, data_source_name=data_source_name, split_name=split_name
                    ),
                    error_callback=partial(
                        error_callback,
                        data_source_name=data_source_name,
                        split_name=split_name,
                        an_error_has_occured=an_error_has_occured,
                    ),
                )
                async_results_from_create_batches.append(async_result)

            # Wait for all async_results to finish:
            for async_result in async_results_from_create_batches:
                async_result.wait()
                if an_error_has_occured.is_set():
                    # An error has occurred but, at this point in the code, we don't know which
                    # worker process raised the exception.  But, with luck, the worker process
                    # will have logged an informative exception via the _error_callback func.
                    raise RuntimeError(
                        f"A worker process raised an exception whilst working on {split_name}!"
                    )

            logger.info(f"Finished creating batches for {split_name}!")
