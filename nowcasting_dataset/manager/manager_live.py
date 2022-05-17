"""Manager live class."""

import logging
import multiprocessing
from datetime import datetime
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd

import nowcasting_dataset.utils as nd_utils
from nowcasting_dataset.consts import (
    N_GSPS,
    SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME,
)
from nowcasting_dataset.data_sources.metadata.metadata_model import (
    Metadata,
    SpaceTimeLocation,
    load_from_csv,
)
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.filesystem import utils as nd_fs_utils
from nowcasting_dataset.manager.base import ManagerBase
from nowcasting_dataset.manager.utils import callback, error_callback

logger = logging.getLogger(__name__)


class ManagerLive(ManagerBase):
    """The Manager initializes and manage a dict of DataSource objects.

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

    def create_files_specifying_spatial_and_temporal_locations_of_each_example(
        self, t0_datetime: datetime, n_gsps: Optional[int] = N_GSPS
    ) -> None:
        """
        Creates CSV files specifying the locations of each example if those files don't exist yet.

        - Creates one file per split, in this location:
        `<output_data.filepath> / 'live' / spatial_and_temporal_locations_of_each_example.csv`
        - Creates the output directory if it does not exist.
        - This works on any compute environment.

        Args:
            t0_datetime: The datetime for the batch
            n_gsps: the number of gsps we want to make, this is defaulted to N_GSPS=338

        Returns: Nothing
        """

        logger.info(
            f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME} does not exist so"
            " will create..."
        )

        split_name = "live"
        datetimes_for_split = [t0_datetime]

        path_for_csv = self.config.output_data.filepath / split_name

        n_batches_requested = int(np.ceil(n_gsps / self.config.process.batch_size))

        logger.debug(
            f"Creating {n_batches_requested:,d} batches x {self.config.process.batch_size:,d}"
            f" examples per batch = {n_batches_requested:,d} examples for {split_name}."
        )

        locations = self.sample_spatial_and_temporal_locations_for_examples(
            t0_datetime=datetimes_for_split[0], n_examples=n_gsps
        )

        metadata = Metadata(
            batch_size=self.config.process.batch_size, space_time_locations=locations
        )
        # output_filename = self._filename_of_locations_csv_file(split_name="live")
        logger.info(f"Making {path_for_csv} if it does not exist.")
        nd_fs_utils.makedirs(path_for_csv, exist_ok=True)
        logger.debug(f"Writing {path_for_csv}")
        metadata.save_to_csv(path_for_csv)

    def sample_spatial_and_temporal_locations_for_examples(
        self, t0_datetime: datetime, n_examples: Optional[int] = None
    ) -> List[SpaceTimeLocation]:
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
        locations: List[
            SpaceTimeLocation
        ] = self.data_source_which_defines_geospatial_locations.get_all_locations(
            t0_datetimes_utc=pd.DatetimeIndex([t0_datetime])
        )

        # reduce locations to n_examples
        if n_examples is not None:
            locations = locations[:n_examples]

        # find out the number of examples in the last batch,
        # we maybe need to duplicate the last example into order to get a full batch
        n_examples_last_batch = len(locations) % self.config.process.batch_size
        # Note 0 means the examples fit into the batches

        if n_examples_last_batch != 0:
            # add extra examples to make sure the number of examples fits into batches.
            # There could be a different way to do this,
            # but this keeps it pretty simple for the moment
            extra_examples_needed = self.config.process.batch_size - n_examples_last_batch
            for _ in range(0, extra_examples_needed):
                locations.append(
                    SpaceTimeLocation(
                        t0_datetime_utc=locations[0].t0_datetime_utc,
                        x_center_osgb=locations[0].x_center_osgb,
                        y_center_osgb=locations[0].y_center_osgb,
                        id=locations[0].id,
                    )
                )

        return locations

    def create_batches(self, use_async: Optional[bool] = True) -> None:
        """Create batches (if necessary).

        Make dirs: `<output_data.filepath> / <split_name> / <data_source_name>`.

        Also make `local_temp_path` if necessary.

        """
        logger.debug("Entering Manager.create_batches...")

        # Load locations for each example off disk.
        # locations_for_each_example_of_each_split: dict[split.SplitName, pd.DataFrame] = {}
        # for split_name in splits_which_need_more_batches:
        filename = self._filename_of_locations_csv_file(split_name="live")
        logger.info(f"Loading {filename}.")

        # TODO add pydantic model for this
        metadata = load_from_csv(path=filename, batch_size=self.config.process.batch_size)
        locations_for_each_example = metadata.space_time_locations

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

                if ~use_async:
                    # Sometimes when debuggin it is easy to use non async
                    data_source.create_batches(**kwargs_for_create_batches)
                else:

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
                            # An error has occurred but, at this point in the code,
                            # we don't know which worker process raised the exception.
                            # But, with luck, the worker process
                            # will have logged an informative exception via the
                            # _error_callback func.
                            raise RuntimeError(
                                f"A worker process raised an exception whilst "
                                f"working on {split_name}!"
                            )

            logger.info(f"Finished creating batches for {split_name}!")

    def save_batch(self, batch_idx, path: str):
        """
        Option to save batch to a differnt location

        Args:
            path: the path to save the file to
            batch_idx: the batch index

        """

        # load batch
        batch = Batch.load_netcdf(
            batch_idx=batch_idx,
            data_sources_names=self.data_sources,
            local_netcdf_path=self.config.output_data.filepath / "live",
        )

        # save batch
        batch.save_netcdf(batch_i=batch_idx, path=path)
