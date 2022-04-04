"""Base Manager class."""

import logging
from pathlib import Path
from typing import Optional

import nowcasting_dataset.utils as nd_utils
from nowcasting_dataset import config
from nowcasting_dataset.data_sources import ALL_DATA_SOURCE_NAMES, MAP_DATA_SOURCE_NAME_TO_CLASS

logger = logging.getLogger(__name__)


class ManagerBase:
    """The Manager initializes and manage a dict of DataSource objects.

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

    def load_yaml_configuration(self, filename: str, set_git: bool = True) -> None:
        """Load YAML config from `filename`."""
        logger.debug(f"Loading YAML configuration file {filename}")
        self.config = config.load_yaml_configuration(filename)
        if set_git:
            self.config = config.set_git_commit(self.config)
        self.save_batches_locally_and_upload = self.config.process.upload_every_n_batches > 0
        logger.debug(f"config={self.config}")

    def initialize_data_sources(
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

            # TODO: #631 remove
            if data_source_name == "pv":
                config_for_data_source.pop("filename")
                config_for_data_source.pop("metadata_filename")

            data_source_class = MAP_DATA_SOURCE_NAME_TO_CLASS[data_source_name]
            try:
                data_source = data_source_class(**config_for_data_source)
            except Exception:
                logger.exception(
                    f"Exception whilst instantiating {data_source_name}! "
                    f"Tried with configuration {config_for_data_source} "
                    f"in {data_source_class}"
                )
                raise
            self.data_sources[data_source_name] = data_source

        # Set data_source_which_defines_geospatial_locations:
        try:
            self.data_source_which_defines_geospatial_locations = self.data_sources[
                self.config.input_data.data_source_which_defines_geospatial_locations
            ]
        except KeyError:
            if self._locations_csv_file_exists():
                logger.info(
                    f"{self.config.input_data.data_source_which_defines_geospatial_locations=}"
                    " is not a member of the DataSources, but that does not matter because the CSV"
                    " files which specify the locations of the examples already exists!"
                )
            else:
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

    def _locations_csv_file_exists(self):
        return False

    def _filename_of_locations_csv_file(self, split_name: str) -> Path:
        return self.config.output_data.filepath / split_name
