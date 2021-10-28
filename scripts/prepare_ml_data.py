#!/usr/bin/env python3

"""Pre-prepares batches of data.
"""

import logging
import click
from pathy import Pathy
import nowcasting_dataset
from nowcasting_dataset.manager import Manager, ALL_DATA_SOURCE_NAMES

# Set up logging.
logging.basicConfig(format="%(asctime)s %(levelname)s %(pathname)s %(lineno)d %(message)s")
logging.getLogger("nowcasting_dataset.data_source").setLevel(logging.WARNING)
logger = logging.getLogger("nowcasting_dataset")
logger.setLevel(logging.DEBUG)

default_config_filename = Pathy(nowcasting_dataset.__file__).parent / "config" / "on_premises.yaml"


@click.command()
@click.option(
    "--config_filename",
    default=default_config_filename,
    help="The filename of the YAML configuration file.",
)
@click.option(
    "--data_source",
    multiple=True,
    default=ALL_DATA_SOURCE_NAMES,
    type=click.Choice(ALL_DATA_SOURCE_NAMES),
    help=(
        "If you want to process just a subset of the DataSources in the config file, then enter"
        " the names of those DataSources using the --data_source option.  Enter one name per"
        " --data_source option.  You can use --data_source multiple times.  For example:"
        " --data_source nwp --data_source satellite."
    ),
)
@click.option(
    "--overwrite",
    default=False,
    help=(
        "Overwrite any existing batches in the destination directory, for the selected"
        " DataSource(s).  If this flag is not set, and if there are existing batches,"
        " then this script will start generating new batches (if necessary) after the"
        " existing batches."
    ),
)
def main(config_filename: str, data_source: list[str], overwrite: bool):
    """Generate pre-prepared batches of data."""
    logger.info(f"config_filename={config_filename}")
    logger.info(f"data_sources={data_source}")
    logger.info(f"overwrite={overwrite}")
    manager = Manager()
    manager.load_yaml_configuration(config_filename)
    manager.initialise_data_sources(names_of_selected_data_sources=data_source)
    manager.make_destination_paths()
    manager.check_paths_exist()
    # TODO: If not overwrite, for each DataSource, get the maximum_batch_id already on disk.
    # TODO: Check if the spatial_and_temporal_locations_of_each_example.csv files exist. If not, create these files.
    # TODO: Load spatial_and_temporal_locations_of_each_example.csv files
    # TODO: Fire up a separate process for each DataSource, and pass it a list of batches to create, and whether to utils.upload_and_delete_local_files()
    # TODO: Wait for all processes to complete.
    # TODO: save_yaml_configuration(config)
    # TODO: Validate ML data.


if __name__ == "__main__":
    main()
