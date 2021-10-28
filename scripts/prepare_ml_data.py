#!/usr/bin/env python3

"""Pre-prepares batches of data.
"""
import logging
import click
from pathy import Pathy

# nowcasting_dataset imports
import nowcasting_dataset
from nowcasting_dataset.manager import Manager
from nowcasting_dataset.data_sources import ALL_DATA_SOURCE_NAMES
from nowcasting_dataset import utils

# Set up logging.
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s at %(pathname)s#L%(lineno)d")
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
        " --data_source nwp --data_source satellite.  Note that only these DataSources"
        " always be used when computing the available datetime periods across all the"
        " DataSources, so be very careful about setting --data_source when creating the"
        " spatial_and_temporal_locations_of_each_example.csv files!"
    ),
)
@click.option(
    "--overwrite_batches",
    default=False,
    help=(
        "Overwrite any existing batches in the destination directory, for the selected"
        " DataSource(s).  If this flag is not set, and if there are existing batches,"
        " then this script will start generating new batches (if necessary) after the"
        " existing batches."
    ),
)
@utils.arg_logger
def main(config_filename: str, data_source: list[str], overwrite_batches: bool):
    """Generate pre-prepared batches of data."""
    manager = Manager()
    manager.load_yaml_configuration(config_filename)
    manager.initialise_data_sources(names_of_selected_data_sources=data_source)
    manager.make_destination_paths_if_necessary()
    # manager.check_paths_exist()
    manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()
    # TODO: If not overwrite, for each DataSource, get the maximum_batch_id already on disk.
    # TODO: Load spatial_and_temporal_locations_of_each_example.csv files
    # TODO: Fire up a separate process for each DataSource, and pass it a list of batches to create, and whether to utils.upload_and_delete_local_files()
    # TODO: Wait for all processes to complete.
    # TODO: save_yaml_configuration(config)
    # TODO: Validate ML data.


if __name__ == "__main__":
    main()
