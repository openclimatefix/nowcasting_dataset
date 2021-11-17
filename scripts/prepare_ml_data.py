#!/usr/bin/env python3

"""Pre-prepares batches of data.

Please run `./prepare_ml_data.py --help` for full details!
"""
import logging

import click
from pathy import Pathy

import nowcasting_dataset
from nowcasting_dataset import utils
from nowcasting_dataset.consts import LOG_LEVELS
from nowcasting_dataset.data_sources import ALL_DATA_SOURCE_NAMES
from nowcasting_dataset.manager import Manager

logger = logging.getLogger(__name__)

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
    is_flag=True,
    help=(
        "Overwrite any existing batches in the destination directory, for the selected"
        " DataSource(s).  If this flag is not set, and if there are existing batches,"
        " then this script will start generating new batches (if necessary) after the"
        " existing batches."
    ),
)
@click.option(
    "--log_level",
    default="DEBUG",
    type=click.Choice(LOG_LEVELS),
    help=("The log level represented as a string.  Defaults to DEBUG."),
)
@utils.arg_logger
def main(config_filename: str, data_source: list[str], overwrite_batches: bool, log_level=str):
    """Generate pre-prepared batches of data."""
    manager = Manager()
    manager.load_yaml_configuration(config_filename)
    manager.configure_loggers(log_level=log_level, names_of_selected_data_sources=data_source)
    manager.initialise_data_sources(names_of_selected_data_sources=data_source)
    manager.initialize_data_sources(names_of_selected_data_sources=data_source)
    # TODO: Issue 323: maybe don't allow
    # create_files_specifying_spatial_and_temporal_locations_of_each_example to be run if a subset
    # of data_sources is passed in at the command line.
    manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()
    manager.create_batches(overwrite_batches)
    manager.create_derived_batches(overwrite_batches)
    manager.save_yaml_configuration()
    # TODO: Issue #317: Validate ML data.
    logger.info("Done!")


if __name__ == "__main__":
    main()
