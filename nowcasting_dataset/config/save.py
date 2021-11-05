""" Save functions for the configuration model"""
import json
import logging
from typing import Optional, Union

import fsspec
import yaml
from pathy import Pathy

from nowcasting_dataset.config.model import Configuration

logger = logging.getLogger(__name__)


def save_yaml_configuration(
    configuration: Configuration, filename: Optional[Union[str, Pathy]] = None
):
    """
    Save a local yaml file which has the a configuration in it.

    If `filename` is None then saves to configuration.output_data.filepath / configuration.yaml.

    Will save to GCP, AWS, or local, depending on the protocol suffix of filepath.
    """
    # make a dictionary from the configuration,
    # Note that we make the object json'able first, so that it can be saved to a yaml file
    d = json.loads(configuration.json())
    if filename is None:
        filename = Pathy(configuration.output_data.filepath) / "configuration.yaml"
    logger.info(f"Saving configuration to {filename}")

    # save to a yaml file
    with fsspec.open(filename, "w") as yaml_file:
        yaml.safe_dump(d, yaml_file, default_flow_style=False)
