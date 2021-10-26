""" Loading configuration functions """
import logging
from typing import Union

import fsspec
import yaml
from pathy import Pathy

from nowcasting_dataset.config.model import Configuration

logger = logging.getLogger(__name__)


def load_yaml_configuration(filename: Union[str, Pathy]) -> Configuration:
    """
    Load a yaml file which has a configuration in it

    Args:
        filename: the file name that you want to load.  Will load from local, AWS, or GCP
            depending on the protocol suffix (e.g. 's3://bucket/config.yaml').

    Returns:pydantic class

    """
    # load the file to a dictionary
    with fsspec.open(filename, mode="r") as stream:
        configuration = yaml.safe_load(stream)

    # turn into pydantic class
    configuration = Configuration(**configuration)

    return configuration
