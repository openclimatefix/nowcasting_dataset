import yaml
import logging
import fsspec
from pathy import Pathy
from nowcasting_dataset.config.model import Configuration
from typing import Optional, Union

logger = logging.getLogger(__name__)


def save_yaml_configuration(
    configuration: Configuration, filename: Optional[Union[str, Pathy]] = None
):
    """
    Save a local yaml file which has the a configuration in it.

    If `filename` is None then saves to configuration.output_data.filepath / configuration.yaml.

    Will save to GCP, AWS, or local, depending on the protocol suffix of filepath.
    """

    # make a dictionary from the configuration
    d = configuration.dict()
    if filename is None:
        filename = Pathy(configuration.output_data.filepath) / "configuration.yaml"
    logger.info(f"Saving configuration to {filename}")

    # save to a yaml file
    with fsspec.open(filename, "w") as yaml_file:
        yaml.safe_dump(d, yaml_file, default_flow_style=False)
