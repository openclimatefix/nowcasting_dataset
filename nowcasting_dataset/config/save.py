import yaml
import logging
import gcsfs
import tempfile
import os
from nowcasting_dataset.config.model import Configuration

logger = logging.getLogger(__name__)


def save_yaml_configuration(configuration: Configuration, filename: str):
    """
    Load a local yaml file which has the a configuration in it
    filename: the file name that you want to load
    Returns: pydnaictic class
    """

    # make a dictionary from the configuration
    d = configuration.dict()

    # save to a yaml file
    with open(filename, "w") as yaml_file:
        yaml.safe_dump(d, yaml_file, default_flow_style=False)


def save_configuration_to_gcs(configuration: Configuration):
    """
    Save configuration to gcs
    """

    logger.info('Uploading configuration to gcs')
    gcp_filepath = os.path.join(configuration.output_data.filepath, 'configuration.yaml')

    with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:
        # save configuration to temp file
        save_yaml_configuration(configuration=configuration, filename=fp.name)

        # save file to gcs
        logger.debug(f'Will be saving file to {gcp_filepath}')
        gcs = gcsfs.GCSFileSystem()
        gcs.put(fp.name, gcp_filepath)
