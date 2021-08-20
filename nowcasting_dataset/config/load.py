import logging
import gcsfs
import os
import io
import yaml
from nowcasting_dataset.config.model import Configuration

logger = logging.getLogger(__name__)


def load_yaml_configuration(filename: str) -> Configuration:
    """
    Load a yaml file which has a configuration in it
    filename: the file name that you want to load
    Returns: pydantic class
    """

    # load the file to a dictionary
    with open(filename, "r") as stream:
        configuration = yaml.safe_load(stream)

    # turn into pydantic class
    configuration = Configuration(**configuration)

    return configuration


def load_configuration_from_gcs(
    gcp_dir: str, bucket: str = "solar-pv-nowcasting-data", filename: str = "configuration.yaml"
) -> Configuration:
    """
    Load configuration from gcs

    gcp_dir: the directory where the configruation is saved
    bucket: the gcs bucket to load from
    filename: the filename that will be loaded

    Returns: configuration class
    """

    logger.info("Loading configuration from gcs")

    bucket_and_dir = os.path.join(f"gs://{bucket}", gcp_dir)
    filename = os.path.join(bucket_and_dir, filename)
    logger.debug(f'Will be opening {filename}')

    # set up gcs
    gcs = gcsfs.GCSFileSystem(access="read_only")

    # load the file into bytes
    with gcs.open(filename, mode="rb") as file:
        file_bytes = file.read()

    # load the bytes to yaml
    with io.BytesIO(file_bytes) as file:
        data = yaml.load(file)

    # put into pydantic class and returns
    return Configuration(**data)
