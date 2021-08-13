import yaml
from .model import Configuration


def load_yaml_configuration(filename: str) -> Configuration:
    """
    Load a yaml file which has the a configruation in it
    filename: the file name that you want to load
    Returns: pydnaictic class
    """

    # load the file to a dictionary
    with open(filename, "r") as stream:
        configuration = yaml.safe_load(stream)

    # turn into pydantic class
    configuration = Configuration(**configuration)

    return configuration
