import yaml
from .model import Configuration


def save_yaml_configuration(configuration: Configuration, filename: str):
    """
    Load a yaml file which has the a configruation in it
    filename: the file name that you want to load
    Returns: pydnaictic class
    """

    # make a dictionary from the configuration
    d = configuration.dict()

    # save to a yaml file
    with open(filename, "w") as yaml_file:
        yaml.safe_dump(d, yaml_file, default_flow_style=False)
