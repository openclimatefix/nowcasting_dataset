from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.config.save import save_yaml_configuration
from nowcasting_dataset.config.model import Configuration
import nowcasting_dataset
import os
import tempfile
import pytest


def test_default():
    """
    Test default pydantic class
    """

    _ = Configuration()


def test_yaml_load():
    """Test that yaml loading works"""

    filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "example.yaml")

    config = load_yaml_configuration(filename)

    assert isinstance(config, Configuration)


def test_yaml_save():
    """
    Check a configuration can be saved to a yaml file
    """

    with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:

        name = fp.name

        # check that temp file cant be loaded
        with pytest.raises(TypeError):
            _ = load_yaml_configuration(name)

        # save default config to file
        save_yaml_configuration(Configuration(), name)

        # check the file can be loaded
        _ = load_yaml_configuration(name)
