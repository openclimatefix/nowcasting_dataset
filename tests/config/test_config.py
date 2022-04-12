"""Test config."""
import os
import tempfile
from datetime import datetime

import pytest
from pathy import Pathy
from pydantic import ValidationError

import nowcasting_dataset
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.config.model import PV, Configuration, set_git_commit
from nowcasting_dataset.config.save import save_yaml_configuration


def test_default():
    """
    Test default pydantic class
    """

    _ = Configuration()


def test_yaml_load():
    """Test that yaml loading works"""

    filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")

    config = load_yaml_configuration(filename)

    assert isinstance(config, Configuration)
    assert type(config.output_data.filepath) == Pathy

    # Need to make sure 'gs://' is in file path, not 'gs:/' as it is needed for loading
    assert "gs://" in str(config.output_data.filepath)


def test_yaml_load_on_premises():
    """Test that yaml loading works for 'on_premises.yaml'"""

    filename = os.path.join(
        os.path.dirname(nowcasting_dataset.__file__), "config", "on_premises.yaml"
    )

    config = load_yaml_configuration(filename)

    assert isinstance(config, Configuration)


def test_yaml_save():
    """
    Check a configuration can be saved to a yaml file
    """

    with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:

        filename = fp.name

        # check that temp file cant be loaded
        with pytest.raises(TypeError):
            _ = load_yaml_configuration(filename)

        # save default config to file
        save_yaml_configuration(Configuration(), filename)

        # check the file can be loaded
        _ = load_yaml_configuration(filename)


def test_extra_field():
    """
    Check a extra parameters in config causes error
    """

    configuration = Configuration()
    configuration_dict = configuration.dict()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError):
        _ = Configuration(**configuration_dict)


def test_incorrect_time_resolution():
    """
    Check a a time resolution not divisible by 5 causes an error
    """

    configuration = Configuration()
    configuration.input_data = configuration.input_data.set_all_to_defaults()
    configuration.input_data.satellite.time_resolution_minutes = 27
    with pytest.raises(Exception):
        _ = Configuration(**configuration.dict())


@pytest.mark.skip("Skiping test as CD does not have google credentials")
def test_save_to_gcs():
    """
    Check that configuration can be saved to gcs
    """
    save_yaml_configuration(
        configuration=Configuration(),
        filename="gs://solar-pv-nowcasting-data/temp_dir_for_unit_tests/test_config.yaml",
    )


@pytest.mark.skip("Skiping test as CD does not have google credentials")
def test_load_to_gcs():
    """
    Check that configuration can be loaded to gcs
    """
    config = load_yaml_configuration(
        filename="gs://solar-pv-nowcasting-data/prepared_ML_training_data/v-default/configuration.yaml"  # noqa: E501
    )

    assert isinstance(config, Configuration)


def test_config_get():
    """Test that git commit is working"""

    filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
    config = load_yaml_configuration(filename)

    config = set_git_commit(configuration=config)

    assert config.git is not None
    assert type(config.git.message) == str
    assert type(config.git.hash) == str
    assert type(config.git.committed_date) == datetime


def test_old_pv():
    """Test that old config pydantic works"""
    pv = PV(pv_metadata_filename="test_metadata.csv", pv_filename="test.csv")

    # validate model, this happens normal when doing a whole config
    pv = pv.model_validation(v=pv)
    assert len(pv.pv_files_groups) == 1
    assert pv.pv_files_groups[0].pv_metadata_filename == "test_metadata.csv"
    assert pv.pv_files_groups[0].pv_filename == "test.csv"
    assert pv.pv_filename is None
    assert pv.pv_metadata_filename is None


def test_old_pv_load():
    """Test that old pv yaml loading works"""

    filename = os.path.join(
        os.path.dirname(nowcasting_dataset.__file__), "../tests/config", "test_old_pv.yaml"
    )

    _ = load_yaml_configuration(filename)
