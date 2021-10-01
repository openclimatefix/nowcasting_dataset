import os
import tempfile
from datetime import datetime

import boto3
import moto
import pytest

import nowcasting_dataset
from nowcasting_dataset.config.load import load_yaml_configuration, load_configuration_from_gcs
from nowcasting_dataset.config.model import Configuration, set_git_commit
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


@pytest.mark.skip("Skiping test as CD does not have google credentials")
def test_save_to_gcs():
    """
    Check that configuration can be saved to gcs
    """
    save_yaml_configuration(
        configuration=Configuration(),
        filename="gs://solar-pv-nowcasting-data/temp_dir_for_unit_tests/test_config.yaml",
    )


@pytest.mark.skip(
    "Skipping due to unresolved bug in moto: https://github.com/aio-libs/aiobotocore/issues/755"
)
# See https://github.com/openclimatefix/nowcasting_dataset/issues/164
# for more details and some possible work-arounds.
@moto.mock_s3()
def test_save_to_aws():
    """
    Check that configuration can be saved to gcs
    """

    bucket_name = "test_bucket"

    s3_client = boto3.client("s3")
    s3_client.create_bucket(Bucket=bucket_name)

    save_yaml_configuration(
        configuration=Configuration(), filename=f"s3://{bucket_name}/test_config.yaml"
    )


@pytest.mark.skip("Skiping test as CD does not have google credentials")
def test_load_to_gcs():
    """
    Check that configuration can be loaded to gcs
    """
    config = load_configuration_from_gcs(gcp_dir="prepared_ML_training_data/v-default")

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
