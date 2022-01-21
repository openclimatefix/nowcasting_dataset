"""Test Manager."""

from pathlib import Path

from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.manager.base import ManagerBase


def test_load_yaml_configuration(test_configuration_filename):  # noqa: D103
    manager = ManagerBase()
    manager.load_yaml_configuration(filename=test_configuration_filename)


def test_create_initialise_data_sources(
    test_configuration_filename,
):
    """Test to create locations"""

    manager = ManagerBase()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.initialise_data_sources()
    assert len(manager.data_sources) == 8
    assert isinstance(manager.data_source_which_defines_geospatial_locations, GSPDataSource)
    assert isinstance(manager.config.process.local_temp_path, Path)
