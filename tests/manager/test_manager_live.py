"""Test Manager."""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import nowcasting_dataset
from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SatelliteDataSource
from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource
from nowcasting_dataset.dataset.split.split import SplitMethod
from nowcasting_dataset.manager.manager_live import ManagerLive


def test_sample_spatial_and_temporal_locations_for_examples(
    test_configuration_filename,
):  # noqa: D103
    local_path = Path(nowcasting_dataset.__file__).parent.parent

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_datetime=datetime(2020, 4, 1),
        end_datetime=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    sun = SunDataSource(
        zarr_path=f"{local_path}/tests/data/sun/test.zarr",
        history_minutes=30,
        forecast_minutes=60,
    )

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.data_sources = {"gsp": gsp, "sun": sun}
    manager.data_source_which_defines_geospatial_locations = gsp
    # t0_datetimes = manager.get_t0_datetimes_across_all_data_sources(freq="30T")
    locations = manager.sample_spatial_and_temporal_locations_for_examples(
        t0_datetime=datetime(2020, 4, 1, 12)
    )

    assert locations.columns.to_list() == ["t0_datetime_UTC", "x_center_OSGB", "y_center_OSGB"]
    assert len(locations) == 32


def test_load_yaml_configuration(test_configuration_filename):  # noqa: D103
    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    manager.initialise_data_sources()
    assert len(manager.data_sources) == 8
    assert isinstance(manager.data_source_which_defines_geospatial_locations, GSPDataSource)
    assert isinstance(manager.config.process.local_temp_path, Path)


def test_create_files_specifying_spatial_and_temporal_locations_of_each_example(
    test_configuration_filename,
):
    """Test to create locations"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.config.process.n_train_batches = 10
    manager.config.process.n_validation_batches = 10
    manager.config.process.n_test_batches = 10
    manager.config.process.split_method = SplitMethod.SAME
    manager.initialise_data_sources()
    t0_datetime = datetime(2021, 4, 1)

    batch_size = manager.config.process.batch_size

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
            t0_datetime=t0_datetime
        )  # noqa 101

        live_file = f"{dst_path}/live/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"

        assert os.path.exists(live_file)
        locations_df = pd.read_csv(live_file)
        assert len(locations_df) == batch_size

        locations_df = pd.read_csv(live_file)
        assert len(locations_df) == batch_size


def test_batches(test_configuration_filename):
    """Test that batches can be made"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        # set local temp path, and dst path
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        # just set satellite as data source
        filename = (
            Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"
        )

        sat = SatelliteDataSource(
            zarr_path=filename,
            history_minutes=30,
            forecast_minutes=60,
            image_size_pixels=24,
            meters_per_pixel=6000,
            channels=("IR_016",),
        )

        filename = (
            Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
        )

        gsp = GSPDataSource(
            zarr_path=filename,
            start_datetime=datetime(2020, 4, 1),
            end_datetime=datetime(2020, 4, 2),
            history_minutes=30,
            forecast_minutes=60,
            image_size_pixels=64,
            meters_per_pixel=2000,
        )

        # Set data sources
        manager.data_sources = {"gsp": gsp, "satellite": sat}
        manager.data_source_which_defines_geospatial_locations = gsp

        # make file for locations
        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
            t0_datetime=datetime(2020, 4, 1, 15)
        )  # noqa 101

        # make batches
        manager.create_batches(overwrite_batches=True)

        assert os.path.exists(f"{dst_path}/live")
        assert os.path.exists(f"{dst_path}/live/gsp")
        assert os.path.exists(f"{dst_path}/live/gsp/000000.nc")
        assert os.path.exists(f"{dst_path}/live/satellite/000000.nc")


def test_run_error(test_configuration_filename):
    """Test to initialize data sources and get batches"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    manager.config.input_data.satellite.forecast_minutes = 17

    with pytest.raises(Exception):
        manager.initialise_data_sources()


def test_run_error_data_source_which_defines_geospatial_locations(test_configuration_filename):
    """Test to initialize data sources and get batches"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    manager.config.input_data.data_source_which_defines_geospatial_locations = "test"

    with pytest.raises(Exception):
        manager.initialise_data_sources()


def test_run(test_configuration_filename):
    """Test to initialize data sources and get batches"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.initialise_data_sources(names_of_selected_data_sources=["gsp", "nwp"])

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
            t0_datetime=datetime(2020, 4, 1, 15)
        )  # noqa 101
        manager.create_batches(overwrite_batches=True)
