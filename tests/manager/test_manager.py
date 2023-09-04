"""Test Manager."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.dataset.split.split import SplitMethod
from nowcasting_dataset.manager.manager import Manager


def test_configure_loggers(test_configuration_filename):
    """Test to check loggers can be configured"""

    # set up
    manager = Manager()

    # make configuration
    manager.load_yaml_configuration(filename=test_configuration_filename)

    with tempfile.TemporaryDirectory() as dst_path:
        filepath = f"{dst_path}/extra_temp_folder"
        manager.config.output_data.filepath = Path(filepath)

        manager.configure_loggers(log_level="DEBUG")


def test_sample_spatial_and_temporal_locations_for_examples(gsp, sun):  # noqa: D103
    manager = Manager()
    manager.data_sources = {"gsp": gsp, "sun": sun}
    manager.data_source_which_defines_geospatial_locations = gsp
    t0_datetimes = manager.get_t0_datetimes_across_all_data_sources(freq="30T")
    locations = manager.sample_spatial_and_temporal_locations_for_examples(
        t0_datetimes=t0_datetimes, n_examples=10
    )

    assert len(locations) == 10
    # assert (t0_datetimes[0] <= locations["t0_datetime_UTC"]).all()
    # assert (t0_datetimes[-1] >= locations["t0_datetime_UTC"]).all()


def test_initialize_data_source_with_loggers(test_configuration_filename):
    """Check that initialize the data source and check it ends up in the logger"""

    # set up
    manager = Manager()

    # make configuration
    manager.load_yaml_configuration(filename=test_configuration_filename)

    with tempfile.TemporaryDirectory() as dst_path:
        manager.config.output_data.filepath = Path(dst_path)
        manager.configure_loggers(log_level="DEBUG")
        manager.initialize_data_sources()

        # check logs is appended to
        for log_file in ["gsp", "satellite", "hrvsatellite"]:
            filename = f"{dst_path}/{log_file}.log"
            assert os.path.exists(filename)
            with open(filename) as f:
                line_0 = next(f)
                assert "The configuration for" in line_0


def test_get_daylight_datetime_index(sat_filename, sat):
    """Check that 'manager' gets the correct t0 datetime over nighttime"""

    manager = Manager()
    manager.data_sources = {"sat": sat}
    manager.data_source_which_defines_geospatial_locations = sat
    t0_datetimes = manager.get_t0_datetimes_across_all_data_sources(freq="5T")

    # The testing sat_data.zarr has contiguous data from 12:00 to 18:00.
    # nowcasting_datamodule.history_minutes = 30
    # nowcasting_datamodule.forecast_minutes = 60
    # Daylight ends after 19:34.
    # So the expected t0_datetimes start at 12:30 (12:00 + 30 minutes)
    # and end at 17:00 (18:00 - 60 minutes)

    correct_t0_datetimes = pd.date_range("2020-04-01 12:30", "2020-04-01 13:00", freq="5 min")
    np.testing.assert_array_equal(t0_datetimes, correct_t0_datetimes)


def test_create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary(
    test_configuration_filename,
):
    """Test to create locations"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.config.process.n_train_batches = 10
    manager.config.process.n_validation_batches = 10
    manager.config.process.n_test_batches = 10
    manager.config.process.split_method = SplitMethod.SAME
    manager.initialize_data_sources()

    batch_size = manager.config.process.batch_size

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101

        train_file = f"{dst_path}/train/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"
        validation_file = (
            f"{dst_path}/validation/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"
        )
        test_file = f"{dst_path}/test/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"

        assert os.path.exists(train_file)
        assert os.path.exists(validation_file)
        assert os.path.exists(test_file)

        for file in train_file, validation_file:
            locations_df = pd.read_csv(file)
            assert len(locations_df) == 10 * batch_size

        locations_df = pd.read_csv(test_file)
        assert len(locations_df) == batch_size * manager.config.process.n_test_batches


def test_create_files_specifying_spatial_and_temporal_locations_of_each_example_twice(
    test_configuration_filename,
):
    """Test to create files for locations, twice, and check no error"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.config.process.n_train_batches = 10
    manager.config.process.n_validation_batches = 10
    manager.config.process.n_test_batches = 10
    manager.config.process.split_method = SplitMethod.SAME
    manager.initialize_data_sources()

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101
        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101


def test_error_create_files_specifying_spatial_and_temporal_locations_of_each_example(
    test_configuration_filename,
):
    """Test to initialize data sources and get batches"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.config.process.n_train_batches = 0
    manager.config.process.n_validation_batches = 0
    manager.config.process.n_test_batches = 0
    manager.config.process.split_method = SplitMethod.SAME
    manager.initialize_data_sources()

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)
        with pytest.raises(RuntimeError):
            manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101


def test_batches(test_configuration_filename_no_hrv, sat, gsp):
    """Test that batches can be made"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename_no_hrv)

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101
        # set local temp path, and dst path
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        # set up loggers
        manager.configure_loggers(
            log_level="DEBUG", names_of_selected_data_sources=["gsp", "satellite"]
        )

        # Set data sources
        manager.data_sources = {"gsp": gsp, "satellite": sat}
        manager.data_source_which_defines_geospatial_locations = gsp

        # make file for locations
        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101

        # make batches
        manager.create_batches(overwrite_batches=True)

        assert os.path.exists(f"{dst_path}/train")
        assert os.path.exists(f"{dst_path}/train/gsp")
        assert os.path.exists(f"{dst_path}/train/gsp/000000.nc")
        assert os.path.exists(f"{dst_path}/train/gsp/000001.nc")
        assert os.path.exists(f"{dst_path}/train/satellite/000000.nc")
        assert os.path.exists(f"{dst_path}/train/satellite/000001.nc")

        # check logs is appended to
        for log_file in ["combined", "gsp", "satellite"]:
            filename = f"{dst_path}/{log_file}.log"
            assert os.path.exists(filename)
            with open(filename) as f:
                num_lines = sum(1 for line in f)
                assert num_lines > 0, f"Log {filename} is empty"


def test_save_config(test_configuration_filename):
    """Test that configuration file is saved"""

    manager = Manager()

    # load config
    manager.load_yaml_configuration(filename=test_configuration_filename)

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101
        # set local temp path, and dst path
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        # save config
        manager.save_yaml_configuration()

        assert os.path.exists(f"{dst_path}/configuration.yaml")


def test_run_error(test_configuration_filename):
    """Test to initialize data sources and get batches"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    manager.config.input_data.satellite.forecast_minutes = 17

    with pytest.raises(Exception):
        manager.initialize_data_sources()


def test_run_error_data_source_which_defines_geospatial_locations(test_configuration_filename):
    """Test to initialize data sources and get batches"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    manager.config.input_data.data_source_which_defines_geospatial_locations = "test"

    with pytest.raises(Exception):
        manager.initialize_data_sources()


def test_run(test_configuration_filename_no_hrv):
    """Test to initialize data sources and get batches"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename_no_hrv)
    manager.initialize_data_sources()

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101
        manager.create_batches(overwrite_batches=True)


def test_run_overwrite_batches_false(test_configuration_filename_no_hrv):
    """Test to initialize data sources and get batches, but dont overwrite"""

    manager = Manager()
    manager.load_yaml_configuration(filename=test_configuration_filename_no_hrv)
    manager.initialize_data_sources()

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101
        manager.create_batches(overwrite_batches=False)
        manager.create_batches(overwrite_batches=False)
