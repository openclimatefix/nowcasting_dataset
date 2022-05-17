"""Test Manager Live."""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.dataset.split.split import SplitMethod
from nowcasting_dataset.manager.manager_live import ManagerLive


def test_sample_spatial_and_temporal_locations_for_examples(
    test_configuration_filename, gsp, sun
):  # noqa: D103

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.data_sources = {"gsp": gsp, "sun": sun}
    manager.data_source_which_defines_geospatial_locations = gsp
    # t0_datetimes = manager.get_t0_datetimes_across_all_data_sources(freq="30T")
    locations = manager.sample_spatial_and_temporal_locations_for_examples(
        t0_datetime=datetime(2020, 4, 1, 12)
    )

    assert len(locations) == 20


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
    manager.initialize_data_sources()
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


def test_create_files_locations_of_each_example_reduced(
    test_configuration_filename,
):
    """Test to create locations, for a reduced number of n_gsps"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.config.process.batch_size = 5
    manager.config.process.split_method = SplitMethod.SAME
    manager.initialize_data_sources()
    t0_datetime = datetime(2021, 4, 1)

    batch_size = manager.config.process.batch_size

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
            t0_datetime=t0_datetime, n_gsps=7
        )  # noqa 101

        live_file = f"{dst_path}/live/{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}"

        assert os.path.exists(live_file)
        locations_df = pd.read_csv(live_file)
        # we've asked for 7 examples, but the batchsize is 5, so we get 10 examples,
        # the last 3 being filled in
        assert len(locations_df) == batch_size * 2
        # check last 3 are filled in from the first one
        assert locations_df.iloc[-3].id == 1
        assert locations_df.iloc[-2].id == 1
        assert locations_df.iloc[-1].id == 1


def test_batches(test_configuration_filename, sat, gsp):
    """Test that batches can be made"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        # set local temp path, and dst path
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        # Set data sources
        manager.data_sources = {"gsp": gsp, "satellite": sat}
        manager.data_source_which_defines_geospatial_locations = gsp

        # make file for locations
        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
            t0_datetime=datetime(2020, 4, 1, 13)
        )  # noqa 101

        # make batches
        manager.create_batches()

        assert os.path.exists(f"{dst_path}/live")
        assert os.path.exists(f"{dst_path}/live/gsp")
        assert os.path.exists(f"{dst_path}/live/gsp/000000.nc")
        assert os.path.exists(f"{dst_path}/live/satellite/000000.nc")


def test_batches_not_async(test_configuration_filename, sat, gsp):
    """Test that batches can be made"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        # set local temp path, and dst path
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        # Set data sources
        manager.data_sources = {"gsp": gsp, "satellite": sat}
        manager.data_source_which_defines_geospatial_locations = gsp

        # make file for locations
        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
            t0_datetime=datetime(2020, 4, 1, 13)
        )  # noqa 101

        # make batches
        manager.create_batches(use_async=False)

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
        manager.initialize_data_sources()


def test_run_error_data_source_which_defines_geospatial_locations(test_configuration_filename):
    """Test to initialize data sources and get batches"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)

    manager.config.input_data.data_source_which_defines_geospatial_locations = "test"

    with pytest.raises(Exception):
        manager.initialize_data_sources()


def test_run(test_configuration_filename):
    """Test to initialize data sources and get batches"""

    manager = ManagerLive()
    manager.load_yaml_configuration(filename=test_configuration_filename)
    manager.initialize_data_sources(names_of_selected_data_sources=["gsp", "nwp"])

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example(
            t0_datetime=datetime(2020, 4, 1, 15)
        )  # noqa 101
        manager.create_batches()

        with tempfile.TemporaryDirectory() as local_temp_path_save:
            manager.save_batch(batch_idx=0, path=local_temp_path_save)
            assert os.path.exists(f"{local_temp_path_save}/gsp/000000.nc")
