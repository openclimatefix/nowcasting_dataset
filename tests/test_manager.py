"""Test Manager."""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nowcasting_dataset
from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.satellite.satellite_data_source import SatelliteDataSource
from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource
from nowcasting_dataset.manager import Manager

# TODO: Issue #322: Write test for Manager.configure_loggers()


def test_sample_spatial_and_temporal_locations_for_examples():  # noqa: D103
    local_path = Path(nowcasting_dataset.__file__).parent.parent

    gsp = GSPDataSource(
        zarr_path=f"{local_path}/tests/data/gsp/test.zarr",
        start_dt=datetime(2020, 4, 1),
        end_dt=datetime(2020, 4, 2),
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

    manager = Manager()
    manager.data_sources = {"gsp": gsp, "sun": sun}
    manager.data_source_which_defines_geospatial_locations = gsp
    t0_datetimes = manager.get_t0_datetimes_across_all_data_sources(freq="30T")
    locations = manager.sample_spatial_and_temporal_locations_for_examples(
        t0_datetimes=t0_datetimes, n_examples=10
    )

    assert locations.columns.to_list() == ["t0_datetime_UTC", "x_center_OSGB", "y_center_OSGB"]
    assert len(locations) == 10
    assert (t0_datetimes[0] <= locations["t0_datetime_UTC"]).all()
    assert (t0_datetimes[-1] >= locations["t0_datetime_UTC"]).all()


def test_load_yaml_configuration():  # noqa: D103
    manager = Manager()
    local_path = Path(nowcasting_dataset.__file__).parent.parent
    filename = local_path / "tests" / "config" / "test.yaml"
    manager.load_yaml_configuration(filename=filename)
    manager.initialise_data_sources()
    assert len(manager.data_sources) == 7
    assert isinstance(manager.data_source_which_defines_geospatial_locations, GSPDataSource)


def test_get_daylight_datetime_index():
    """Check that 'manager' gets the correct t0 datetime over nighttime"""
    filename = Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"

    sat = SatelliteDataSource(
        zarr_path=filename,
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

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

    correct_t0_datetimes = pd.date_range("2020-04-01 12:30", "2020-04-01 17:00", freq="5 min")
    np.testing.assert_array_equal(t0_datetimes, correct_t0_datetimes)


def test_batches():
    """Test that batches can be made"""
    filename = Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "sat_data.zarr"

    sat = SatelliteDataSource(
        zarr_path=filename,
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=24,
        meters_per_pixel=6000,
        channels=("IR_016",),
    )

    filename = (
        Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "hrv_sat_data.zarr"
    )
    hrvsat = SatelliteDataSource(
        zarr_path=filename,
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
        channels=("HRV",),
    )

    filename = (
        Path(nowcasting_dataset.__file__).parent.parent / "tests" / "data" / "gsp" / "test.zarr"
    )

    gsp = GSPDataSource(
        zarr_path=filename,
        start_dt=datetime(2020, 4, 1),
        end_dt=datetime(2020, 4, 2),
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels=64,
        meters_per_pixel=2000,
    )

    manager = Manager()

    # load config
    local_path = Path(nowcasting_dataset.__file__).parent.parent
    filename = local_path / "tests" / "config" / "test.yaml"
    manager.load_yaml_configuration(filename=filename)

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        # set local temp path, and dst path
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        # just set satellite as data source
        manager.data_sources = {"gsp": gsp, "sat": sat, "hrvsat": hrvsat}
        manager.data_source_which_defines_geospatial_locations = gsp

        # make file for locations
        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101

        # make batches
        manager.create_batches(overwrite_batches=True)

        assert os.path.exists(f"{dst_path}/train")
        assert os.path.exists(f"{dst_path}/train/gsp")
        assert os.path.exists(f"{dst_path}/train/gsp/000000.nc")
        assert os.path.exists(f"{dst_path}/train/sat/000000.nc")
        assert os.path.exists(f"{dst_path}/train/gsp/000001.nc")
        assert os.path.exists(f"{dst_path}/train/sat/000001.nc")
        assert os.path.exists(f"{dst_path}/train/hrvsat/000001.nc")
        assert os.path.exists(f"{dst_path}/train/hrvsat/000000.nc")


def test_save_config():
    """Test that configuration file is saved"""

    manager = Manager()

    # load config
    local_path = Path(nowcasting_dataset.__file__).parent.parent
    filename = local_path / "tests" / "config" / "test.yaml"
    manager.load_yaml_configuration(filename=filename)

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        # set local temp path, and dst path
        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        # save config
        manager.save_yaml_configuration()

        assert os.path.exists(f"{dst_path}/configuration.yaml")


def test_run():
    """Test to initialize data sources and get batches"""

    manager = Manager()
    local_path = Path(nowcasting_dataset.__file__).parent.parent
    filename = local_path / "tests" / "config" / "test.yaml"
    manager.load_yaml_configuration(filename=filename)
    manager.initialise_data_sources()

    with tempfile.TemporaryDirectory() as local_temp_path, tempfile.TemporaryDirectory() as dst_path:  # noqa 101

        manager.config.output_data.filepath = Path(dst_path)
        manager.local_temp_path = Path(local_temp_path)

        manager.create_files_specifying_spatial_and_temporal_locations_of_each_example_if_necessary()  # noqa 101
        manager.create_batches(overwrite_batches=True)
