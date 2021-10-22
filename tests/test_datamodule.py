import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import nowcasting_dataset
from nowcasting_dataset.config.load import load_yaml_configuration

from nowcasting_dataset.dataset import datamodule
from nowcasting_dataset.dataset.datamodule import NowcastingDataModule
from nowcasting_dataset.dataset.split.split import SplitMethod
from nowcasting_dataset.dataset.batch import Batch

logging.basicConfig(format="%(asctime)s %(levelname)s %(pathname)s %(lineno)d %(message)s")
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.DEBUG)


@pytest.fixture
def nowcasting_datamodule(sat_filename: Path):
    return datamodule.NowcastingDataModule(sat_filename=sat_filename)


def test_prepare_data(nowcasting_datamodule: datamodule.NowcastingDataModule):
    nowcasting_datamodule.prepare_data()


def test_get_daylight_datetime_index(
    nowcasting_datamodule: datamodule.NowcastingDataModule, use_cloud_data: bool
):
    # Check it throws RuntimeError if we try running
    # _get_daylight_datetime_index() before running prepare_data():
    with pytest.raises(RuntimeError):
        nowcasting_datamodule._get_t0_datetimes()
    nowcasting_datamodule.prepare_data()
    nowcasting_datamodule.t0_datetime_freq = "5T"
    t0_datetimes = nowcasting_datamodule._get_t0_datetimes()
    assert isinstance(t0_datetimes, pd.DatetimeIndex)
    if not use_cloud_data:
        # The testing sat_data.zarr has contiguous data from 12:05 to 18:00.
        # nowcasting_datamodule.history_minutes = 30
        # nowcasting_datamodule.forecast_minutes = 60
        # Daylight ends at 16:20.
        # So the expected t0_datetimes start at 12:35 (12:05 + 30 minutes)
        # and end at 15:20 (16:20 - 60 minutes)
        print(t0_datetimes)
        correct_t0_datetimes = pd.date_range("2019-01-01 12:35", "2019-01-01 15:20", freq="5 min")
        np.testing.assert_array_equal(t0_datetimes, correct_t0_datetimes)


def test_setup(nowcasting_datamodule: datamodule.NowcastingDataModule):
    # Check it throws RuntimeError if we try running
    # setup() before running prepare_data():
    with pytest.raises(RuntimeError):
        nowcasting_datamodule.setup()
    nowcasting_datamodule.prepare_data()
    nowcasting_datamodule.setup()


def _get_config_with_test_paths(config_filename: str):
    """Sets the base paths to point to the testing data in this repository."""
    local_path = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "../")

    # load configuration, this can be changed to a different filename as needed
    filename = os.path.join(local_path, "tests", "config", config_filename)
    config = load_yaml_configuration(filename)
    config.set_base_path(local_path)
    return config


@pytest.mark.parametrize("config_filename", ["test.yaml", "nwp_size_test.yaml"])
def test_data_module(config_filename):

    # load configuration, this can be changed to a different filename as needed
    config = _get_config_with_test_paths(config_filename)

    data_module = NowcastingDataModule(
        batch_size=config.process.batch_size,
        history_minutes=30,  #: Number of timesteps of history, not including t0.
        forecast_minutes=60,  #: Number of timesteps of forecast.
        satellite_image_size_pixels=config.input_data.satellite.satellite_image_size_pixels,
        nwp_image_size_pixels=config.input_data.nwp.nwp_image_size_pixels,
        nwp_channels=config.input_data.nwp.nwp_channels[0:1],
        sat_channels=config.input_data.satellite.sat_channels,  # reduced for test data
        pv_power_filename=config.input_data.pv.solar_pv_data_filename,
        pv_metadata_filename=config.input_data.pv.solar_pv_metadata_filename,
        sat_filename=config.input_data.satellite.satellite_zarr_path,
        nwp_base_path=config.input_data.nwp.nwp_zarr_path,
        gsp_filename=config.input_data.gsp.gsp_zarr_path,
        topographic_filename=config.input_data.topographic.topographic_filename,
        sun_filename=config.input_data.sun.sun_zarr_path,
        pin_memory=True,  #: Passed to DataLoader.
        num_workers=0,  #: Passed to DataLoader.
        prefetch_factor=8,  #: Passed to DataLoader.
        n_samples_per_timestep=16,  #: Passed to NowcastingDataset
        n_training_batches_per_epoch=200,  # Add pre-fetch factor!
        n_validation_batches_per_epoch=200,
        collate_fn=lambda x: x,
        skip_n_train_batches=0,
        skip_n_validation_batches=0,
        train_validation_percentage_split=50,
        pv_load_azimuth_and_elevation=True,
        split_method=SplitMethod.SAME,
    )

    _LOG.info("prepare_data()")
    data_module.prepare_data()
    _LOG.info("setup()")
    data_module.setup()

    data_generator = iter(data_module.train_dataset)
    batch = next(data_generator)

    assert batch.batch_size == config.process.batch_size
    assert type(batch) == Batch

    assert batch.satellite is not None
    assert batch.nwp is not None
    assert batch.sun is not None
    assert batch.topographic is not None
    assert batch.pv is not None
    assert batch.gsp is not None
    assert batch.metadata is not None
    assert batch.datetime is not None


def test_batch_to_batch_to_dataset():
    config = _get_config_with_test_paths("test.yaml")

    data_module = NowcastingDataModule(
        batch_size=config.process.batch_size,
        history_minutes=30,  #: Number of timesteps of history, not including t0.
        forecast_minutes=60,  #: Number of timesteps of forecast.
        satellite_image_size_pixels=config.input_data.satellite.satellite_image_size_pixels,
        nwp_image_size_pixels=config.input_data.nwp.nwp_image_size_pixels,
        nwp_channels=config.input_data.nwp.nwp_channels[0:1],
        sat_channels=config.input_data.satellite.sat_channels,  # reduced for test data
        pv_power_filename=config.input_data.pv.solar_pv_data_filename,
        pv_metadata_filename=config.input_data.pv.solar_pv_metadata_filename,
        sat_filename=config.input_data.satellite.satellite_zarr_path,
        nwp_base_path=config.input_data.nwp.nwp_zarr_path,
        gsp_filename=config.input_data.gsp.gsp_zarr_path,
        topographic_filename=config.input_data.topographic.topographic_filename,
        sun_filename=config.input_data.sun.sun_zarr_path,
        pin_memory=True,  #: Passed to DataLoader.
        num_workers=0,  #: Passed to DataLoader.
        prefetch_factor=8,  #: Passed to DataLoader.
        n_samples_per_timestep=16,  #: Passed to NowcastingDataset
        n_training_batches_per_epoch=200,  # Add pre-fetch factor!
        n_validation_batches_per_epoch=200,
        collate_fn=lambda x: x,
        skip_n_train_batches=0,
        skip_n_validation_batches=0,
        train_validation_percentage_split=50,
        pv_load_azimuth_and_elevation=False,
        split_method=SplitMethod.SAME,
    )

    _LOG.info("prepare_data()")
    data_module.prepare_data()
    _LOG.info("setup()")
    data_module.setup()

    data_generator = iter(data_module.train_dataset)
    batch = next(data_generator)

    assert type(batch) == Batch
