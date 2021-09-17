import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import pytest

import nowcasting_dataset
from nowcasting_dataset.dataset import datamodule
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.dataset.datamodule import NowcastingDataModule
from nowcasting_dataset.dataset.example import validate_example
from nowcasting_dataset.dataset.batch import batch_to_dataset
from nowcasting_dataset.dataset.example import Example

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
        nowcasting_datamodule._get_datetimes()
    nowcasting_datamodule.prepare_data()
    datetimes = nowcasting_datamodule._get_datetimes(
        interpolate_for_30_minute_data=False, adjust_for_sequence_length=False
    )
    assert isinstance(datetimes, pd.DatetimeIndex)
    if not use_cloud_data:
        correct_datetimes = pd.date_range("2019-01-01 12:05", "2019-01-01 16:20", freq="5 min")
        np.testing.assert_array_equal(datetimes, correct_datetimes)


def test_setup(nowcasting_datamodule: datamodule.NowcastingDataModule):
    # Check it throws RuntimeError if we try running
    # setup() before running prepare_data():
    with pytest.raises(RuntimeError):
        nowcasting_datamodule.setup()
    nowcasting_datamodule.prepare_data()
    nowcasting_datamodule.setup()


@pytest.mark.parametrize("config_name", ["test.yaml", "nwp_size_test.yaml"])
def test_data_module(config_name):

    local_path = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "../")

    # load configuration, this can be changed to a different filename as needed
    filename = os.path.join(local_path, "tests", "config", config_name)
    config = load_yaml_configuration(filename)

    data_module = NowcastingDataModule(
        batch_size=config.process.batch_size,
        history_minutes=30,  #: Number of timesteps of history, not including t0.
        forecast_minutes=60,  #: Number of timesteps of forecast.
        satellite_image_size_pixels=config.process.satellite_image_size_pixels,
        nwp_image_size_pixels=config.process.nwp_image_size_pixels,
        nwp_channels=config.process.nwp_channels,
        sat_channels=config.process.sat_channels,  # reduced for test data
        pv_power_filename=config.input_data.solar_pv_data_filename,
        pv_metadata_filename=config.input_data.solar_pv_metadata_filename,
        sat_filename=config.input_data.satelite_filename,
        nwp_base_path=config.input_data.npw_base_path,
        gsp_filename=config.input_data.gsp_filename,
        pin_memory=True,  #: Passed to DataLoader.
        num_workers=0,  #: Passed to DataLoader.
        prefetch_factor=8,  #: Passed to DataLoader.
        n_samples_per_timestep=16,  #: Passed to NowcastingDataset
        n_training_batches_per_epoch=200,  # Add pre-fetch factor!
        n_validation_batches_per_epoch=200,
        collate_fn=lambda x: x,
        convert_to_numpy=False,  #: Leave data as Pandas / Xarray for pre-preparing.
        normalise_sat=False,
        skip_n_train_batches=0,
        skip_n_validation_batches=0,
        train_validation_percentage_split=50,
        pv_load_azimuth_and_elevation=True,
    )

    _LOG.info("prepare_data()")
    data_module.prepare_data()
    _LOG.info("setup()")
    data_module.setup()

    data_generator = iter(data_module.train_dataset)
    batch = next(data_generator)

    assert len(batch) == config.process.batch_size

    for key in list(Example.__annotations__.keys()):
        assert key in batch[0].keys()

    seq_len_30_minutes = 4  # 30 minutes history, 60 minutes in the future plus now, is 4)
    seq_len_5_minutes = (
        19  # 30 minutes history (=6), 60 minutes in the future (=12) plus now, is 19)
    )

    for x in batch:
        validate_example(
            data=x,
            n_nwp_channels=len(config.process.nwp_channels),
            nwp_image_size=config.process.nwp_image_size_pixels,  # TODO why is this zero
            n_sat_channels=len(config.process.sat_channels),
            sat_image_size=config.process.satellite_image_size_pixels,
            seq_len_30_minutes=seq_len_30_minutes,
            seq_len_5_minutes=seq_len_5_minutes,
        )


def test_batch_to_batch_to_dataset():

    local_path = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "../")

    # load configuration, this can be changed to a different filename as needed
    filename = os.path.join(local_path, "tests", "config", "test.yaml")
    config = load_yaml_configuration(filename)

    data_module = NowcastingDataModule(
        batch_size=config.process.batch_size,
        history_minutes=30,  #: Number of timesteps of history, not including t0.
        forecast_minutes=60,  #: Number of timesteps of forecast.
        satellite_image_size_pixels=config.process.satellite_image_size_pixels,
        nwp_image_size_pixels=config.process.nwp_image_size_pixels,
        nwp_channels=config.process.nwp_channels,
        sat_channels=config.process.sat_channels,  # reduced for test data
        pv_power_filename=config.input_data.solar_pv_data_filename,
        pv_metadata_filename=config.input_data.solar_pv_metadata_filename,
        sat_filename=config.input_data.satelite_filename,
        nwp_base_path=config.input_data.npw_base_path,
        gsp_filename=config.input_data.gsp_filename,
        pin_memory=True,  #: Passed to DataLoader.
        num_workers=0,  #: Passed to DataLoader.
        prefetch_factor=8,  #: Passed to DataLoader.
        n_samples_per_timestep=16,  #: Passed to NowcastingDataset
        n_training_batches_per_epoch=200,  # Add pre-fetch factor!
        n_validation_batches_per_epoch=200,
        collate_fn=lambda x: x,
        convert_to_numpy=False,  #: Leave data as Pandas / Xarray for pre-preparing.
        normalise_sat=False,
        skip_n_train_batches=0,
        skip_n_validation_batches=0,
        train_validation_percentage_split=50,
        pv_load_azimuth_and_elevation=False,
    )

    _LOG.info("prepare_data()")
    data_module.prepare_data()
    _LOG.info("setup()")
    data_module.setup()

    data_generator = iter(data_module.train_dataset)
    batch = next(data_generator)

    batch_xr = batch_to_dataset(batch=batch)
    assert type(batch_xr) == xr.Dataset
