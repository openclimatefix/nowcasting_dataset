#!/usr/bin/env python3

"""Pre-prepares batches of data on Google Cloud Storage.

Usage:

First, manually create the GCS directories given by the constants
DST_TRAIN_PATH and DST_VALIDATION_PATH, and create the
LOCAL_TEMP_PATH.  Note that all files will be deleted from
LOCAL_TEMP_PATH when this script starts up.

Currently caluclating azimuth and elevation angles, takes about 15 mins for 2548 PV systems, for about 1 year

"""

from nowcasting_dataset.cloud import utils
from nowcasting_dataset.cloud import local

import nowcasting_dataset
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.config.save import save_yaml_configuration
from nowcasting_dataset.config.model import set_git_commit
from nowcasting_dataset.cloud.local import check_path_exists

from nowcasting_dataset.dataset.datamodule import NowcastingDataModule
from nowcasting_dataset.dataset.batch import write_batch_locally
from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from pathy import Pathy
from pathlib import Path
import fsspec
import torch
import os
import numpy as np
from typing import Union

import neptune.new as neptune
from neptune.new.integrations.python_logger import NeptuneHandler

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(pathname)s %(lineno)d %(message)s")
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.INFO)

logging.getLogger("nowcasting_dataset.data_source").setLevel(logging.WARNING)

ENABLE_NEPTUNE_LOGGING = False

# load configuration, this can be changed to a different filename as needed.
# TODO: Pass this in as a command-line argument.
# See https://github.com/openclimatefix/nowcasting_dataset/issues/171
filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "on_premises.yaml")
config = load_yaml_configuration(filename)
config = set_git_commit(config)

# Solar PV data
PV_DATA_FILENAME = config.input_data.solar_pv_data_filename
PV_METADATA_FILENAME = config.input_data.solar_pv_metadata_filename

# Satellite data
SAT_ZARR_PATH = config.input_data.satellite_zarr_path

# Numerical weather predictions
NWP_ZARR_PATH = config.input_data.nwp_zarr_path

# GSP data
GSP_ZARR_PATH = config.input_data.gsp_zarr_path

# Topographic data
TOPO_TIFF_PATH = config.input_data.topographic_filename

# Paths for output data.
DST_NETCDF4_PATH = Pathy(config.output_data.filepath)
DST_TRAIN_PATH = DST_NETCDF4_PATH / "train"
DST_VALIDATION_PATH = DST_NETCDF4_PATH / "validation"
DST_TEST_PATH = DST_NETCDF4_PATH / "test"
LOCAL_TEMP_PATH = Path(config.process.local_temp_path).expanduser()

UPLOAD_EVERY_N_BATCHES = config.process.upload_every_n_batches
CLOUD = config.general.cloud  # either gcp or aws

# Necessary to avoid "RuntimeError: receieved 0 items of ancdata".  See:
# https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/2
torch.multiprocessing.set_sharing_strategy("file_system")

np.random.seed(config.process.seed)


def check_directories_exist():
    _LOG.info("Checking if all paths exist...")
    for path in [
        PV_DATA_FILENAME,
        PV_METADATA_FILENAME,
        SAT_ZARR_PATH,
        NWP_ZARR_PATH,
        GSP_ZARR_PATH,
        TOPO_TIFF_PATH,
        DST_TRAIN_PATH,
        DST_VALIDATION_PATH,
        DST_TEST_PATH,
    ]:
        check_path_exists(path)

    if UPLOAD_EVERY_N_BATCHES > 0:
        check_path_exists(LOCAL_TEMP_PATH)
    _LOG.info("Success!  All paths exist!")


def get_data_module():
    num_workers = 4

    # get the batch id already made
    maximum_batch_id_train = utils.get_maximum_batch_id(DST_TRAIN_PATH)
    maximum_batch_id_validation = utils.get_maximum_batch_id(DST_VALIDATION_PATH)
    maximum_batch_id_test = utils.get_maximum_batch_id(DST_TEST_PATH)

    if maximum_batch_id_train is None:
        maximum_batch_id_train = 0

    if maximum_batch_id_validation is None:
        maximum_batch_id_validation = 0

    if maximum_batch_id_test is None:
        maximum_batch_id_test = 0

    data_module = NowcastingDataModule(
        batch_size=config.process.batch_size,
        history_minutes=config.process.history_minutes,  #: Number of minutes of history, not including t0.
        forecast_minutes=config.process.forecast_minutes,  #: Number of minutes of forecast.
        satellite_image_size_pixels=config.process.satellite_image_size_pixels,
        nwp_image_size_pixels=config.process.nwp_image_size_pixels,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
        pv_power_filename=PV_DATA_FILENAME,
        pv_metadata_filename=PV_METADATA_FILENAME,
        sat_filename=SAT_ZARR_PATH,
        nwp_base_path=NWP_ZARR_PATH,
        gsp_filename=GSP_ZARR_PATH,
        topographic_filename=TOPO_TIFF_PATH,
        pin_memory=False,  #: Passed to DataLoader.
        num_workers=num_workers,  #: Passed to DataLoader.
        prefetch_factor=8,  #: Passed to DataLoader.
        n_samples_per_timestep=8,  #: Passed to NowcastingDataset
        n_training_batches_per_epoch=25_008,  # Add pre-fetch factor!
        n_validation_batches_per_epoch=1_008,
        n_test_batches_per_epoch=1_008,
        collate_fn=lambda x: x,
        convert_to_numpy=False,  #: Leave data as Pandas / Xarray for pre-preparing.
        normalise_sat=False,
        skip_n_train_batches=maximum_batch_id_train // num_workers,
        skip_n_validation_batches=maximum_batch_id_validation // num_workers,
        skip_n_test_batches=maximum_batch_id_test // num_workers,
        seed=config.process.seed,
    )
    _LOG.info("prepare_data()")
    data_module.prepare_data()
    _LOG.info("setup()")
    data_module.setup()
    return data_module


def iterate_over_dataloader_and_write_to_disk(
    dataloader: torch.utils.data.DataLoader, dst_path: Union[Pathy, Path]
):
    _LOG.info("Getting first batch")
    if UPLOAD_EVERY_N_BATCHES > 0:
        local_output_path = LOCAL_TEMP_PATH
    else:
        local_output_path = dst_path

    for batch_i, batch in enumerate(dataloader):
        _LOG.info(f"Got batch {batch_i}")
        if len(batch) > 0:
            write_batch_locally(batch, batch_i, local_output_path)
        if UPLOAD_EVERY_N_BATCHES > 0 and batch_i > 0 and batch_i % UPLOAD_EVERY_N_BATCHES == 0:
            utils.upload_and_delete_local_files(dst_path, LOCAL_TEMP_PATH, cloud=CLOUD)

    # Make sure we upload the last few batches, if necessary.
    if UPLOAD_EVERY_N_BATCHES > 0:
        utils.upload_and_delete_local_files(dst_path, LOCAL_TEMP_PATH, cloud=CLOUD)


def main():
    if ENABLE_NEPTUNE_LOGGING:
        run = neptune.init(
            project="OpenClimateFix/nowcasting-data",
            capture_stdout=True,
            capture_stderr=True,
            capture_hardware_metrics=False,
        )
        _LOG.addHandler(NeptuneHandler(run=run))

    check_directories_exist()
    if UPLOAD_EVERY_N_BATCHES > 0:
        local.delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)

    datamodule = get_data_module()

    _LOG.info("Finished preparing datamodule!")
    _LOG.info("Preparing training data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.train_dataloader(), DST_TRAIN_PATH)
    _LOG.info("Preparing validation data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.val_dataloader(), DST_VALIDATION_PATH)
    _LOG.info("Preparing test data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.test_dataloader(), DST_TEST_PATH)
    _LOG.info("Done!")

    save_yaml_configuration(config)


if __name__ == "__main__":
    main()
