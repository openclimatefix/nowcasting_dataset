#!/usr/bin/env python3

"""Pre-prepares batches of data on Google Cloud Storage.

Usage:

First, manually create the GCS directories given by the constants
DST_TRAIN_PATH and DST_VALIDATION_PATH, and create the
LOCAL_TEMP_PATH.  Note that all files will be deleted from
LOCAL_TEMP_PATH when this script starts up.

Currently caluclating azimuth and elevation angles, takes about 15 mins for 2548 PV systems, for about 1 year

"""

from nowcasting_dataset.cloud.gcp import check_path_exists
from nowcasting_dataset.cloud.utils import upload_and_delete_local_files
from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path


import nowcasting_dataset
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.config.save import save_configuration_to_cloud

from nowcasting_dataset.dataset.datamodule import NowcastingDataModule
from nowcasting_dataset.dataset.batch import write_batch_locally
from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from nowcasting_dataset.utils import get_maximum_batch_id_from_gcs
from pathlib import Path
import torch
import os
import numpy as np

import neptune.new as neptune
from neptune.new.integrations.python_logger import NeptuneHandler

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s %(pathname)s %(lineno)d %(message)s")
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.INFO)

logging.getLogger("nowcasting_dataset.data_source").setLevel(logging.WARNING)

# load configuration, this can be changed to a different filename as needed
filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), "config", "gcp.yaml")
config = load_yaml_configuration(filename)

# set the gcs bucket name
BUCKET = Path(config.input_data.bucket)

# Solar PV data
PV_PATH = BUCKET / config.input_data.solar_pv_path
PV_DATA_FILENAME = PV_PATH / config.input_data.solar_pv_data_filename
PV_METADATA_FILENAME = PV_PATH / config.input_data.solar_pv_metadata_filename

SAT_FILENAME = BUCKET / config.input_data.satelite_filename

# Numerical weather predictions
NWP_BASE_PATH = BUCKET / config.input_data.npw_base_path

# GSP data
GSP_FILENAME = BUCKET / config.input_data.gsp_filename

DST_NETCDF4_PATH = config.output_data.filepath
DST_TRAIN_PATH = os.path.join(DST_NETCDF4_PATH, "train")
DST_VALIDATION_PATH = os.path.join(DST_NETCDF4_PATH, "validation")
DST_TEST_PATH = os.path.join(DST_NETCDF4_PATH, "test")
LOCAL_TEMP_PATH = Path("~/temp/").expanduser()

UPLOAD_EVERY_N_BATCHES = 16
CLOUD = "gcp"  # either gcp or aws

# Necessary to avoid "RuntimeError: receieved 0 items of ancdata".  See:
# https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/2
torch.multiprocessing.set_sharing_strategy("file_system")

np.random.seed(config.process.seed)


def get_data_module():
    num_workers = 4

    # get the batch id already made
    maximum_batch_id_train = get_maximum_batch_id_from_gcs(f"gs://{DST_TRAIN_PATH}")
    maximum_batch_id_validation = get_maximum_batch_id_from_gcs(f"gs://{DST_VALIDATION_PATH}")
    maximum_batch_id_test = get_maximum_batch_id_from_gcs(f"gs://{DST_TEST_PATH}")

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
        pv_power_filename=f"gs://{PV_DATA_FILENAME}",
        pv_metadata_filename=f"gs://{PV_METADATA_FILENAME}",
        sat_filename=f"gs://{SAT_FILENAME}",
        nwp_base_path=f"gs://{NWP_BASE_PATH}",
        gsp_filename=f"gs://{GSP_FILENAME}",
        pin_memory=True,  #: Passed to DataLoader.
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
    dataloader: torch.utils.data.DataLoader, dst_path: str
):
    _LOG.info("Getting first batch")
    for batch_i, batch in enumerate(dataloader):
        _LOG.info(f"Got batch {batch_i}")
        if len(batch) > 0:
            write_batch_locally(batch, batch_i)
        if batch_i > 0 and batch_i % UPLOAD_EVERY_N_BATCHES == 0:
            upload_and_delete_local_files(dst_path, LOCAL_TEMP_PATH, cloud=CLOUD)
    upload_and_delete_local_files(dst_path, LOCAL_TEMP_PATH, cloud=CLOUD)


def check_directories():
    if CLOUD == "gcp":
        for path in [DST_TRAIN_PATH, DST_VALIDATION_PATH, DST_TEST_PATH]:
            check_path_exists(path)


def main():

    run = neptune.init(
        project="OpenClimateFix/nowcasting-data",
        capture_stdout=True,
        capture_stderr=True,
        capture_hardware_metrics=False,
    )
    _LOG.addHandler(NeptuneHandler(run=run))

    check_directories()
    delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)
    datamodule = get_data_module()

    _LOG.info("Finished preparing datamodule!")
    _LOG.info("Preparing training data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.train_dataloader(), DST_TRAIN_PATH)
    _LOG.info("Preparing validation data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.val_dataloader(), DST_VALIDATION_PATH)
    _LOG.info("Preparing test data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.test_dataloader(), DST_TEST_PATH)
    _LOG.info("Done!")

    # save configuration to gcs
    save_configuration_to_cloud(configuration=config, cloud=CLOUD)


if __name__ == "__main__":
    main()
