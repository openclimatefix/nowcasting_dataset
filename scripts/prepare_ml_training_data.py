#!/usr/bin/env python3

"""Pre-prepares batches of data on Google Cloud Storage.

Usage:

First, manually create the GCS directories given by the constants
DST_TRAIN_PATH and DST_VALIDATION_PATH, and create the
LOCAL_TEMP_PATH.  Note that all files will be deleted from
LOCAL_TEMP_PATH when this script starts up.
"""

from nowcasting_dataset.cloud.gcp import check_path_exists
from nowcasting_dataset.cloud.utils import upload_and_delete_local_files
from nowcasting_dataset.cloud.local import delete_all_files_in_temp_path


import nowcasting_dataset
from nowcasting_dataset.config.load import load_yaml_configuration
from nowcasting_dataset.config.save import save_configuration_to_cloud

from nowcasting_dataset.datamodule import NowcastingDataModule
from nowcasting_dataset.example import Example, DATETIME_FEATURE_NAMES
from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
from pathlib import Path
import numpy as np
import xarray as xr
import torch
import os
from typing import List, Optional

from nowcasting_dataset.utils import get_netcdf_filename
import neptune.new as neptune
from neptune.new.integrations.python_logger import NeptuneHandler

import logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(pathname)s %(lineno)d %(message)s')
_LOG = logging.getLogger("nowcasting_dataset")
_LOG.setLevel(logging.DEBUG)

# load configuration, this can be changed to a different filename as needed
filename = os.path.join(os.path.dirname(nowcasting_dataset.__file__), 'config', 'example.yaml')
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
DST_TRAIN_PATH = os.path.join(DST_NETCDF4_PATH, 'train')
DST_VALIDATION_PATH = os.path.join(DST_NETCDF4_PATH, 'validation')
LOCAL_TEMP_PATH = Path('~/temp/').expanduser()

UPLOAD_EVERY_N_BATCHES = 16
CLOUD = "aws" # either gcp or aws

# Necessary to avoid "RuntimeError: receieved 0 items of ancdata".  See:
# https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/2
torch.multiprocessing.set_sharing_strategy("file_system")


def get_data_module():
    data_module = NowcastingDataModule(
        batch_size=config.process.batch_size,
        history_len=config.process.history_length,  #: Number of timesteps of history, not including t0.
        forecast_len=config.process.forecast_length,  #: Number of timesteps of forecast.
        image_size_pixels=config.process.image_size_pixels,
        nwp_channels=NWP_VARIABLE_NAMES,
        sat_channels=SAT_VARIABLE_NAMES,
        pv_power_filename=PV_DATA_FILENAME,
        pv_metadata_filename=f"gs://{PV_METADATA_FILENAME}",
        sat_filename=f"gs://{SAT_FILENAME}",
        nwp_base_path=f"gs://{NWP_BASE_PATH}",
        gsp_filename=f"gs://{GSP_FILENAME}",
        pin_memory=True,  #: Passed to DataLoader.
        num_workers=6,  #: Passed to DataLoader.
        prefetch_factor=8,  #: Passed to DataLoader.
        n_samples_per_timestep=8,  #: Passed to NowcastingDataset
        n_training_batches_per_epoch=25_008,  # Add pre-fetch factor!
        n_validation_batches_per_epoch=1_008,
        collate_fn=lambda x: x,
        convert_to_numpy=False,  #: Leave data as Pandas / Xarray for pre-preparing.
        normalise_sat=False,
        skip_n_train_batches=0,
        skip_n_validation_batches=0,
    )
    _LOG.info("prepare_data()")
    data_module.prepare_data()
    _LOG.info("setup()")
    data_module.setup()
    return data_module


def coord_to_range(da: xr.DataArray, dim: str, prefix: Optional[str], dtype=np.int32) -> xr.DataArray:
    # TODO: Actually, I think this is over-complicated?  I think we can
    # just strip off the 'coord' from the dimension.
    coord = da[dim]
    da[dim] = np.arange(len(coord), dtype=dtype)
    if prefix is not None:
        da[f"{prefix}_{dim}_coords"] = xr.DataArray(coord, coords=[da[dim]], dims=[dim])
    return da


def batch_to_dataset(batch: List[Example]) -> xr.Dataset:
    """Concat all the individual fields in an Example into a single Dataset.

    Args:
      batch: List of Example objects, which together constitute a single batch.
    """
    datasets = []
    for i, example in enumerate(batch):
        try:
            individual_datasets = []
            example_dim = {"example": np.array([i], dtype=np.int32)}
            for name in ["sat_data", "nwp"]:
                ds = example[name].to_dataset(name=name)
                short_name = name.replace("_data", "")
                if name == "nwp":
                    ds = ds.rename({"target_time": "time"})
                for dim in ["time", "x", "y"]:
                    ds = coord_to_range(ds, dim, prefix=short_name)
                ds = ds.rename(
                    {
                        "variable": f"{short_name}_variable",
                        "x": f"{short_name}_x",
                        "y": f"{short_name}_y",
                    }
                )
                individual_datasets.append(ds)

            # Datetime features
            for name in DATETIME_FEATURE_NAMES:
                ds = example[name].rename(name).to_xarray().to_dataset().rename({"index": "time"})
                ds = coord_to_range(ds, "time", prefix=None)
                individual_datasets.append(ds)

            # PV
            pv_yield = xr.DataArray(example["pv_yield"], dims=["time", "pv_system"])
            pv_yield = pv_yield.to_dataset(name="pv_yield")
            n_pv_systems = len(example["pv_system_id"])
            # This will expand all dataarrays to have an 'example' dim.
            # 0D
            for name in ["x_meters_center", "y_meters_center"]:
                try:
                    pv_yield[name] = xr.DataArray([example[name]], coords=example_dim, dims=["example"])
                except Exception as e:
                    _LOG.error(f'Could not make pv_yield data for {name} with example_dim={example_dim}')
                    if name not in example.keys():
                        _LOG.error(f'{name} not in data keys: {example.keys()}')
                    _LOG.error(e)
                    raise Exception

            # 1D
            for name in ["pv_system_id", "pv_system_row_number", "pv_system_x_coords", "pv_system_y_coords"]:
                pv_yield[name] = xr.DataArray(
                    example[name][None, :],
                    coords=example_dim | {"pv_system": np.arange(n_pv_systems, dtype=np.int32)},
                    dims=["example", "pv_system"],
                )

            individual_datasets.append(pv_yield)

            # Merge
            merged_ds = xr.merge(individual_datasets)
            datasets.append(merged_ds)
        except Exception as e:
            print(e)
            _LOG.error(e)
            raise Exception

    return xr.concat(datasets, dim="example")


def fix_dtypes(concat_ds):
    ds_dtypes = {
        "example": np.int32,
        "sat_x_coords": np.int32,
        "sat_y_coords": np.int32,
        "nwp": np.float32,
        "nwp_x_coords": np.float32,
        "nwp_y_coords": np.float32,
        "pv_system_id": np.float32,
        "pv_system_row_number": np.float32,
        "pv_system_x_coords": np.float32,
        "pv_system_y_coords": np.float32,
    }

    for name, dtype in ds_dtypes.items():
        concat_ds[name] = concat_ds[name].astype(dtype)

    assert concat_ds["sat_data"].dtype == np.int16
    return concat_ds


def write_batch_locally(batch: List[Example], batch_i: int):
    dataset = batch_to_dataset(batch)
    dataset = fix_dtypes(dataset)
    encoding = {name: {"compression": "lzf"} for name in dataset.data_vars}
    filename = get_netcdf_filename(batch_i)
    local_filename = LOCAL_TEMP_PATH / filename
    dataset.to_netcdf(local_filename, engine="h5netcdf", mode="w", encoding=encoding)


def iterate_over_dataloader_and_write_to_disk(dataloader: torch.utils.data.DataLoader, dst_path: str):
    _LOG.info("Getting first batch")
    for batch_i, batch in enumerate(dataloader):
        _LOG.info(f"Got batch {batch_i}")
        if len(batch) > 0:
            write_batch_locally(batch, batch_i)
        if batch_i > 0 and batch_i % UPLOAD_EVERY_N_BATCHES == 0:
            upload_and_delete_local_files(dst_path, LOCAL_TEMP_PATH, cloud=CLOUD)
    upload_and_delete_local_files(dst_path, LOCAL_TEMP_PATH, cloud=CLOUD)


def check_directories():
    if CLOUD == 'gcp':
        for path in [DST_TRAIN_PATH, DST_VALIDATION_PATH]:
            check_path_exists(path)


def main():

    run = neptune.init(project='OpenClimateFix/nowcasting-data',
                       capture_stdout=True,
                       capture_stderr=True,
                       capture_hardware_metrics=False)
    _LOG.addHandler(NeptuneHandler(run=run))

    check_directories()
    delete_all_files_in_temp_path(path=LOCAL_TEMP_PATH)
    datamodule = get_data_module()

    _LOG.info("Finished preparing datamodule!")
    _LOG.info("Preparing training data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.train_dataloader(), DST_TRAIN_PATH)
    _LOG.info("Preparing validation data...")
    iterate_over_dataloader_and_write_to_disk(datamodule.val_dataloader(), DST_VALIDATION_PATH)
    _LOG.info("Done!")

    # save configuration to gcs
    save_configuration_to_cloud(configuration=config, cloud=CLOUD)




if __name__ == '__main__':
    main()
