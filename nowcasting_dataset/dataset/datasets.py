import datetime
import pandas as pd
from numbers import Number
from typing import List, Tuple, Iterable, Callable, Union, Optional
import nowcasting_dataset.consts
from nowcasting_dataset import data_sources
from dataclasses import dataclass
from concurrent import futures
import logging
import gcsfs
import boto3
import os
import numpy as np
import xarray as xr
from nowcasting_dataset import utils as nd_utils
from nowcasting_dataset.dataset import example
import torch

from nowcasting_dataset.cloud.gcp import gcp_download_to_local
from nowcasting_dataset.cloud.aws import aws_download_to_local

from nowcasting_dataset.consts import (
    GSP_ID,
    GSP_YIELD,
    GSP_X_COORDS,
    GSP_Y_COORDS,
    GSP_DATETIME_INDEX,
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    SATELLITE_DATA,
    NWP_DATA,
    NWP_X_COORDS,
    NWP_Y_COORDS,
    PV_SYSTEM_X_COORDS,
    PV_SYSTEM_Y_COORDS,
    PV_YIELD,
    PV_AZIMUTH_ANGLE,
    PV_ELEVATION_ANGLE,
    PV_SYSTEM_ID,
    PV_SYSTEM_ROW_NUMBER,
    Y_METERS_CENTER,
    X_METERS_CENTER,
    SATELLITE_DATETIME_INDEX,
    NWP_TARGET_TIME,
    PV_DATETIME_INDEX,
    DATETIME_FEATURE_NAMES,
)
from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES


"""
This file contains the following classes
NetCDFDataset- torch.utils.data.Dataset: Use for loading pre-made batches
NowcastingDataset - torch.utils.data.IterableDataset: Dataset for making batches
"""

SAT_MEAN = xr.DataArray(
    data=[
        93.23458,
        131.71373,
        843.7779,
        736.6148,
        771.1189,
        589.66034,
        862.29816,
        927.69586,
        90.70885,
        107.58985,
        618.4583,
        532.47394,
    ],
    dims=["sat_variable"],
    coords={"sat_variable": list(SAT_VARIABLE_NAMES)},
).astype(np.float32)

SAT_STD = xr.DataArray(
    data=[
        115.34247,
        139.92636,
        36.99538,
        57.366386,
        30.346825,
        149.68007,
        51.70631,
        35.872967,
        115.77212,
        120.997154,
        98.57828,
        99.76469,
    ],
    dims=["sat_variable"],
    coords={"sat_variable": list(SAT_VARIABLE_NAMES)},
).astype(np.float32)

_LOG = logging.getLogger(__name__)


class NetCDFDataset(torch.utils.data.Dataset):
    """Loads data saved by the `prepare_ml_training_data.py` script.
    Moved from predict_pv_yield
    """

    def __init__(
        self,
        n_batches: int,
        src_path: str,
        tmp_path: str,
        cloud: str = "gcp",
        required_keys: Union[Tuple[str], List[str]] = [
            NWP_DATA,
            NWP_X_COORDS,
            NWP_Y_COORDS,
            SATELLITE_DATA,
            SATELLITE_X_COORDS,
            SATELLITE_Y_COORDS,
            PV_YIELD,
            PV_SYSTEM_ID,
            PV_SYSTEM_ROW_NUMBER,
            PV_SYSTEM_X_COORDS,
            PV_SYSTEM_Y_COORDS,
            X_METERS_CENTER,
            Y_METERS_CENTER,
            GSP_ID,
            GSP_YIELD,
            GSP_X_COORDS,
            GSP_Y_COORDS,
            GSP_DATETIME_INDEX,
        ]
        + list(DATETIME_FEATURE_NAMES),
        history_minutes: Optional[int] = None,
        forecast_minutes: Optional[int] = None,
        current_timestep_index: Optional[int] = None,
    ):
        """
        Args:
          n_batches: Number of batches available on disk.
          src_path: The full path (including 'gs://') to the data on
            Google Cloud storage.
          tmp_path: The full path to the local temporary directory
            (on a local filesystem).
        required_keys: Tuple or list of keys required in the example for it to be considered usable
        history_minutes: How many past minutes of data to use, if subsetting the batch
        forecast_minutes: How many future minutes of data to use, if reducing the amount of forecast time
        current_timestep_index: Index into either sat_datetime_index or nwp_target_time indicating the current time,
                                if None, then the entire batch is used, if not None, then subselecting is turned on, and
                                only history_minutes + current time + forecast_minutes data is used.

        """
        self.n_batches = n_batches
        self.src_path = src_path
        self.tmp_path = tmp_path
        self.cloud = cloud
        self.required_keys = list(required_keys)
        self.history_minutes = history_minutes
        self.forecast_minutes = forecast_minutes
        self.current_timestep_index = current_timestep_index

        # setup cloud connections as None
        self.gcs = None
        self.s3_resource = None

        assert cloud in ["gcp", "aws"]

        if not os.path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)

    def per_worker_init(self, worker_id: int):
        if self.cloud == "gcp":
            self.gcs = gcsfs.GCSFileSystem()
        else:
            self.s3_resource = boto3.resource("s3")

    def __len__(self):
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> example.Example:
        """Returns a whole batch at once.

        Args:
          batch_idx: The integer index of the batch. Must be in the range
          [0, self.n_batches).

        Returns:
            NamedDict where each value is a numpy array. The size of this
            array's first dimension is the batch size.
        """
        if not 0 <= batch_idx < self.n_batches:
            raise IndexError(
                "batch_idx must be in the range" f" [0, {self.n_batches}), not {batch_idx}!"
            )
        netcdf_filename = nd_utils.get_netcdf_filename(batch_idx)
        remote_netcdf_filename = os.path.join(self.src_path, netcdf_filename)
        local_netcdf_filename = os.path.join(self.tmp_path, netcdf_filename)

        if self.cloud == "gcp":
            gcp_download_to_local(
                remote_filename=remote_netcdf_filename,
                local_filename=local_netcdf_filename,
                gcs=self.gcs,
            )
        else:
            aws_download_to_local(
                remote_filename=remote_netcdf_filename,
                local_filename=local_netcdf_filename,
                s3_resource=self.s3_resource,
            )

        netcdf_batch = xr.load_dataset(local_netcdf_filename)
        os.remove(local_netcdf_filename)

        batch = example.Example(
            sat_datetime_index=netcdf_batch.sat_time_coords,
            nwp_target_time=netcdf_batch.nwp_time_coords,
        )
        for key in self.required_keys:
            try:
                batch[key] = netcdf_batch[key]
            except KeyError:
                pass

        sat_data = batch[SATELLITE_DATA]
        if sat_data.dtype == np.int16:
            sat_data = sat_data.astype(np.float32)
            sat_data = sat_data - SAT_MEAN
            sat_data /= SAT_STD
            batch[SATELLITE_DATA] = sat_data

        if self.current_timestep_index is not None:
            batch = subselect_data(
                batch=batch,
                required_keys=self.required_keys,
                history_minutes=self.history_minutes,
                forecast_minutes=self.forecast_minutes,
                current_timestep_index=self.current_timestep_index,
            )

        batch = example.to_numpy(batch)

        return batch


@dataclass
class NowcastingDataset(torch.utils.data.IterableDataset):
    """
    The first data_source will be used to select the geo locations each batch.
    """

    batch_size: int
    n_batches_per_epoch_per_worker: int
    #: Number of times to re-use each timestep. Must exactly divide batch_size.
    n_samples_per_timestep: int
    data_sources: List[data_sources.DataSource]
    t0_datetimes: pd.DatetimeIndex  #: Valid t0 datetimes.
    collate_fn: Callable = torch.utils.data._utils.collate.default_collate

    # useful way to skip batches if creating dataset fails halfway through.
    # This might not be that useful, as re-running creation of datasets may cause off issues like duplicate data.
    skip_batch_index: int = 0
    batch_index: int = 0

    def __post_init__(self):
        super().__init__()
        self._per_worker_init_has_run = False
        self._n_timesteps_per_batch = self.batch_size // self.n_samples_per_timestep

        # Sanity checks.
        if self.batch_size % self.n_samples_per_timestep != 0:
            raise ValueError("n_crops_per_timestep must exactly divide batch_size!")
        if len(self.t0_datetimes) < self._n_timesteps_per_batch:
            raise ValueError(
                f"start_dt_index only has {len(self.start_dt_index)}"
                " timestamps."
                f"  Must have at least {self._n_timesteps_per_batch}!"
            )

        if self.skip_batch_index > 0:
            _LOG.warning(f"Will be skipping {self.skip_batch_index}, is this correct?")

    def per_worker_init(self, worker_id: int) -> None:
        """Called by worker_init_fn on each copy of NowcastingDataset after
        the worker process has been spawned."""
        # Each worker must have a different seed for its random number gen.
        # Otherwise all the workers will output exactly the same data!
        self.worker_id = worker_id
        seed = torch.initial_seed()
        self.rng = np.random.default_rng(seed=seed)

        # Initialise each data_source.
        for data_source in self.data_sources:
            _LOG.debug(f"Opening {type(data_source).__name__}")
            data_source.open()

        self._per_worker_init_has_run = True

    def __iter__(self):
        """Yields a complete batch at a time."""
        if not self._per_worker_init_has_run:
            raise RuntimeError("per_worker_init() must be run!")
        for _ in range(self.n_batches_per_epoch_per_worker):
            yield self._get_batch()

    def _get_batch(self) -> torch.Tensor:

        _LOG.debug(f"Getting batch {self.batch_index}")

        self.batch_index += 1
        if self.batch_index < self.skip_batch_index:
            _LOG.debug(f"Skipping batch {self.batch_index}")
            return []

        t0_datetimes = self._get_t0_datetimes_for_batch()
        x_locations, y_locations = self._get_locations_for_batch(t0_datetimes)

        examples = None
        n_threads = len(self.data_sources)
        with futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit tasks to the executor.
            future_examples_per_source = []
            for data_source in self.data_sources:
                future_examples = executor.submit(
                    data_source.get_batch,
                    t0_datetimes=t0_datetimes,
                    x_locations=x_locations,
                    y_locations=y_locations,
                )
                future_examples_per_source.append(future_examples)

            # Collect results from each thread.
            for future_examples in future_examples_per_source:
                examples_from_source = future_examples.result()
                if examples is None:
                    examples = examples_from_source
                else:
                    for i in range(self.batch_size):
                        examples[i].update(examples_from_source[i])

        return self.collate_fn(examples)

    def _get_t0_datetimes_for_batch(self) -> pd.DatetimeIndex:
        # Pick random datetimes.
        t0_datetimes = self.rng.choice(
            self.t0_datetimes, size=self._n_timesteps_per_batch, replace=False
        )
        # Duplicate these random datetimes.
        t0_datetimes = np.tile(t0_datetimes, reps=self.n_samples_per_timestep)
        return pd.DatetimeIndex(t0_datetimes)

    def _get_locations_for_batch(
        self, t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        return self.data_sources[0].get_locations_for_batch(t0_datetimes)


def worker_init_fn(worker_id):
    """Configures each dataset worker process.

    Just has one job!  To call NowcastingDataset.per_worker_init().
    """
    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        print("worker_info is None!")
    else:
        # The NowcastingDataset copy in this worker process.
        dataset_obj = worker_info.dataset
        dataset_obj.per_worker_init(worker_info.id)


def select_time_period(
    batch: example.Example,
    keys: List[str],
    time_of_first_example: pd.DatetimeIndex,
    start_time: xr.DataArray,
    end_time: xr.DataArray,
) -> example.Example:
    """
    Selects a subset of data between the indicies of [start, end] for each key in keys

    Args:
        batch: Example containing the data
        keys: Keys in batch to use
        time_of_first_example: Datetime of the current time in the first example of the batch
        start_time: Start time DataArray
        end_time: End time DataArray

    Returns:
        Example containing the subselected data
    """
    start_i, end_i = np.searchsorted(time_of_first_example, [start_time.data, end_time.data])
    for key in keys:
        batch[key] = batch[key].isel(time=slice(start_i, end_i))

    return batch


def subselect_data(
    batch: example.Example,
    required_keys: Union[Tuple[str], List[str]],
    current_timestep_index: int,
    history_minutes: int,
    forecast_minutes: int,
) -> example.Example:
    """
    Subselects the data temporally. This function selects all data within the time range [t0 - history_minutes, t0 + forecast_minutes]

    Args:
        batch: Example dictionary containing at least the required_keys
        required_keys: The required keys present in the dictionary to use
        current_timestep_index: The index into either SATELLITE_DATETIME_INDEX or NWP_TARGET_TIME giving the current timestep
        history_minutes: How many minutes of history to use
        forecast_minutes: How many minutes of future data to use for forecasting

    Returns:
        Example with only data between [t0 - history_minutes, t0 + forecast_minutes] remaining
    """

    # We are subsetting the data
    date_time_index_to_use = (
        SATELLITE_DATETIME_INDEX if SATELLITE_DATA in required_keys else NWP_TARGET_TIME
    )
    current_time = batch[date_time_index_to_use].isel(time=current_timestep_index)[0]
    # Datetimes are in seconds, so just need to convert minutes to second + 30sec buffer
    # Only need to do it for the first example in the batch, as masking indicies should be the same for all of them
    # The extra 30 seconds is added to ensure that the first and last timestep are always contained
    # within the [start_time, end_time] range
    start_time = current_time - pd.to_timedelta(f"{history_minutes} minute 30 second")
    end_time = current_time + pd.to_timedelta(f"{forecast_minutes} minute 30 second")
    if SATELLITE_DATA in required_keys:
        batch = select_time_period(
            batch,
            keys=[SATELLITE_DATA, SATELLITE_DATETIME_INDEX] + list(DATETIME_FEATURE_NAMES),
            time_of_first_example=batch[SATELLITE_DATETIME_INDEX][0].data,
            start_time=start_time,
            end_time=end_time,
        )
        _LOG.debug(
            f"Sat Datetime Shape: {batch[SATELLITE_DATETIME_INDEX].shape} Sat Data Shape: {batch[SATELLITE_DATA].shape}"
        )

    # Now for NWP, if used
    if NWP_DATA in required_keys:
        batch = select_time_period(
            batch,
            keys=(
                [NWP_DATA, NWP_TARGET_TIME] + list(DATETIME_FEATURE_NAMES)
                if SATELLITE_DATA not in required_keys
                else [NWP_DATA, NWP_TARGET_TIME]
            ),
            time_of_first_example=batch[NWP_TARGET_TIME][0].data,
            start_time=start_time,
            end_time=end_time,
        )
        _LOG.debug(
            f"NWP Datetime Shape: {batch[NWP_TARGET_TIME].shape} NWP Data Shape: {batch[NWP_DATA].shape}"
        )

    # Now GSP, if used
    if GSP_YIELD in required_keys and GSP_DATETIME_INDEX in batch:
        batch = select_time_period(
            batch,
            keys=[GSP_DATETIME_INDEX, GSP_YIELD],
            time_of_first_example=batch[GSP_DATETIME_INDEX][0].data,
            start_time=start_time,
            end_time=end_time,
        )
        _LOG.debug(
            f"GSP Datetime Shape: {batch[GSP_DATETIME_INDEX].shape} GSP Data Shape: {batch[GSP_YIELD].shape}"
        )

    # Now PV systems, if used
    if PV_YIELD in required_keys and PV_DATETIME_INDEX in batch:
        batch = select_time_period(
            batch,
            keys=[PV_DATETIME_INDEX, PV_YIELD, PV_AZIMUTH_ANGLE, PV_ELEVATION_ANGLE],
            time_of_first_example=batch[PV_DATETIME_INDEX][0].data,
            start_time=start_time,
            end_time=end_time,
        )
        _LOG.debug(
            f"PV Datetime Shape: {batch[PV_DATETIME_INDEX].shape} PV Data Shape: {batch[PV_YIELD].shape}"
            f" PV Azimuth Shape: {batch[PV_AZIMUTH_ANGLE].shape} PV Elevation Shape: {batch[PV_ELEVATION_ANGLE].shape}"
        )

    return batch
