import pandas as pd
from numbers import Number
from typing import List, Tuple, Iterable, Callable
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
from nowcasting_dataset import example
import torch

from nowcasting_dataset.cloud.gcp import gcp_download_to_local
from nowcasting_dataset.cloud.aws import aws_download_to_local

from nowcasting_dataset.data_sources.satellite_data_source import SAT_VARIABLE_NAMES


SAT_MEAN = xr.DataArray(
    data=[
        93.23458, 131.71373, 843.7779, 736.6148, 771.1189, 589.66034,
        862.29816, 927.69586,  90.70885, 107.58985, 618.4583, 532.47394],
    dims=['sat_variable'],
    coords={'sat_variable': list(SAT_VARIABLE_NAMES)}).astype(np.float32)

SAT_STD = xr.DataArray(
    data=[
        115.34247, 139.92636,  36.99538,  57.366386,  30.346825,
        149.68007,  51.70631,  35.872967, 115.77212, 120.997154,
        98.57828,  99.76469],
    dims=['sat_variable'],
    coords={'sat_variable': list(SAT_VARIABLE_NAMES)}).astype(np.float32)

_LOG = logging.getLogger('nowcasting_dataset')


class NetCDFDataset(torch.utils.data.Dataset):
    """Loads data saved by the `prepare_ml_training_data.py` script.
    Moved from predict_pv_yield
    """

    def __init__(
            self, n_batches: int, src_path: str, tmp_path: str, cloud: str = 'gcp'):
        """
        Args:
          n_batches: Number of batches available on disk.
          src_path: The full path (including 'gs://') to the data on
            Google Cloud storage.
          tmp_path: The full path to the local temporary directory
            (on a local filesystem).
        """
        self.n_batches = n_batches
        self.src_path = src_path
        self.tmp_path = tmp_path
        self.cloud = cloud

        # setup cloud connections as None
        self.gcs = None
        self.s3_resource = None

        assert cloud in ['gcp', 'aws']

        if not os.path.isdir(self.tmp_path):
            os.mkdir(self.tmp_path)

    def per_worker_init(self, worker_id: int):
        if self.cloud == 'gcp':
            self.gcs = gcsfs.GCSFileSystem()
        else:
            self.s3_resource = boto3.resource('s3')

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
                'batch_idx must be in the range'
                f' [0, {self.n_batches}), not {batch_idx}!')
        netcdf_filename = nd_utils.get_netcdf_filename(batch_idx)
        remote_netcdf_filename = os.path.join(self.src_path, netcdf_filename)
        local_netcdf_filename = os.path.join(self.tmp_path, netcdf_filename)

        if self.cloud == 'gcp':
            gcp_download_to_local(remote_filename=remote_netcdf_filename,
                                  local_filename=local_netcdf_filename,
                                  gcs=self.gcs)
        else:
            aws_download_to_local(remote_filename=remote_netcdf_filename,
                                  local_filename=local_netcdf_filename,
                                  s3_resource=self.s3_resource)

        netcdf_batch = xr.load_dataset(local_netcdf_filename)
        os.remove(local_netcdf_filename)

        batch = example.Example(
            sat_datetime_index=netcdf_batch.sat_time_coords,
            nwp_target_time=netcdf_batch.nwp_time_coords)
        for key in [
            'nwp', 'nwp_x_coords', 'nwp_y_coords',
            'sat_data', 'sat_x_coords', 'sat_y_coords',
            'pv_yield', 'pv_system_id', 'pv_system_row_number',
            'pv_system_x_coords', 'pv_system_y_coords',
            'x_meters_center', 'y_meters_center'
        ] + list(example.DATETIME_FEATURE_NAMES):
            try:
                batch[key] = netcdf_batch[key]
            except KeyError:
                pass

        sat_data = batch['sat_data']
        if sat_data.dtype == np.int16:
            sat_data = sat_data.astype(np.float32)
            sat_data = sat_data - SAT_MEAN
            sat_data /= SAT_STD
            batch['sat_data'] = sat_data

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
        self._n_timesteps_per_batch = (
            self.batch_size // self.n_samples_per_timestep)

        # Sanity checks.
        if self.batch_size % self.n_samples_per_timestep != 0:
            raise ValueError(
                'n_crops_per_timestep must exactly divide batch_size!')
        if len(self.t0_datetimes) < self._n_timesteps_per_batch:
            raise ValueError(
                f'start_dt_index only has {len(self.start_dt_index)}'
                ' timestamps.'
                f'  Must have at least {self._n_timesteps_per_batch}!')

        if self.skip_batch_index > 0:
            _LOG.warning(f'Will be skipping {self.skip_batch_index}, is this correct?')

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
            data_source.open()

        self._per_worker_init_has_run = True

    def __iter__(self):
        """Yields a complete batch at a time."""
        if not self._per_worker_init_has_run:
            raise RuntimeError('per_worker_init() must be run!')
        for _ in range(self.n_batches_per_epoch_per_worker):
            yield self._get_batch()

    def _get_batch(self) -> torch.Tensor:

        _LOG.debug(f'Getting batch {self.batch_index}')

        self.batch_index += 1
        if self.batch_index < self.skip_batch_index:
            _LOG.debug(f'Skipping batch {self.batch_index}')
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
                    y_locations=y_locations)
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
            self.t0_datetimes, size=self._n_timesteps_per_batch, replace=False)
        # Duplicate these random datetimes.
        t0_datetimes = np.tile(t0_datetimes, reps=self.n_samples_per_timestep)
        return pd.DatetimeIndex(t0_datetimes)

    def _get_locations_for_batch(
            self,
            t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[List[Number], List[Number]]:
        return self.data_sources[0].get_locations_for_batch(t0_datetimes)


@dataclass
class ContiguousNowcastingDataset(NowcastingDataset):
    """Each batch contains contiguous timesteps for a single location."""

    def __post_init__(self):
        super().__post_init__()

    def _get_t0_datetimes_for_batch(self) -> pd.DatetimeIndex:
        max_i = len(self.t0_datetimes) - self.batch_size
        start_i = self.rng.integers(low=0, high=max_i)
        end_i = start_i + self.batch_size
        t0_datetimes = self.t0_datetimes[start_i:end_i]
        return pd.DatetimeIndex(t0_datetimes)

    def _get_locations_for_batch(
            self,
            t0_datetimes: pd.DatetimeIndex
    ) -> Tuple[Iterable[Number], Iterable[Number]]:
        x_locations, y_locations = super()._get_locations_for_batch(
            t0_datetimes)
        x_locations = np.repeat(x_locations[0], repeats=self.batch_size)
        y_locations = np.repeat(y_locations[0], repeats=self.batch_size)
        return x_locations, y_locations


def worker_init_fn(worker_id):
    """Configures each dataset worker process.

    Just has one job!  To call NowcastingDataset.per_worker_init().
    """
    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        print('worker_info is None!')
    else:
        # The NowcastingDataset copy in this worker process.
        dataset_obj = worker_info.dataset
        dataset_obj.per_worker_init(worker_info.id)
