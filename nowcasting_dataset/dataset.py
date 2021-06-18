import dask
import pandas as pd
import numpy as np
from numbers import Number
from typing import List
import itertools
import nowcasting_dataset
from nowcasting_dataset import data_sources
from nowcasting_dataset import time as nd_time
from dataclasses import dataclass
import torch


@dataclass
class NowcastingDataset(torch.utils.data.IterableDataset):
    batch_size: int
    n_samples_per_timestep: int #: Number of times to re-use each timestep.  Must exactly divide batch_size.
    history_len: int
    forecast_len: int
    data_sources: List[data_sources.DataSource]
    start_dt_index: pd.DatetimeIndex  #: Valid start times for examples.

    def __post_init__(self):
        super().__init__()
        self._colate_fn = dask.delayed(
            torch.utils.data._utils.collate.default_collate)
        self.total_seq_len = self.history_len + self.forecast_len
        self.total_seq_duration = nd_time.timesteps_to_duration(
            self.total_seq_len - 1)
        self.history_duration = nd_time.timesteps_to_duration(
            self.history_len - 1)
        self._per_worker_init_has_run = False
        self._n_timesteps_per_batch = self.batch_size // self.n_samples_per_timestep

        # Sanity checks.
        if self.batch_size % self.n_samples_per_timestep != 0:
            raise ValueError('n_crops_per_timestep must exactly divide batch_size!')
        if len(self.start_dt_index) < self._n_timesteps_per_batch:
            raise ValueError(
                f'start_dt_index only has {len(self.start_dt_index)} timestamps.'
                f'  Must have at least {self._n_timesteps_per_batch}!')

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
        while True:
            yield self._get_batch()

    def _get_batch(self):
        # Pick datetimes
        start_datetimes = self.rng.choice(
            self.start_dt_index, 
            size=self._n_timesteps_per_batch,
            replace=False)
        start_datetimes = pd.DatetimeIndex(start_datetimes)
        
        # Pick locations.
        # TODO: Do this properly, using PV locations!
        # Locations is a list of 2-tuples (<x_meters_center, y_meters_center>).
        # The length of locations is self.n_samples_per_timesteps
        locations = [(20_000, 40_000), (500_000, 600_000), (100_000, 100_000), (250_000, 250_000)]
        
        examples = []
        for start_dt, location in itertools.product(start_datetimes, locations):
            x_meters_center, y_meters_center = location
            example = self._get_example(
                start_dt=start_dt,
                x_meters_center=x_meters_center,
                y_meters_center=y_meters_center)
            example = nowcasting_dataset.example.to_numpy(example)
            examples.append(example)
        batch_delayed = self._colate_fn(examples)
        return dask.compute(batch_delayed)[0]

    def _get_example(
        self, 
        start_dt: pd.Timestamp, 
        x_meters_center: Number, 
        y_meters_center: Number) -> nowcasting_dataset.example.Example:
        
        end_dt = start_dt + self.total_seq_duration
        t0_dt = start_dt + self.history_duration
        example = nowcasting_dataset.example.Example(
            start_dt=start_dt, end_dt=end_dt, t0_dt=t0_dt)
        for data_source in self.data_sources:
            example_from_source = data_source.get_sample(
                start_dt=start_dt, end_dt=end_dt, t0_dt=t0_dt,
                x_meters_center=x_meters_center, 
                y_meters_center=y_meters_center)
            example.update(example_from_source)
        return example


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
