import pandas as pd
import numpy as np
from numbers import Number
from typing import List
import nowcasting_dataset
from nowcasting_dataset import data_sources
from dataclasses import dataclass
import torch


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

    def _get_batch(self):
        # Pick datetimes.
        t0_datetimes = self.rng.choice(
            self.t0_datetimes,
            size=self._n_timesteps_per_batch,
            replace=False)
        # Duplicate these random datetimes.
        t0_datetimes = list(t0_datetimes) * self.n_samples_per_timestep
        t0_datetimes = pd.DatetimeIndex(t0_datetimes)

        # Pick locations.
        locations = self.data_sources[0].pick_locations_for_batch(t0_datetimes)

        # Loop to construct batch.
        examples = []
        for t0_dt, location in zip(t0_datetimes, locations):
            x_meters_center, y_meters_center = location
            example = self._get_example(
                t0_dt=t0_dt,
                x_meters_center=x_meters_center,
                y_meters_center=y_meters_center)
            examples.append(example)

        # Tell the DataSources that we've finished sampling this batch.
        for data_source in self.data_sources:
            data_source.batch_end()

        return torch.utils.data._utils.collate.default_collate(examples)

    def _get_example(
            self,
            t0_dt: pd.Timestamp,
            x_meters_center: Number,
            y_meters_center: Number) -> nowcasting_dataset.example.Example:

        example = nowcasting_dataset.example.Example(t0_dt=t0_dt)
        for data_source in self.data_sources:
            example_from_source = data_source.get_sample(
                t0_dt=t0_dt,
                x_meters_center=x_meters_center,
                y_meters_center=y_meters_center)
            example.update(example_from_source)
        example = nowcasting_dataset.example.to_numpy(example)
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
