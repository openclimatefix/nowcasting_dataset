import pandas as pd
import numpy as np
from numbers import Number
from typing import List, Tuple, Iterable
import nowcasting_dataset
from nowcasting_dataset import data_sources
from dataclasses import dataclass
import torch
from concurrent import futures


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
        t0_datetimes = self._get_t0_datetimes_for_batch()
        x_locations, y_locations = self._get_locations_for_batch(t0_datetimes)

        # Load the first _n_timesteps_per_batch concurrently.  This
        # loads the timesteps from disk concurrently, and fills the
        # DataSource caches.  If we try loading all examples
        # concurrently, then all the DataSources try reading from
        # empty caches, and things are much slower!
        zipped = list(zip(t0_datetimes, x_locations, y_locations))
        with futures.ThreadPoolExecutor(
                max_workers=self.batch_size) as executor:
            future_examples = []
            for coords in zipped[:self._n_timesteps_per_batch]:
                t0_datetime, x_location, y_location = coords
                future_example = executor.submit(
                    self._get_example, t0_datetime, x_location, y_location)
                future_examples.append(future_example)
            examples = [
                future_example.result() for future_example in future_examples]

        # Load the remaining examples.  This should hit the DataSource caches.
        for coords in zipped[self._n_timesteps_per_batch:]:
            t0_datetime, x_location, y_location = coords
            example = self._get_example(t0_datetime, x_location, y_location)
            examples.append(example)

        # Tell the DataSources that we've finished sampling this batch.
        for data_source in self.data_sources:
            data_source.batch_end()

        return torch.utils.data._utils.collate.default_collate(examples)

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


class ContiguousNowcastingDataset(NowcastingDataset):
    """Each batch contains contiguous timesteps for a single location."""

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
