import dask
import pandas as pd
import numpy as np
from typing import List
import nowcasting_dataset
from nowcasting_dataset import data_sources
from nowcasting_dataset import time as nd_time
from dataclasses import dataclass
import torch


@dataclass
class NowcastingDataset(torch.utils.data.IterableDataset):
    batch_size: int
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
            self.total_seq_len)
        self.history_duration = nd_time.timesteps_to_duration(
            self.history_len)

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

    def __iter__(self):
        """Yields a complete batch at a time."""
        while True:
            yield self._get_batch()

    def _get_batch(self):
        examples = []
        for _ in range(self.batch_size):
            examples.append(self._get_example())
        batch_delayed = self._colate_fn(examples)
        return dask.compute(batch_delayed)

    def _get_example(self) -> nowcasting_dataset.example.Example:
        start_dt = self.rng.choice(self.start_dt_index)
        end_dt = start_dt + self.total_seq_duration
        t0_dt = start_dt + self.history_duration
        x_meters = y_meters = 20_000  # TODO: Change this hard-coding!
        example = nowcasting_dataset.example.Example(
            start_datetime=start_dt,
            end_datetime=end_dt,
            t0_datetime=t0_dt)
        for data_source in self.data_sources:
            example_from_source = data_source.get_sample(
                start=start_dt,
                end=end_dt,
                t0=t0_dt,
                x_meters_center=x_meters,
                y_meters_center=y_meters)
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
