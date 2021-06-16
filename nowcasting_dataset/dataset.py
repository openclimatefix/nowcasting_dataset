import pandas as pd
import numpy as np
from typing import List
from nowcasting_dataset import data_sources, segment
from dataclasses import dataclass
import torch


@dataclass
class NowcastingDataset(torch.utils.data.IterableDataset):
    batch_size: int
    history_len: int
    forecast_len: int
    data_sources: List[data_sources.DataSource]
    dt_index: pd.DatetimeIndex
    contiguous_segments: List[segment.Segment]

    def __post_init__(self):
        pass

    def __iter__(self):
        """Yields a complete batch at a time."""
        # blah blah yield...

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
