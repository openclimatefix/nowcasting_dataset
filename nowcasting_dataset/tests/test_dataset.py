import pandas as pd
import numpy as np
from nowcasting_dataset.dataset import NowcastingDataset
import pytest


@pytest.fixture
def dataset(sat_data_source):
    # TODO: Use better approach to getting start_dt_index!
    start_dt_index = sat_data_source.available_timestamps()[:-2]
    return NowcastingDataset(
        batch_size=8, history_len=1, forecast_len=1,
        n_samples_per_timestep=2,
        data_sources=[sat_data_source],
        start_dt_index=start_dt_index)


def test_post_init(dataset: NowcastingDataset):
    assert dataset.total_seq_len == 2
    assert dataset.total_seq_duration == pd.Timedelta('5 minutes')
    assert dataset.history_duration == pd.Timedelta('0 minutes')
    assert dataset._colate_fn is not None


def test_per_worker_init(dataset: NowcastingDataset):
    WORKER_ID = 1
    dataset.per_worker_init(worker_id=WORKER_ID)
    assert isinstance(dataset.rng, np.random.Generator)
    assert dataset.worker_id == WORKER_ID


def test_get_batch(dataset: NowcastingDataset):
    dataset.per_worker_init(worker_id=1)
    example = dataset._get_batch()
    assert isinstance(example, dict)
    assert 'sat_data' in example
    assert example['sat_data'].shape == (
        8, 2, pytest.IMAGE_SIZE_PIXELS, pytest.IMAGE_SIZE_PIXELS, 1)
