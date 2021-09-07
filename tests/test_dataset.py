import numpy as np
from nowcasting_dataset.dataset import NowcastingDataset, NetCDFDataset
import nowcasting_dataset.time as nd_time
import pytest


@pytest.fixture
def dataset(sat_data_source):
    all_datetimes = sat_data_source.datetime_index()
    t0_datetimes = nd_time.get_t0_datetimes(
        datetimes=all_datetimes, total_seq_len=2,
        history_len=0)
    return NowcastingDataset(
        batch_size=8,
        n_batches_per_epoch_per_worker=64,
        n_samples_per_timestep=2,
        data_sources=[sat_data_source],
        t0_datetimes=t0_datetimes)


def test_post_init(dataset: NowcastingDataset):
    assert dataset._n_timesteps_per_batch == 4
    assert not dataset._per_worker_init_has_run


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
