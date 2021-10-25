import numpy as np
import pandas as pd
import pytest

import nowcasting_dataset.time as nd_time
from nowcasting_dataset.dataset.datasets import NowcastingDataset
from nowcasting_dataset.dataset.batch import Batch


def _get_t0_datetimes(data_source, freq) -> pd.DatetimeIndex:
    t0_periods = data_source.get_contiguous_t0_time_periods()
    t0_datetimes = nd_time.time_periods_to_datetime_index(t0_periods, freq=freq)
    return t0_datetimes


@pytest.fixture
def dataset(sat_data_source, general_data_source):
    t0_datetimes = _get_t0_datetimes(sat_data_source, freq="5T")

    return NowcastingDataset(
        batch_size=8,
        n_batches_per_epoch_per_worker=64,
        n_samples_per_timestep=2,
        data_sources=[sat_data_source, general_data_source],
        t0_datetimes=t0_datetimes,
    )


@pytest.fixture
def dataset_gsp(gsp_data_source, general_data_source):
    t0_datetimes = _get_t0_datetimes(gsp_data_source, freq="30T")

    return NowcastingDataset(
        batch_size=8,
        n_batches_per_epoch_per_worker=64,
        n_samples_per_timestep=2,
        data_sources=[gsp_data_source, general_data_source],
        t0_datetimes=t0_datetimes,
    )


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
    with pytest.raises(NotImplementedError):
        _ = dataset._get_batch()


def test_get_batch_gsp(dataset_gsp: NowcastingDataset):
    dataset_gsp.per_worker_init(worker_id=1)
    batch = dataset_gsp._get_batch()
    assert isinstance(batch, Batch)

    assert batch.gsp is not None
