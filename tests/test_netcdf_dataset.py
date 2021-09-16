import os
import torch
from nowcasting_dataset.dataset.datasets import NetCDFDataset, worker_init_fn, subselect_data
from nowcasting_dataset.consts import (
    SATELLITE_X_COORDS,
    SATELLITE_Y_COORDS,
    SATELLITE_DATA,
    NWP_DATA,
    SATELLITE_DATETIME_INDEX,
    NWP_TARGET_TIME,
    NWP_Y_COORDS,
    NWP_X_COORDS,
    PV_YIELD,
    GSP_YIELD,
)
import plotly.graph_objects as go
import plotly
import pandas as pd
import pytest
import tempfile
from pathlib import Path
import numpy as np


def test_subselect_date():
    batch_size = 32
    seq_length = 20
    x = {
        SATELLITE_DATA: np.random.random((batch_size, seq_length, 128, 128, 12)),
        PV_YIELD: np.random.random((batch_size, seq_length, 128)),
        "pv_system_id": np.random.random((batch_size, 128)),
        NWP_DATA: np.random.random((batch_size, 10, seq_length, 2, 2)),
        "hour_of_day_sin": np.random.random((batch_size, seq_length)),
        "hour_of_day_cos": np.random.random((batch_size, seq_length)),
        "day_of_year_sin": np.random.random((batch_size, seq_length)),
        "day_of_year_cos": np.random.random((batch_size, seq_length)),
    }

    # add a nan
    x[PV_YIELD][0, 0, :] = float("nan")

    # add fake x and y coords, and make sure they are sorted
    x[SATELLITE_X_COORDS] = np.random.random((batch_size, seq_length))
    x[SATELLITE_Y_COORDS] = np.random.random((batch_size, seq_length))

    # add sorted (fake) time series, 5min*60 = 300second steps
    timeseries = np.arange(start=0, stop=300 * batch_size * seq_length, step=300).reshape(
        (batch_size, seq_length)
    )
    x[SATELLITE_DATETIME_INDEX] = timeseries
    x[NWP_TARGET_TIME] = timeseries

    batch = subselect_data(
        x,
        required_keys=(NWP_DATA, NWP_TARGET_TIME, SATELLITE_DATA, SATELLITE_DATETIME_INDEX),
        current_timestep_index=7,
        history_minutes=10,
        forecast_minutes=10,
    )
    assert batch[SATELLITE_DATA].shape[1] == 5
    assert batch[NWP_DATA].shape[2] == 5


@pytest.mark.skip("CD does not have access to GCS")
def test_get_dataloaders_gcp():
    DATA_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/"
    TEMP_PATH = "../nowcasting_dataset"

    train_dataset = NetCDFDataset(
        24_900, os.path.join(DATA_PATH, "train"), os.path.join(TEMP_PATH, "train")
    )

    dataloader_config = dict(
        pin_memory=True,
        num_workers=2,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        # Disable automatic batching because dataset
        # returns complete batches.
        batch_size=None,
    )

    _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

    train_dataset.per_worker_init(1)
    t = iter(train_dataset)
    data = next(t)

    # image
    z = data[SATELLITE_DATA][0][0][:, :, 0]
    _ = data[GSP_YIELD][0][:, 0]

    _ = pd.to_datetime(data[SATELLITE_DATETIME_INDEX][0, 0], unit="s")

    fig = go.Figure(data=go.Contour(z=z))

    plotly.offline.plot(fig, filename="../filename.html", auto_open=True)


@pytest.mark.skip("CD does not have access to AWS")
def test_get_dataloaders_aws():

    with tempfile.TemporaryDirectory() as tmpdirname:
        TEMP_PATH = Path(tmpdirname)
        DATA_PATH = "prepared_ML_training_data/v4/"

        os.mkdir(os.path.join(TEMP_PATH, "train"))

        train_dataset = NetCDFDataset(
            24_900, os.path.join(DATA_PATH, "train"), os.path.join(TEMP_PATH, "train"), cloud="aws"
        )

        dataloader_config = dict(
            pin_memory=True,
            num_workers=2,
            prefetch_factor=8,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

        _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

        train_dataset.per_worker_init(1)
        t = iter(train_dataset)
        data = next(t)

        assert SATELLITE_DATA in data.keys()


@pytest.mark.skip("CD does not have access to GCP")
def test_required_keys_gcp():

    DATA_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/"
    TEMP_PATH = "../nowcasting_dataset"
    if os.path.isdir(os.path.join(TEMP_PATH, "train")):
        os.removedirs(os.path.join(TEMP_PATH, "train"))
    os.mkdir(os.path.join(TEMP_PATH, "train"))

    train_dataset = NetCDFDataset(
        24_900,
        os.path.join(DATA_PATH, "train"),
        os.path.join(TEMP_PATH, "train"),
        cloud="gcp",
        required_keys=[
            NWP_DATA,
            NWP_X_COORDS,
            NWP_Y_COORDS,
            SATELLITE_DATA,
            SATELLITE_X_COORDS,
            SATELLITE_Y_COORDS,
        ],
    )

    dataloader_config = dict(
        pin_memory=True,
        num_workers=2,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        # Disable automatic batching because dataset
        # returns complete batches.
        batch_size=None,
    )

    _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

    train_dataset.per_worker_init(1)
    t = iter(train_dataset)
    data = next(t)

    assert SATELLITE_DATA in data.keys()
    assert PV_YIELD not in data.keys()


@pytest.mark.skip("CD does not have access to GCP")
def test_subsetting_gcp():

    DATA_PATH = "gs://solar-pv-nowcasting-data/prepared_ML_training_data/v5/"
    TEMP_PATH = "../nowcasting_dataset"
    if os.path.isdir(os.path.join(TEMP_PATH, "train")):
        os.removedirs(os.path.join(TEMP_PATH, "train"))
    os.mkdir(os.path.join(TEMP_PATH, "train"))

    train_dataset = NetCDFDataset(
        24_900,
        os.path.join(DATA_PATH, "train"),
        os.path.join(TEMP_PATH, "train"),
        cloud="gcp",
        history_minutes=10,
        forecast_minutes=10,
        current_timestep_index=7,
    )

    dataloader_config = dict(
        pin_memory=True,
        num_workers=2,
        prefetch_factor=8,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        # Disable automatic batching because dataset
        # returns complete batches.
        batch_size=None,
    )

    _ = torch.utils.data.DataLoader(train_dataset, **dataloader_config)

    train_dataset.per_worker_init(1)
    t = iter(train_dataset)
    data = next(t)

    sat_data = data[SATELLITE_DATA]

    # Sat is in 5min increments, so should have 2 history + current + 2 future
    assert sat_data.shape[1] == 5
    assert data[NWP_DATA].shape[2] == 5
