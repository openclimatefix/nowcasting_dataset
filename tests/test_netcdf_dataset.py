import os
import torch
from nowcasting_dataset.dataset.datasets import NetCDFDataset, worker_init_fn
import plotly.graph_objects as go
import plotly
import pandas as pd
import pytest
import tempfile
from pathlib import Path


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
    z = data["sat_data"][0][0][:, :, 0]
    _ = data["gsp_yield"][0][:, 0]

    _ = pd.to_datetime(data["sat_datetime_index"][0, 0], unit="s")

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

        assert "sat_data" in data.keys()


@pytest.mark.skip("CD does not have access to AWS")
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
            "nwp",
            "nwp_x_coords",
            "nwp_y_coords",
            "sat_data",
            "sat_x_coords",
            "sat_y_coords",
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

    assert "sat_data" in data.keys()
    assert "pv_yield" not in data.keys()


@pytest.mark.skip("CD does not have access to AWS")
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
        current_sat_timestep_index=7,
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

    sat_data = data["sat_data"]

    # Sat is in 5min increments, so should have 2 history + current + 2 future
    assert sat_data.shape[1] == 5
    assert data["nwp"].shape[2] == 5
