import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly
import plotly.graph_objects as go
import pytest
import torch
import xarray as xr

import nowcasting_dataset
import nowcasting_dataset.dataset.batch
from nowcasting_dataset.config.model import Configuration
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
    GSP_DATETIME_INDEX,
    T0_DT,
)

# from nowcasting_dataset.dataset import example
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.dataset.datasets import NetCDFDataset, worker_init_fn
from nowcasting_dataset.dataset.subset import subselect_data


def test_subselect_date(test_data_folder):

    x = Batch.fake()

    batch = subselect_data(
        x,
        current_timestep_index=7,
        history_minutes=10,
        forecast_minutes=10,
    )

    assert batch.satellite.data.shape == (32, 5, 64, 64, 12)
    assert batch.nwp.data.shape == (32, 5, 64, 64, 10)


#
def test_subselect_date_with_to_dt(test_data_folder):

    # x = Batch.load_netcdf(f"{test_data_folder}/0.nc")
    x = Batch.fake()

    batch = subselect_data(
        x,
        history_minutes=10,
        forecast_minutes=10,
    )

    assert batch.satellite.data.shape == (32, 5, 64, 64, 12)
    assert batch.nwp.data.shape == (32, 5, 64, 64, 10)
