import os
import pandas as pd

import nowcasting_dataset
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWPDataSource


PATH = os.path.dirname(nowcasting_dataset.__file__)

# Solar PV data (test data)
NWP_ZARR_PATH = f"{PATH}/../tests/data/nwp_data/test.zarr"


def test_nwp_data_source_init():
    _ = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=30,
        forecast_minutes=60,
        n_timesteps_per_batch=8,
    )


def test_nwp_data_source_open():
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=30,
        forecast_minutes=60,
        n_timesteps_per_batch=8,
        channels=["t"],
    )

    nwp.open()


def test_nwp_data_source_batch():
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=30,
        forecast_minutes=60,
        n_timesteps_per_batch=8,
        channels=["t"],
    )

    nwp.open()

    t0_datetimes = nwp._data.init_time[2:10].values
    x = nwp._data.x[0:4].values
    y = nwp._data.y[0:4].values

    batch = nwp.get_batch(t0_datetimes=t0_datetimes, x_locations=x, y_locations=y)

    assert batch.data.shape == (4, 1, 19, 2, 2)


def test_nwp_get_contiguous_time_periods():
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=30,
        forecast_minutes=60,
        n_timesteps_per_batch=8,
        channels=["t"],
    )

    contiguous_time_periods = nwp.get_contiguous_time_periods()
    correct_time_periods = pd.DataFrame(
        [{"start_dt": pd.Timestamp("2019-01-01 00:00"), "end_dt": pd.Timestamp("2019-01-02 02:00")}]
    )
    pd.testing.assert_frame_equal(contiguous_time_periods, correct_time_periods)


def test_nwp_get_contiguous_t0_time_periods():
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=30,
        forecast_minutes=60,
        n_timesteps_per_batch=8,
        channels=["t"],
    )

    contiguous_time_periods = nwp.get_contiguous_t0_time_periods()
    correct_time_periods = pd.DataFrame(
        [{"start_dt": pd.Timestamp("2019-01-01 00:30"), "end_dt": pd.Timestamp("2019-01-02 01:00")}]
    )
    pd.testing.assert_frame_equal(contiguous_time_periods, correct_time_periods)
