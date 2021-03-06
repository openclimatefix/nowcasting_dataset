# noqa: D100
import os

import pandas as pd

import nowcasting_dataset
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWPDataSource

PATH = os.path.dirname(nowcasting_dataset.__file__)

# Solar PV data (test data)
NWP_ZARR_PATH = f"{PATH}/../tests/data/nwp_data/test.zarr"


def test_nwp_data_source_init():  # noqa: D103
    _ = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=60,
        forecast_minutes=60,
    )


def test_nwp_data_source_open():  # noqa: D103
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=60,
        forecast_minutes=60,
        channels=["t"],
    )

    nwp.open()


def test_nwp_data_source_batch():  # noqa: D103
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=60,
        forecast_minutes=60,
        channels=["t"],
    )

    nwp.open()

    t0_datetimes = [pd.Timestamp(t) for t in nwp._data.init_time[2:6].values]
    x = nwp._data.x_osgb[0:4].values
    y = nwp._data.y_osgb[0:4].values

    locations = []
    for i in range(0, 4):
        locations.append(
            SpaceTimeLocation(
                t0_datetime_utc=t0_datetimes[i], x_center_osgb=x[i], y_center_osgb=y[i]
            )
        )

    batch = nwp.get_batch(locations)

    # batch size 4
    # time series, 1 int he past, 1 now, 1 in the future
    # x,y of size 2
    # channel 1
    assert batch.data.shape == (4, 3, 2, 2, 1)


def test_nwp_data_source_batch_not_on_hour():  # noqa: D103
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=60,
        forecast_minutes=60,
        channels=["t"],
    )

    nwp.open()

    t0_datetimes = [pd.Timestamp("2020-04-01 12:05:00")]
    x = nwp._data.x_osgb[0:1].values
    y = nwp._data.y_osgb[0:1].values

    locations = []
    locations.append(
        SpaceTimeLocation(t0_datetime_utc=t0_datetimes[0], x_center_osgb=x[0], y_center_osgb=y[0])
    )

    batch = nwp.get_batch(locations)

    # batch size 1
    # time series, 1 int he past, 1 now, 1 in the future
    # x,y of size 2
    # channel 1
    assert batch.data.shape == (1, 3, 2, 2, 1)


def test_nwp_get_contiguous_time_periods():  # noqa: D103
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=60,
        forecast_minutes=60,
        channels=["t"],
    )

    contiguous_time_periods = nwp.get_contiguous_time_periods()
    correct_time_periods = pd.DataFrame(
        [{"start_dt": pd.Timestamp("2020-04-01 00:00"), "end_dt": pd.Timestamp("2020-04-02 04:00")}]
    )
    pd.testing.assert_frame_equal(contiguous_time_periods, correct_time_periods)


def test_nwp_get_contiguous_t0_time_periods():  # noqa: D103
    nwp = NWPDataSource(
        zarr_path=NWP_ZARR_PATH,
        history_minutes=60,
        forecast_minutes=60,
        channels=["t"],
    )

    contiguous_time_periods = nwp.get_contiguous_t0_time_periods()
    correct_time_periods = pd.DataFrame(
        [{"start_dt": pd.Timestamp("2020-04-01 01:00"), "end_dt": pd.Timestamp("2020-04-02 03:00")}]
    )
    pd.testing.assert_frame_equal(contiguous_time_periods, correct_time_periods)
