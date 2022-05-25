""" Test for Sun data source """
import pandas as pd

from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource


def test_init(test_data_folder):  # noqa 103
    zarr_path = test_data_folder + "/sun/test.zarr"

    sun = SunDataSource(zarr_path=zarr_path, history_minutes=30, forecast_minutes=60)
    _ = sun.datetime_index()


def test_get_example(test_data_folder):  # noqa 103
    zarr_path = test_data_folder + "/sun/test.zarr"

    sun_data_source = SunDataSource(zarr_path=zarr_path, history_minutes=30, forecast_minutes=60)

    x = 256895.63164759654
    y = 666180.3018829626
    start_dt = pd.Timestamp("2020-04-01 12:00:00.000")

    example = sun_data_source.get_example(
        SpaceTimeLocation(t0_datetime_utc=start_dt, x_center_osgb=x, y_center_osgb=y)
    )

    assert len(example.elevation) == 19
    assert len(example.azimuth) == 19


def test_get_example_different_year(test_data_folder):  # noqa 103
    zarr_path = test_data_folder + "/sun/test.zarr"

    sun_data_source = SunDataSource(zarr_path=zarr_path, history_minutes=30, forecast_minutes=60)

    x = 256895.63164759654
    y = 666180.3018829626
    start_dt = pd.Timestamp("2021-04-01 12:00:00.000")

    example = sun_data_source.get_example(
        location=SpaceTimeLocation(t0_datetime_utc=start_dt, x_center_osgb=x, y_center_osgb=y)
    )

    assert len(example.elevation) == 19
    assert len(example.azimuth) == 19


def test_get_load_live():  # noqa 103

    sun_data_source = SunDataSource(
        zarr_path="", history_minutes=30, forecast_minutes=60, load_live=True
    )
    _ = sun_data_source.datetime_index()

    x = 256895.63164759654
    y = 666180.3018829626
    start_dt = pd.Timestamp("2021-04-01 12:00:00.000")

    example = sun_data_source.get_example(
        location=SpaceTimeLocation(t0_datetime_utc=start_dt, x_center_osgb=x, y_center_osgb=y)
    )

    assert len(example.elevation) == 19
    assert len(example.azimuth) == 19
