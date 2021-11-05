import pandas as pd

from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource


def test_init(test_data_folder):
    zarr_path = test_data_folder + "/sun/test.zarr"

    _ = SunDataSource(zarr_path=zarr_path, history_minutes=30, forecast_minutes=60)


def test_get_example(test_data_folder):
    zarr_path = test_data_folder + "/sun/test.zarr"

    sun_data_source = SunDataSource(zarr_path=zarr_path, history_minutes=30, forecast_minutes=60)

    x = 256895.63164759654
    y = 666180.3018829626
    start_dt = pd.Timestamp("2019-01-01 12:00:00.000")

    example = sun_data_source.get_example(t0_dt=start_dt, x_meters_center=x, y_meters_center=y)

    assert len(example.elevation) == 19
    assert len(example.azimuth) == 19


def test_get_example_different_year(test_data_folder):
    zarr_path = test_data_folder + "/sun/test.zarr"

    sun_data_source = SunDataSource(zarr_path=zarr_path, history_minutes=30, forecast_minutes=60)

    x = 256895.63164759654
    y = 666180.3018829626
    start_dt = pd.Timestamp("2021-01-01 12:00:00.000")

    example = sun_data_source.get_example(t0_dt=start_dt, x_meters_center=x, y_meters_center=y)

    assert len(example.elevation) == 19
    assert len(example.azimuth) == 19
