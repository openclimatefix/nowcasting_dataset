from nowcasting_dataset.data_sources.sun.sun_data_source import SunDataSource
from datetime import datetime
from nowcasting_dataset.dataset.example import Example
from nowcasting_dataset.consts import SUN_ELEVATION_ANGLE, SUN_AZIMUTH_ANGLE
import pandas as pd


def test_init(test_data_folder):
    filename = test_data_folder + "/sun/test.zarr"

    _ = SunDataSource(
        filename=filename, history_minutes=30, forecast_minutes=60, convert_to_numpy=True
    )


def test_get_example(test_data_folder):
    filename = test_data_folder + "/sun/test.zarr"

    sun_data_source = SunDataSource(
        filename=filename, history_minutes=30, forecast_minutes=60, convert_to_numpy=True
    )

    lon = 50.14
    lat = -5.08
    start_dt = pd.Timestamp("2019-01-01 12:00:00.000")

    example = sun_data_source.get_example(t0_dt=start_dt, x_meters_center=lon, y_meters_center=lat)

    assert SUN_ELEVATION_ANGLE in example.keys()
    assert SUN_AZIMUTH_ANGLE in example.keys()
    assert len(example[SUN_ELEVATION_ANGLE]) == 19
    assert len(example[SUN_AZIMUTH_ANGLE]) == 19
