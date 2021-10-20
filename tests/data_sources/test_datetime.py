import pandas as pd

from nowcasting_dataset.data_sources.datetime.datetime_data_source import DatetimeDataSource


def test_datetime_source():
    datetime_source = DatetimeDataSource(
        convert_to_numpy=True,
        forecast_minutes=300,
        history_minutes=10,
    )
    t0_dt = pd.Timestamp("2019-01-01T13:00")
    _ = datetime_source.get_example(t0_dt=t0_dt, x_meters_center=0, y_meters_center=0)


def test_datetime_source_batch():
    datetime_source = DatetimeDataSource(
        convert_to_numpy=True,
        forecast_minutes=300,
        history_minutes=10,
    )
    t0_dt = pd.Timestamp("2019-01-01T13:00")
    _ = datetime_source.get_batch(t0_datetimes=[t0_dt], x_locations=[0], y_locations=[0])
