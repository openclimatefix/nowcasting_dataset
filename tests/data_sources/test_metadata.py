import numpy as np
import pandas as pd
import pytest

from nowcasting_dataset.data_sources.metadata.metadata_data_source import MetadataDataSource


def test_metadata_example():
    data_source = MetadataDataSource(history_minutes=0, forecast_minutes=5, object_at_center="GSP")
    t0 = pd.Timestamp("2021-01-01")
    x_meters_center = 1000
    y_meters_center = 1000
    example = data_source.get_example(
        t0_dt=t0, x_meters_center=x_meters_center, y_meters_center=y_meters_center
    )
    assert "t0_dt_index" in example.coords


def test_metadata_batch():
    data_source = MetadataDataSource(history_minutes=0, forecast_minutes=5, object_at_center="GSP")
    t0_datetimes = pd.date_range("2021-01-01", freq="5T", periods=32) + pd.Timedelta("30T")
    x_meters_centers = np.random.random(32)
    y_meters_centers = np.random.random(32)
    batch = data_source.get_batch(
        t0_datetimes=t0_datetimes, x_locations=x_meters_centers, y_locations=y_meters_centers
    )
    assert "t0_dt_index" in batch.coords
