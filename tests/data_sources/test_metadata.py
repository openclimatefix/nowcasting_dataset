import pandas as pd
import pytest

from nowcasting_dataset.data_sources.metadata.metadata_data_source import MetadataDataSource


def test_metadata_example():
    data_source = MetadataDataSource(history_minutes=0, forecast_minutes=5, object_at_center="GSP")
    t0 = pd.Timestamp('2021-01-01')
    x_meters_center = 1000
    y_meters_center = 1000
    example = data_source.get_example(
        t0_dt=t0, x_meters_center=x_meters_center, y_meters_center=y_meters_center
    )
    assert "t0_dt_index" in example.coords
