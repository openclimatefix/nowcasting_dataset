""" Test for loading pv data from database """
from datetime import datetime, timedelta, timezone

import pandas as pd
from freezegun import freeze_time

from nowcasting_dataset.config.model import PVFiles
from nowcasting_dataset.data_sources.pv.live import (
    get_metadata_from_database,
    get_pv_power_from_database,
)
from nowcasting_dataset.data_sources.pv.pv_data_source import PVDataSource


def test_get_metadata_from_database(pv_yields_and_systems):
    """Test get meteadata from database"""
    meteadata = get_metadata_from_database()

    assert len(meteadata) == 2


@freeze_time("2022-01-01")
def test_get_pv_power_from_database(pv_yields_and_systems):
    """Get pv power from database"""
    pv_power = get_pv_power_from_database(history_duration=timedelta(hours=1))

    assert len(pv_power) == 72  # 6 hours at 5 mins = 6*12
    assert len(pv_power.columns) == 2
    assert pv_power.columns[0] == "11"
    assert (
        pd.to_datetime(pv_power.index[0]).isoformat()
        == datetime(2022, 1, 1, 4, tzinfo=timezone.utc).isoformat()
    )


@freeze_time("2022-01-01")
def test_get_example_and_batch(pv_yields_and_systems):
    """Test PVDataSource with data source from database"""

    pv_data_source = PVDataSource(
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
        is_live=True,
        files_groups=[
            PVFiles(
                pv_filename="not needed",
                pv_metadata_filename="not needed",
                label="pvoutput",
            )
        ],
        start_datetime=datetime.fromisoformat("2022-04-26 00:00:00.000"),
        end_datetime=datetime.fromisoformat("2022-04-27 00:00:00.000"),
        load_azimuth_and_elevation=False,
        load_from_gcs=False,
    )

    assert len(pv_data_source.pv_power) > 0
    assert len(pv_data_source.pv_metadata) > 0

    locations = pv_data_source.get_locations(pv_data_source.pv_power.index)
    assert len(locations) == 72  # 6 hours at 5 mins

    _ = pv_data_source.get_example(location=locations[0])