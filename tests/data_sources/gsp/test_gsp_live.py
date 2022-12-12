""" Test for loading pv data from database """
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from freezegun import freeze_time
from nowcasting_datamodel.models.gsp import GSPYieldSQL

from nowcasting_dataset.data_sources.gsp.gsp_data_source import GSPDataSource
from nowcasting_dataset.data_sources.gsp.live import get_gsp_power_from_database
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation


@freeze_time("2022-01-01 01:00")
def test_get_pv_power_from_database(gsp_yields, db_session):
    """Get pv power from database"""

    gsp_power, gsp_capacity = get_gsp_power_from_database(
        history_duration=timedelta(hours=1), interpolate_minutes=30, load_extra_minutes=0
    )

    assert len(gsp_power) == 3  # 1 hours at 30 mins + 1
    assert len(gsp_power.columns) == 1
    assert gsp_power.columns[0] == 1
    assert (
        pd.to_datetime(gsp_power.index[0]).isoformat()
        == datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc).isoformat()
    )
    assert gsp_power.max().max() < 1
    # this because units have changed from kw to mw


@freeze_time("2022-01-01 05:00")
def test_get_pv_power_from_database_no_data(db_session):
    """Get gsp power from database"""

    # remove all data points
    db_session.query(GSPYieldSQL).delete()
    db_session.commit()

    gsp_data_source = GSPDataSource(
        history_minutes=30,
        forecast_minutes=60,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
        is_live=True,
        start_datetime=datetime.fromisoformat("2022-04-26 00:00:00.000"),
        end_datetime=datetime.fromisoformat("2022-04-27 00:00:00.000"),
        get_center=False,
        zarr_path="not_needed",
    )

    location = SpaceTimeLocation(
        t0_datetime_utc=datetime(2022, 1, 1, 5), x_center_osgb=1234, y_center_osgb=555
    )
    with pytest.raises(Exception):
        _ = gsp_data_source.get_batch(locations=[location])


@freeze_time("2022-01-01 03:10")
def test_get_example_and_batch(gsp_yields):
    """Test GSPDataSource with data source from database"""

    gsp_data_source = GSPDataSource(
        history_minutes=120,
        forecast_minutes=480,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
        is_live=True,
        start_datetime=datetime.fromisoformat("2022-04-26 00:00:00.000"),
        end_datetime=datetime.fromisoformat("2022-04-27 00:00:00.000"),
        zarr_path="not_needed",
    )

    assert len(gsp_data_source.gsp_power) > 0
    assert len(gsp_data_source.metadata) > 0

    locations = gsp_data_source.get_locations(gsp_data_source.gsp_power.index)
    assert len(locations) == 5  # 120 minutes at 30 mins, inclusive
    assert (
        gsp_data_source.gsp_capacity[1].iloc[0]
        == gsp_yields["gsp_systems"][0].installed_capacity_mw
    )

    location = locations[0]
    location.t0_datetime_utc = datetime(2022, 1, 1, 3, 0, 5, tzinfo=timezone.utc)

    example = gsp_data_source.get_example(location=location)
    assert len(example.time) == 5
    assert example.time.values[-1] == pd.to_datetime(datetime(2022, 1, 1, 3, 30))
