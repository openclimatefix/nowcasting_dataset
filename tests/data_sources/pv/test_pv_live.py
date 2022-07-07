""" Test for loading pv data from database """
from datetime import datetime, timedelta, timezone

import pandas as pd
from freezegun import freeze_time
from nowcasting_datamodel.models import PVSystem, PVSystemSQL, pv_output

from nowcasting_dataset.config.model import PVFiles
from nowcasting_dataset.data_sources.metadata.metadata_model import SpaceTimeLocation
from nowcasting_dataset.data_sources.pv.live import (
    get_metadata_from_database,
    get_pv_power_from_database,
)
from nowcasting_dataset.data_sources.pv.pv_data_source import PVDataSource
from nowcasting_dataset.data_sources.pv.pv_model import PV
from nowcasting_dataset.geospatial import lat_lon_to_osgb


def test_get_metadata_from_database(pv_yields_and_systems):
    """Test get meteadata from database"""
    meteadata = get_metadata_from_database()

    assert len(meteadata) == 4


@freeze_time("2022-01-01 08:00")
def test_get_pv_power_from_database_no_pv_yields(db_session):
    """Test that nans are return when there are no pv yields in the database"""

    pv_system_sql_1: PVSystemSQL = PVSystem(
        pv_system_id=1,
        provider=pv_output,
        status_interval_minutes=5,
        longitude=0,
        latitude=55,
        ml_capacity_kw=123,
    ).to_orm()
    db_session.add(pv_system_sql_1)
    pv_system_sql_2: PVSystemSQL = PVSystem(
        pv_system_id=2,
        provider=pv_output,
        status_interval_minutes=5,
        longitude=0,
        latitude=55,
        ml_capacity_kw=123,
    ).to_orm()
    db_session.add(pv_system_sql_2)
    db_session.commit()

    """Get pv power from database"""
    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=1),
        load_extra_minutes=30,
        interpolate_minutes=30,
        load_extra_minutes_and_keep=30,
    )

    assert len(pv_power) == 19  # 1.5 hours at 5 mins = 6*5
    assert len(pv_power.columns) == 2
    assert pv_power.columns[0] == "10"
    assert (
        pd.to_datetime(pv_power.index[0]).isoformat()
        == datetime(2022, 1, 1, 6, 30, tzinfo=timezone.utc).isoformat()
    )
    # some values have been filled with 0.0
    assert pv_power.isna().sum().sum() == 22


@freeze_time("2022-01-01 05:00")
def test_get_pv_power_from_database(pv_yields_and_systems):
    """Get pv power from database"""
    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=1),
        load_extra_minutes=30,
        interpolate_minutes=30,
        load_extra_minutes_and_keep=30,
    )

    assert len(pv_power) == 19  # 1.5 hours at 5 mins = 6*12
    assert len(pv_power.columns) == 2
    assert pv_power.columns[0] == "10"
    assert (
        pd.to_datetime(pv_power.index[0]).isoformat()
        == datetime(2022, 1, 1, 3, 30, tzinfo=timezone.utc).isoformat()
    )


@freeze_time("2022-01-01 10:54:59")
def test_get_pv_power_from_database_interpolate(pv_yields_and_systems):
    """Get pv power from database, test out get extra minutes and interpolate"""

    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=0.5),
        load_extra_minutes=0,
        interpolate_minutes=0,
        load_extra_minutes_and_keep=0,
    )
    assert len(pv_power) == 7  # last data point is at 09:55, but we now get nans
    assert pv_power.isna().sum().sum() == 7 * 4

    pv_power = get_pv_power_from_database(
        history_duration=timedelta(hours=1),
        load_extra_minutes=60,
        interpolate_minutes=30,
        load_extra_minutes_and_keep=30,
    )
    assert len(pv_power) == 19  # 1.5 hours at 5 mins = 12
    assert pv_power.isna().sum().sum() == 24  # the last 1 hour is still nans, for 2 pv systems


@freeze_time("2022-01-01 05:00")
def test_get_pv_power_from_database_no_data(db_session):
    """Get pv power from database"""

    # add one system
    pv_system_sql_1: PVSystemSQL = PVSystem(
        pv_system_id=1,
        provider=pv_output,
        status_interval_minutes=5,
        longitude=0,
        latitude=55,
        ml_capacity_kw=123,
    ).to_orm()
    db_session.add(pv_system_sql_1)
    db_session.commit()

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
                label="pvoutput.org",
            )
        ],
        start_datetime=datetime.fromisoformat("2022-04-26 00:00:00.000"),
        end_datetime=datetime.fromisoformat("2022-04-27 00:00:00.000"),
        load_azimuth_and_elevation=False,
        load_from_gcs=False,
        get_center=False,
    )

    x_osgb, y_osgb = lat_lon_to_osgb(lat=55, lon=0)

    assert len(pv_data_source.pv_power) > 0

    location = SpaceTimeLocation(
        t0_datetime_utc=datetime(2022, 1, 1, 5), x_center_osgb=x_osgb, y_center_osgb=y_osgb
    )
    d = PV(pv_data_source.get_batch(locations=[location]))

    PV.validate(d)

    assert d.power_mw.shape == (1, 7, 2048)


@freeze_time("2022-01-01 05:15")
def test_get_example(pv_yields_and_systems):
    """Test PVDataSource with data source from database"""

    pv_data_source = PVDataSource(
        history_minutes=30,
        forecast_minutes=0,
        image_size_pixels_height=64,
        image_size_pixels_width=64,
        meters_per_pixel=2000,
        is_live=True,
        files_groups=[
            PVFiles(
                pv_filename="not needed",
                pv_metadata_filename="not needed",
                label="pvoutput.org",
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
    assert len(locations) == 13  # 1 hour at 5 mins, inclusive,
    # loads slightly more than 30 mins history
    assert (
        pv_data_source.pv_capacity.iloc[0] == pv_yields_and_systems["pv_systems"][0].ml_capacity_kw
    )

    location = locations[0]
    location.t0_datetime_utc = datetime(2022, 1, 1, 5)

    pv = pv_data_source.get_example(location=location)
    assert len(pv.time) == 7
    assert pd.to_datetime(pv.time.values[0]) == pd.to_datetime(datetime(2022, 1, 1, 4, 30))
    assert pd.to_datetime(pv.time.values[-1]) == pd.to_datetime(location.t0_datetime_utc)
