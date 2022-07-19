""" Fixtures for tests """
import os
import tempfile
from datetime import datetime, timedelta

import pytest
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import Base_Forecast, Base_PV, GSPYield, Location, LocationSQL

from nowcasting_dataset.time import floor_minutes_dt


@pytest.fixture()
def gsp_yields_and_systems(db_session):
    """Create gsp yields and systems

    gsp systems: One systems
    GSP yields:
        For system 1, gsp yields from 2 hours ago to 8 in the future at 30 minutes intervals
        For system 2: 1 gsp yield at 16.00
    """

    # this pv systems has same coordiantes as the first gsp
    gsp_yield_sqls = []
    locations = []
    for i in range(317):
        location_sql_1: LocationSQL = Location(
            gsp_id=i + 1,
            label=f"GSP_{i+1}",
            installed_capacity_mw=123.0,
        ).to_orm()

        t0_datetime_utc = floor_minutes_dt(datetime.utcnow()) - timedelta(hours=3)

        gsp_yield_sqls = []
        for hour in range(0, 10):
            for minute in range(0, 60, 30):
                datetime_utc = t0_datetime_utc + timedelta(hours=hour - 2, minutes=minute)
                gsp_yield_1 = GSPYield(
                    datetime_utc=datetime_utc,
                    solar_generation_kw=20 + hour + minute,
                ).to_orm()
                gsp_yield_1.location = location_sql_1
                gsp_yield_sqls.append(gsp_yield_1)
                locations.append(location_sql_1)

    # add to database
    db_session.add_all(gsp_yield_sqls + locations)

    db_session.commit()

    return {
        "gsp_yields": gsp_yield_sqls,
        "gs_systems": locations,
    }


@pytest.fixture
def db_connection():
    """Create data connection"""

    with tempfile.NamedTemporaryFile(suffix=".db") as temp:
        url = f"sqlite:///{temp.name}"
        os.environ["DB_URL_PV"] = url
        os.environ["DB_URL"] = url

        connection = DatabaseConnection(url=url, base=Base_PV)
        Base_PV.metadata.create_all(connection.engine)
        Base_Forecast.metadata.create_all(connection.engine)

        yield connection

        Base_PV.metadata.drop_all(connection.engine)
        Base_Forecast.metadata.create_all(connection.engine)


@pytest.fixture(scope="function", autouse=True)
def db_session(db_connection):
    """Creates a new database session for a test."""

    connection = db_connection.engine.connect()
    t = connection.begin()

    with db_connection.get_session() as s:
        s.begin()
        yield s
        s.rollback()

    t.rollback()
    connection.close()
