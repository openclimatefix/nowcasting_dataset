""" functions for testing pv live """
import os
import tempfile
from datetime import datetime

import pytest
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models import PVSystem, PVSystemSQL, PVYield
from nowcasting_datamodel.models.pv import Base_PV

"""
This is a bit complicated and sensitive to change
https://gist.github.com/kissgyorgy/e2365f25a213de44b9a2 helped me get going
"""


@pytest.fixture
def db_connection_pv():
    """Create data connection"""

    with tempfile.NamedTemporaryFile(suffix=".db") as temp:
        url = f"sqlite:///{temp.name}"
        os.environ["DB_URL_PV"] = url

        connection = DatabaseConnection(url=url, base=Base_PV)
        Base_PV.metadata.create_all(connection.engine)

        yield connection

        Base_PV.metadata.drop_all(connection.engine)


@pytest.fixture(scope="function", autouse=True)
def db_session_pv(db_connection_pv):
    """Creates a new database session for a test."""

    connection = db_connection_pv.engine.connect()
    t = connection.begin()

    with db_connection_pv.get_session() as s:
        s.begin()
        yield s
        s.rollback()

    t.rollback()
    connection.close()


@pytest.fixture()
def pv_yields_and_systems(db_session_pv):
    """Create pv yields and systems

    Pv systems: Two systems
    PV yields:
        For system 1, pv yields from 4 to 10 at 5 minutes. Last one at 09.55
        For system 2: 1 pv yield at 04.00
    """

    pv_system_sql_1: PVSystemSQL = PVSystem(
        pv_system_id=1,
        provider="pvoutput.org",
        status_interval_minutes=5,
        longitude=0,
        latitude=55,
        installed_capacity_kw=123,
    ).to_orm()
    pv_system_sql_2: PVSystemSQL = PVSystem(
        pv_system_id=2,
        provider="pvoutput.org",
        status_interval_minutes=5,
        longitude=0,
        latitude=56,
        installed_capacity_kw=124,
    ).to_orm()

    pv_yield_sqls = []
    for hour in range(4, 10):
        for minute in range(0, 60, 5):
            pv_yield_1 = PVYield(
                datetime_utc=datetime(2022, 1, 1, hour, minute),
                solar_generation_kw=hour + minute / 100,
            ).to_orm()
            pv_yield_1.pv_system = pv_system_sql_1
            pv_yield_sqls.append(pv_yield_1)

    pv_yield_4 = PVYield(datetime_utc=datetime(2022, 1, 1, 4), solar_generation_kw=4).to_orm()
    pv_yield_4.pv_system = pv_system_sql_2
    pv_yield_sqls.append(pv_yield_4)

    # add to database
    db_session_pv.add_all(pv_yield_sqls)
    db_session_pv.add(pv_system_sql_1)
    db_session_pv.add(pv_system_sql_2)

    db_session_pv.commit()

    return {
        "pv_yields": pv_yield_sqls,
        "pv_systems": [pv_system_sql_1, pv_system_sql_2],
    }
