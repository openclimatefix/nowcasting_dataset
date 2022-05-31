""" functions for testing pv live """
from datetime import datetime

import pytest
from nowcasting_datamodel.models import (
    GSPYield,
    Location,
    LocationSQL,
    PVSystem,
    PVSystemSQL,
    PVYield,
)

"""
This is a bit complicated and sensitive to change
https://gist.github.com/kissgyorgy/e2365f25a213de44b9a2 helped me get going
"""


@pytest.fixture()
def pv_yields_and_systems(db_session):
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
    db_session.add_all(pv_yield_sqls)
    db_session.add(pv_system_sql_1)
    db_session.add(pv_system_sql_2)

    db_session.commit()

    return {
        "pv_yields": pv_yield_sqls,
        "pv_systems": [pv_system_sql_1, pv_system_sql_2],
    }


@pytest.fixture()
def gsp_yields(db_session):
    """Make fake GSP data"""

    gsp_sql_1: LocationSQL = Location(gsp_id=1, label="GSP_1", installed_capacity_mw=1).to_orm()

    gsp_yield_sqls = []
    for hour in range(0, 4):
        for minute in range(0, 60, 30):
            gsp_yield_1 = GSPYield(
                datetime_utc=datetime(2022, 1, 1, hour, minute), solar_generation_kw=hour + minute
            )
            gsp_yield_1_sql = gsp_yield_1.to_orm()
            gsp_yield_1_sql.location = gsp_sql_1
            gsp_yield_sqls.append(gsp_yield_1_sql)

    # add to database
    db_session.add_all(gsp_yield_sqls)
    db_session.commit()

    return {
        "gsp_yields": gsp_yield_sqls,
        "gsp_systems": [gsp_sql_1],
    }
