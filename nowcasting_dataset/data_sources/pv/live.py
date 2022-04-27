""" Function to get data from live database """
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_PV
from nowcasting_datamodel.models.pv import PVSystem, PVSystemSQL, PVYield, PVYieldSQL
from nowcasting_datamodel.read.read_pv import get_pv_systems, get_pv_yield

from nowcasting_dataset.data_sources.pv.utils import encode_label

logger = logging.getLogger(__name__)


def get_metadata_from_database() -> pd.DataFrame:
    """
    Get metadata from database

    Returns: pandas data frame with the following columns
        - latitude
        - longitude
        - kwp
        The index is the pv_system_id
    """

    # make database connection
    url = os.getenv("DB_URL_PV")
    db_connection = DatabaseConnection(url=url, base=Base_PV)

    with db_connection.get_session() as session:
        # read pv systems
        pv_systems: List[PVSystemSQL] = get_pv_systems(session=session)

        # format locations
        pv_systems_df = pd.DataFrame(
            [(PVSystem.from_orm(pv_system)).__dict__ for pv_system in pv_systems]
        )

    pv_systems_df.index = encode_label(pv_systems_df["pv_system_id"], label="pvoutput")
    pv_systems_df = pv_systems_df[["latitude", "longitude"]]

    return pv_systems_df


def get_pv_power_from_database(history_duration: timedelta) -> pd.DataFrame:
    """
    Get pv power from database

    Returns: pandas data frame with the following columns
    - pv systems indexes
    The index is the datetime

    """

    # make database connection
    url = os.getenv("DB_URL_PV")
    db_connection = DatabaseConnection(url=url, base=Base_PV)

    with db_connection.get_session() as session:
        start_utc = datetime.now(tz=timezone.utc) - history_duration
        pv_yields: List[PVYieldSQL] = get_pv_yield(session=session, start_utc=start_utc)

        pv_yields_df = pd.DataFrame(
            [(PVYield.from_orm(pv_yield)).__dict__ for pv_yield in pv_yields]
        )

    if len(pv_yields_df):
        logger.warning("Found no pv yields, this might cause an error")
    else:
        logger.debug(f"Found {len(pv_yields_df)} pv yields")

    # get the system id from 'pv_system_id=xxxx provider=.....'
    print(pv_yields_df.columns)
    print(pv_yields_df["pv_system"])
    pv_yields_df["pv_system_id"] = (
        pv_yields_df["pv_system"].astype(str).str.split(" ").str[0].str.split("=").str[-1]
    )

    # pivot on
    pv_yields_df = pv_yields_df[["datetime_utc", "pv_system_id", "solar_generation_kw"]]
    pv_yields_df.drop_duplicates(
        ["datetime_utc", "pv_system_id", "solar_generation_kw"], keep="last", inplace=True
    )
    pv_yields_df = pv_yields_df.pivot(
        index="datetime_utc", columns="pv_system_id", values="solar_generation_kw"
    )

    pv_yields_df.columns = encode_label(pv_yields_df.columns, label="pvoutput")

    # interpolate in between, maximum 30 mins
    pv_yields_df.interpolate(limit=3, limit_area="inside", inplace=True)

    return pv_yields_df
