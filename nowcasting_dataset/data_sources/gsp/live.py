""" Function to get data from live database """
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from nowcasting_datamodel.models.gsp import GSPYield, GSPYieldSQL, Location
from nowcasting_datamodel.read.read_gsp import get_gsp_yield

logger = logging.getLogger(__name__)


def get_gsp_power_from_database(
    history_duration: timedelta, interpolate_minutes: int, load_extra_minutes: int
) -> (pd.DataFrame, pd.DataFrame):
    """
    Get gsp power from database

    Args:
        history_duration: a timedelta of how many minutes to load in the past
        interpolate_minutes: how many minutes we should interpolate the data froward for
        load_extra_minutes: the extra minutes we should load, in order to load more data.
            This is because some data from a site lags significantly behind 'now'

    Returns:pandas data frame with the following columns pv systems indexes
    The index is the datetime

    """

    logger.info("Loading GSP data from database")
    logger.debug(f"{history_duration=}")

    extra_duration = timedelta(minutes=load_extra_minutes)
    now = pd.to_datetime(datetime.now(tz=timezone.utc)).ceil("30T")
    start_utc = now - history_duration
    start_utc_extra = start_utc - extra_duration

    # create empty dataframe with 30 mins periods
    empty_df = pd.DataFrame(
        index=pd.date_range(start=start_utc_extra, end=now, freq="30T", tz=timezone.utc)
    )

    # make database connection
    url = os.getenv("DB_URL")
    db_connection = DatabaseConnection(url=url, base=Base_Forecast)

    with db_connection.get_session() as session:
        # We minus 1 second just to make sure we ready that value
        gsp_yields: List[GSPYieldSQL] = get_gsp_yield(
            session=session,
            start_datetime_utc=start_utc_extra - timedelta(seconds=1),
            gsp_ids=list(range(1, 10)),
        )

        logger.debug(f"Found {len(gsp_yields)} GSP yields from the database")

        gsp_yields_dict = []
        for gsp_yield in gsp_yields:
            location = Location.from_orm(gsp_yield.location)
            gsp_yield = GSPYield.from_orm(gsp_yield)

            gsp_yield_dict = gsp_yield.__dict__
            gsp_yield_dict["installed_capacity_mw"] = location.installed_capacity_mw
            gsp_yield_dict["solar_generation_mw"] = gsp_yield_dict["solar_generation_kw"] * 1000
            gsp_yield_dict["gsp_id"] = location.gsp_id
            gsp_yields_dict.append(gsp_yield_dict)

        gsp_yields_df = pd.DataFrame(gsp_yields_dict)

        logger.debug(gsp_yields_df.columns)

    if len(gsp_yields_df) == 0:
        logger.warning("Found no gsp yields, this might cause an error")
    else:
        logger.debug(f"Found {len(gsp_yields_df)} gsp yields")

    if len(gsp_yields_df) == 0:
        return pd.DataFrame(columns=["gsp_id"]), pd.DataFrame(columns=["gsp_id"])

    # pivot on
    gsp_yields_df = gsp_yields_df[
        ["datetime_utc", "gsp_id", "solar_generation_mw", "installed_capacity_mw"]
    ]
    logger.debug(gsp_yields_df.columns)
    gsp_yields_df.drop_duplicates(
        ["datetime_utc", "gsp_id", "solar_generation_mw"], keep="last", inplace=True
    )
    logger.debug(gsp_yields_df.columns)
    gsp_power_df = gsp_yields_df.pivot(
        index="datetime_utc", columns="gsp_id", values="solar_generation_mw"
    )

    gsp_capacity_df = gsp_yields_df.pivot(
        index="datetime_utc", columns="gsp_id", values="installed_capacity_mw"
    )

    logger.debug(f"{empty_df=}")
    logger.debug(f"{gsp_power_df=}")
    gsp_power_df = empty_df.join(gsp_power_df)
    gsp_capacity_df = empty_df.join(gsp_capacity_df)

    # interpolate in between, maximum 'live_interpolate_minutes' mins
    # note data is in 30 minutes chunks
    limit = int(interpolate_minutes / 30)
    if limit > 0:
        gsp_power_df.interpolate(
            limit=limit, inplace=True, method="slinear", fill_value="extrapolate"
        )
        gsp_capacity_df.interpolate(
            limit=limit, inplace=True, method="slinear", fill_value="extrapolate"
        )

    # filter out the extra minutes loaded
    logger.debug(f"{len(gsp_power_df)} of datetimes before filter on {start_utc}")
    gsp_power_df = gsp_power_df[gsp_power_df.index >= start_utc]
    gsp_capacity_df = gsp_capacity_df[gsp_capacity_df.index >= start_utc]
    logger.debug(f"{len(gsp_power_df)} of datetimes after filter on {start_utc}")

    return gsp_power_df, gsp_capacity_df
