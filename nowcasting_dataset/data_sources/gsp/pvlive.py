""" Functions used to query the PVlive api """
import logging
from concurrent import futures
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import pytz
from pvlive_api import PVLive
from tqdm import tqdm

from nowcasting_dataset.data_sources.gsp.eso import get_list_of_gsp_ids

logger = logging.getLogger(__name__)

CHUNK_DURATION = timedelta(days=30)


def load_pv_gsp_raw_data_from_pvlive(
    start: datetime, end: datetime, number_of_gsp: int = None, normalize_data: bool = True
) -> pd.DataFrame:
    """
    Load raw pv gsp data from pvlive.

    Note that each gsp is loaded separately. Also the data is loaded in 30 day chunks.

    Args:
        start: the start date for gsp data to load
        end: the end date for gsp data to load
        number_of_gsp: The number of gsp to load. Note that on 2021-09-01 there were 338 to load.
        normalize_data: Option to normalize the generation according to installed capacity

    Returns: Data frame of time series of gsp data. Shows PV data for each GSP from {start} to {end}

    """
    # get a lit of gsp ids
    gsp_ids = get_list_of_gsp_ids(maximum_number_of_gsp=number_of_gsp)

    # setup pv Live class, although here we are getting historic data
    pvl = PVLive()

    # set the first chunk of data, note that 30 day chunks are used except if the end time is
    # smaller than that
    first_start_chunk = start
    first_end_chunk = min([first_start_chunk + CHUNK_DURATION, end])

    gsp_data_df = []
    logger.debug(f"Will be getting data for {len(gsp_ids)} gsp ids")
    # loop over gsp ids
    # limit the total number of concurrent tasks to be 4, so that we don't hit the pvlive api
    # too much
    future_tasks = []
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        for gsp_id in gsp_ids:

            # set the first chunk start and end times
            start_chunk = first_start_chunk
            end_chunk = first_end_chunk

            # loop over 30 days chunks (nice to see progress instead of waiting a long time for
            # one command - this might not be the fastest)
            while start_chunk <= end:
                logger.debug(f"Getting data for gsp id {gsp_id} from {start_chunk} to {end_chunk}")

                task = executor.submit(
                    pvl.between,
                    start=start_chunk,
                    end=end_chunk,
                    entity_type="gsp",
                    entity_id=gsp_id,
                    extra_fields="installedcapacity_mwp",
                    dataframe=True,
                )

                future_tasks.append(task)

                # add 30 days to the chunk, to get the next chunk
                start_chunk = start_chunk + CHUNK_DURATION
                end_chunk = end_chunk + CHUNK_DURATION

                if end_chunk > end:
                    end_chunk = end

        logger.debug("Getting results")
        # Collect results from each thread.
        for task in tqdm(future_tasks):
            one_chunk_one_gsp_gsp_data_df = task.result()

            if normalize_data:
                one_chunk_one_gsp_gsp_data_df["generation_mw"] = (
                    one_chunk_one_gsp_gsp_data_df["generation_mw"]
                    / one_chunk_one_gsp_gsp_data_df["installedcapacity_mwp"]
                )

            # append to longer list
            gsp_data_df.append(one_chunk_one_gsp_gsp_data_df)

    # join together gsp data
    gsp_data_df = pd.concat(gsp_data_df)

    # sort
    gsp_data_df = gsp_data_df.sort_values(by=["gsp_id", "datetime_gmt"])

    # remove any extra data loaded
    gsp_data_df = gsp_data_df[gsp_data_df["datetime_gmt"] <= end]

    # remove any duplicates
    gsp_data_df.drop_duplicates(inplace=True)

    # format data, remove timezone,
    gsp_data_df["datetime_gmt"] = gsp_data_df["datetime_gmt"].dt.tz_localize(None)

    return gsp_data_df


def get_installed_capacity(
    start: Optional[datetime] = datetime(2021, 1, 1, tzinfo=pytz.utc),
    maximum_number_of_gsp: Optional[int] = None,
) -> pd.Series:
    """
    Get the installed capacity of each gsp

    This can take ~30 seconds for getting the full list

    Args:
        start: optional datetime when the installed cpapcity is collected
        maximum_number_of_gsp: Truncate list of GSPs to be no larger than this number of GSPs.
            Set to None to disable truncation.

    Returns: pd.Series of installed capacity indexed by gsp_id

    """
    logger.debug(f"Getting all installed capacity at {start}")

    # get a lit of gsp ids
    gsp_ids = get_list_of_gsp_ids(maximum_number_of_gsp=maximum_number_of_gsp)

    # setup pv Live class, although here we are getting historic data
    pvl = PVLive()

    # loop over gsp_id to get installed capacity
    data = []
    for gsp_id in gsp_ids:
        d = pvl.at_time(
            start,
            entity_type="gsp",
            extra_fields="installedcapacity_mwp",
            dataframe=True,
            entity_id=gsp_id,
        )
        data.append(d)

    # join data together
    data_df = pd.concat(data)

    # set gsp_id as index
    data_df.set_index("gsp_id", inplace=True)

    return data_df["installedcapacity_mwp"]
