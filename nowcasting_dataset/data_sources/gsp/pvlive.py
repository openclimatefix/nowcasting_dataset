from datetime import datetime, timedelta
import logging
import pandas as pd
from pvlive_api import PVLive

from nowcasting_dataset.data_sources.gsp.eso import get_list_of_gsp_ids

logger = logging.getLogger(__name__)

CHUNK_DURATION = timedelta(days=30)


def load_pv_gsp_raw_data_from_pvlive(
    start: datetime, end: datetime, number_of_gsp: int = None
) -> pd.DataFrame:
    """
    Load raw pv gsp data from pvlive. Note that each gsp is loaded separately. Also the data is loaded in 30 day chunks.
    Args:
        start: the start date for gsp data to load
        end: the end date for gsp data to load
        number_of_gsp: The number of gsp to load. Note that on 2021-09-01 there were 338 to load.

    Returns: Data frame of time series of gsp data. Shows PV data for each GSP from {start} to {end}

    """

    # get a lit of gsp ids
    gsp_ids = get_list_of_gsp_ids(maximum_number_of_gsp=number_of_gsp)

    # setup pv Live class, although here we are getting historic data
    pvl = PVLive()

    # set the first chunk of data, note that 30 day chunks are used except if the end time is smaller than that
    first_start_chunk = start
    first_end_chunk = min([first_start_chunk + CHUNK_DURATION, end])

    gsp_data_df = []
    logger.debug(f"Will be getting data for {len(gsp_ids)} gsp ids")
    # loop over gsp ids
    for gsp_id in gsp_ids:

        one_gsp_data_df = []

        # set the first chunk start and end times
        start_chunk = first_start_chunk
        end_chunk = first_end_chunk

        # loop over 30 days chunks (nice to see progress instead of waiting a long time for one command - this might
        # not be the fastest)
        while start_chunk <= end:
            logger.debug(f"Getting data for gsp id {gsp_id} from {start_chunk} to {end_chunk}")

            one_gsp_data_df.append(
                pvl.between(
                    start=start_chunk,
                    end=end_chunk,
                    entity_type="gsp",
                    entity_id=gsp_id,
                    extra_fields="",
                    dataframe=True,
                )
            )

            # add 30 days to the chunk, to get the next chunk
            start_chunk = start_chunk + CHUNK_DURATION
            end_chunk = end_chunk + CHUNK_DURATION

            if end_chunk > end:
                end_chunk = end

        # join together one gsp data, and sort
        one_gsp_data_df = pd.concat(one_gsp_data_df)
        one_gsp_data_df = one_gsp_data_df.sort_values(by=["gsp_id", "datetime_gmt"])

        # append to longer list
        gsp_data_df.append(one_gsp_data_df)

    gsp_data_df = pd.concat(gsp_data_df)

    # remove any extra data loaded
    gsp_data_df = gsp_data_df[gsp_data_df["datetime_gmt"] <= end]

    # remove any duplicates
    gsp_data_df.drop_duplicates(inplace=True)

    # format data, remove timezone,
    gsp_data_df["datetime_gmt"] = gsp_data_df["datetime_gmt"].dt.tz_localize(None)

    return gsp_data_df
