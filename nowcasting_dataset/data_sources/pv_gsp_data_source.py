import pandas as pd


import urllib
import json
import pandas as pd
import logging

from pvlive_api import PVLive
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go

_LOG = logging.getLogger(__name__)


def get_pv_gsp_metadata_from_eso() -> pd.DataFrame:
    """
    Get the metadata for the pv gsp, from ESO.
    @return:
    """

    # call ESO website
    url = (
        "https://data.nationalgrideso.com/api/3/action/datastore_search?"
        "resource_id=bbe2cc72-a6c6-46e6-8f4e-48b879467368&limit=400"
    )
    fileobj = urllib.request.urlopen(url)
    d = json.loads(fileobj.read())

    # make dataframe
    results = d["result"]["records"]
    return pd.DataFrame(results)


def load_pv_gsp_raw_data_from_pvlive(start: datetime, end: datetime, number_of_gsp: int = 380) -> pd.DataFrame:
    """
    Load raw pv gsp data from pvline. Note that each gsp is loaded separately. Also the data is loaded in 30 day chunks.
    @param start: the start date for gsp data to load
    @param end: the end date for gsp data to load
    @param number_of_gsp: The number of gsp to load. Note that on 2021-09-01 there were 380 to load.
    @return: Data frame of time series of gsp data. Shows PV data for each GSP from {start} to {end}
    """
    pvl = PVLive()

    # set the first chunk of data, note that 30 day chunks are used accept if the end time is small than that
    first_start_chunk = start
    first_end_chunk = min([first_start_chunk + timedelta(days=30), end])

    one_month_gsp_data_df = []
    for i in range(0, number_of_gsp):

        # set the first chunk start and end times
        start_chunk = first_start_chunk
        end_chunk = first_end_chunk

        while start_chunk <= end:
            _LOG.debug(f"Getting data for id {i} from {start_chunk} to {end_chunk}")
            one_month_gsp_data_df.append(
                pvl.between(
                    start=start_chunk, end=end_chunk, entity_type="gsp", entity_id=i, extra_fields="", dataframe=True
                )
            )

            # add 30 days to the chunk, to get the next chunk
            start_chunk = start_chunk + timedelta(days=30)
            end_chunk = end_chunk + timedelta(days=30)

            if end_chunk > end:
                end_chunk = end

    one_month_gsp_data_df = pd.concat(one_month_gsp_data_df)

    # remove any extra data loaded
    one_month_gsp_data_df = one_month_gsp_data_df[one_month_gsp_data_df["datetime_gmt"] <= end]

    # remove any duplicates
    one_month_gsp_data_df.drop_duplicates(inplace=True)

    # sort dataframe
    one_month_gsp_data_df = one_month_gsp_data_df.sort_values(by=["gsp_id", "datetime_gmt"])

    return one_month_gsp_data_df
