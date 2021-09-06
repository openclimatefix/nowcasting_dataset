import urllib
import json
import io
import gcsfs
import logging
import zarr
from urllib.request import urlopen
import geopandas as gpd

import pandas as pd
import numpy as np
import xarray as xr

from pvlive_api import PVLive
from typing import List, Union, Optional
from pathlib import Path
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)

WGS84_CRS = "EPSG:4326"


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


def get_pv_gsp_shape() -> gpd.GeoDataFrame:
    """
    Get the the gsp shape file
    """

    logger.debug('Loading GSP shape file')

    url = (
        "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/resource/"
        "a3ed5711-407a-42a9-a63a-011615eea7e0/download/gsp_regions_20181031.geojson"
    )

    with urlopen(url) as response:
        return gpd.read_file(response).to_crs(WGS84_CRS)


def get_list_of_gsp_ids(maximum_number_of_gsp: int) -> List[int]:
    """
    Get list of gsp ids from ESO metadata
    @param maximum_number_of_gsp: clib list by this amount.
    @return: list of gsp ids
    """
    # get a lit of gsp ids
    metadata = get_pv_gsp_metadata_from_eso()

    # get rid of nans, and duplicates
    metadata = metadata[~metadata['gsp_id'].isna()]
    metadata.drop_duplicates(subset=['gsp_id'], inplace=True)

    # make into list
    gsp_ids = metadata['gsp_id'].to_list()
    gsp_ids = [int(gsp_id) for gsp_id in gsp_ids]

    # adjust number of gsp_ids
    if maximum_number_of_gsp is None:
        maximum_number_of_gsp = len(metadata)
    if maximum_number_of_gsp > len(metadata):
        logging.warning(f'Only {len(metadata)} gsp available to load')
    if maximum_number_of_gsp < len(metadata):
        gsp_ids = gsp_ids[0: maximum_number_of_gsp]

    return gsp_ids


def load_pv_gsp_raw_data_from_pvlive(start: datetime, end: datetime, number_of_gsp: int = None) -> pd.DataFrame:
    """
    Load raw pv gsp data from pvline. Note that each gsp is loaded separately. Also the data is loaded in 30 day chunks.
    @param start: the start date for gsp data to load
    @param end: the end date for gsp data to load
    @param number_of_gsp: The number of gsp to load. Note that on 2021-09-01 there were 338 to load.
    @return: Data frame of time series of gsp data. Shows PV data for each GSP from {start} to {end}
    """

    # get a lit of gsp ids
    gsp_ids = get_list_of_gsp_ids(maximum_number_of_gsp=number_of_gsp)

    # setup pv Live class, although here we are getting historic data
    pvl = PVLive()

    # set the first chunk of data, note that 30 day chunks are used accept if the end time is small than that
    first_start_chunk = start
    first_end_chunk = min([first_start_chunk + timedelta(days=30), end])

    gsp_data_df = []
    logger.debug(f'Will be getting data for {len(gsp_ids)} gsp ids')
    for gsp_id in gsp_ids:

        one_gsp_data_df = []

        # set the first chunk start and end times
        start_chunk = first_start_chunk
        end_chunk = first_end_chunk

        while start_chunk <= end:
            logger.debug(f"Getting data for gsp id {gsp_id} from {start_chunk} to {end_chunk}")

            one_gsp_data_df.append(
                pvl.between(
                    start=start_chunk, end=end_chunk, entity_type="gsp", entity_id=gsp_id, extra_fields="", dataframe=True
                )
            )

            # add 30 days to the chunk, to get the next chunk
            start_chunk = start_chunk + timedelta(days=30)
            end_chunk = end_chunk + timedelta(days=30)

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
    gsp_data_df['datetime_gmt'] = gsp_data_df['datetime_gmt'].dt.tz_localize(None)

    return gsp_data_df


def load_solar_pv_gsp_data_from_gcs(
        filename: Union[str, Path],
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None) -> pd.DataFrame:
    """
    Load solar pv gsp data from gcs (although there is an option to load from local - for testing)
    @param filename: filename of file to be loaded, can put 'gs://' files in here too
    @param start_dt: the start datetime, which to trim the data to
    @param end_dt: the end datetime, which to trim the data to
    @return: dataframe of pv data
    """
    logger.debug('Loading Solar PV GCP Data from GCS')
    # Open data - it maye be quicker to open byte file first, but decided just to keep it like this at the moment
    pv_power = xr.open_zarr(filename)

    pv_power = pv_power.sel(datetime_gmt=slice(start_dt, end_dt))
    pv_power_df = pv_power.to_dataframe()

    # Save memory
    del pv_power

    # Process the data a little
    pv_power_df = pv_power_df.dropna(axis='columns', how='all')
    pv_power_df = pv_power_df.clip(lower=0, upper=5E7)

    # make column names ints, not strings
    pv_power_df.columns = [int(col) for col in pv_power_df.columns]

    return pv_power_df
