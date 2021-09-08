import json
import urllib
import logging
from typing import List
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

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


def get_pv_gsp_shape_from_eso() -> gpd.GeoDataFrame:
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