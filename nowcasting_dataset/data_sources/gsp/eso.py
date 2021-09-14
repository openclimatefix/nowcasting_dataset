"""
This file has a few functions that are used to get GSP (Grid Supply Point) information from National Grid ESO.
ESO - Electricity System Operator. General information can be found here
- https://data.nationalgrideso.com/system/gis-boundaries-for-gb-grid-supply-points

get_gsp_metadata_from_eso: gets the gsp metadata
get_gsp_shape_from_eso: gets the shape of the gsp regions
get_list_of_gsp_ids: gets a list of gsp_ids, by using 'get_gsp_metadata_from_eso'

Peter Dudfield
2021-09-13
"""

import json
import urllib
import logging
from typing import List
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

from nowcasting_dataset.geospatial import WGS84_CRS

logger = logging.getLogger(__name__)


def get_gsp_metadata_from_eso() -> pd.DataFrame:
    """
    Get the metadata for the gsp, from ESO.
    @return:
    """

    # call ESO website. There is a possibility that this API will be replaced and its unclear if this original API will
    # will stay operational
    url = (
        "https://data.nationalgrideso.com/api/3/action/datastore_search?"
        "resource_id=bbe2cc72-a6c6-46e6-8f4e-48b879467368&limit=400"
    )
    fileobj = urllib.request.urlopen(url)
    d = json.loads(fileobj.read())

    # make dataframe
    results = d["result"]["records"]
    metadata = pd.DataFrame(results)

    # drop duplicates
    return metadata.drop_duplicates(subset=['gsp_id'])


def get_gsp_shape_from_eso() -> gpd.GeoDataFrame:
    """
    Get the the gsp shape file
    """

    logger.debug('Loading GSP shape file')

    # call ESO website. There is a possibility that this API will be replaced and its unclear if this original API will
    # will stay operational
    url = (
        "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/resource/"
        "a3ed5711-407a-42a9-a63a-011615eea7e0/download/gsp_regions_20181031.geojson"
    )

    with urlopen(url) as response:
        return gpd.read_file(response).to_crs(WGS84_CRS)


def get_list_of_gsp_ids(maximum_number_of_gsp: int) -> List[int]:
    """
    Get list of gsp ids from ESO metadata

    Args:
        maximum_number_of_gsp: Truncate list of GSPs to be no larger than this number of GSPs.
            Set to None to disable truncation.

    Returns:  list of gsp ids

    """

    # get a lit of gsp ids
    metadata = get_gsp_metadata_from_eso()

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