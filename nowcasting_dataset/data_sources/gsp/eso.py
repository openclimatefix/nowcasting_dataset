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
import os
import urllib
import logging
from typing import List, Optional
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

from nowcasting_dataset.geospatial import WGS84_CRS

logger = logging.getLogger(__name__)


def get_gsp_metadata_from_eso(calculate_centroid: bool = True) -> pd.DataFrame:
    """
    Get the metadata for the gsp, from ESO.
    Args:
        calculate_centroid: Load the shape file also, and calculate the Centroid

    Returns:

    """

    logger.debug("Getting GSP shape file")

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
    metadata = metadata.drop_duplicates(subset=["gsp_id"])

    if calculate_centroid:
        # get shape data from eso
        shape_data = get_gsp_shape_from_eso()

        # join data together
        metadata = gpd.GeoDataFrame(
            metadata.merge(shape_data, right_on="RegionID", left_on="region_id", how="left")
        )

        # make centroid
        metadata["centroid_x"] = metadata["geometry"].centroid.x
        metadata["centroid_y"] = metadata["geometry"].centroid.y

    return metadata


def get_gsp_shape_from_eso(
    join_duplicates: bool = True, load_local_file: bool = True, save_local_file: bool = False
) -> gpd.GeoDataFrame:
    """
    Get the the gsp shape file from ESO (or a local file)
    Args:
        join_duplicates: If True, any RegionIDs which have multiple entries, will be joined together to give one entry
        load_local_file: Load from a local file, not from ESO
        save_local_file: Save to a local file, only need to do this is Data is updated.

    Returns: Geo Pandas dataframe of GSP shape data
    """

    logger.debug("Loading GSP shape file")

    local_file = f"{os.path.dirname(os.path.realpath(__file__))}/gsp_shape"

    if not os.path.isfile(local_file):
        logger.debug("There is no local file so going to get it from ESO, and save it afterwards")
        load_local_file = False
        save_local_file = True

    if load_local_file:
        logger.debug("loading local file for GSP shape data")
        shape_gpd = gpd.read_file(local_file)
        logger.debug("loading local file for GSP shape data:done")
    else:
        # call ESO website. There is a possibility that this API will be replaced and its unclear if this original API will
        # will stay operational
        url = (
            "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/resource/"
            "a3ed5711-407a-42a9-a63a-011615eea7e0/download/gsp_regions_20181031.geojson"
        )

        with urlopen(url) as response:
            shape_gpd = gpd.read_file(response).to_crs(WGS84_CRS)

    if save_local_file:
        shape_gpd.to_file(local_file)

    # sort
    shape_gpd = shape_gpd.sort_values(by=["RegionID"])

    # join duplicates, currently some GSP shapes are in split into two
    if join_duplicates:
        logger.debug("Removing duplicates by joining geometry together")

        shape_gpd_no_duplicates = shape_gpd.drop_duplicates(subset=["RegionID"])
        duplicated_raw = shape_gpd[shape_gpd["RegionID"].duplicated()]

        for _, duplicate in duplicated_raw.iterrows():
            # find index in data set with no duplicates
            index_other = shape_gpd_no_duplicates[
                shape_gpd_no_duplicates["RegionID"] == duplicate.RegionID
            ].index

            # join geometries together
            new_geometry = shape_gpd_no_duplicates.loc[index_other]["geometry"].union(
                duplicate.geometry
            )

            # set new geometry
            shape_gpd_no_duplicates.loc[index_other]["geometry"] = new_geometry

        shape_gpd = shape_gpd_no_duplicates

    return shape_gpd


def get_list_of_gsp_ids(maximum_number_of_gsp: Optional[int] = None) -> List[int]:
    """
    Get list of gsp ids from ESO metadata

    Args:
        maximum_number_of_gsp: Truncate list of GSPs to be no larger than this number of GSPs.
            Set to None to disable truncation.

    Returns:  list of gsp ids

    """

    # get a lit of gsp ids
    metadata = get_gsp_metadata_from_eso(calculate_centroid=False)

    # get rid of nans, and duplicates
    metadata = metadata[~metadata["gsp_id"].isna()]
    metadata.drop_duplicates(subset=["gsp_id"], inplace=True)

    # make into list
    gsp_ids = metadata["gsp_id"].to_list()
    gsp_ids = [int(gsp_id) for gsp_id in gsp_ids]

    # adjust number of gsp_ids
    if maximum_number_of_gsp is None:
        maximum_number_of_gsp = len(metadata)
    if maximum_number_of_gsp > len(metadata):
        logging.warning(f"Only {len(metadata)} gsp available to load")
    if maximum_number_of_gsp < len(metadata):
        gsp_ids = gsp_ids[0:maximum_number_of_gsp]

    return gsp_ids
