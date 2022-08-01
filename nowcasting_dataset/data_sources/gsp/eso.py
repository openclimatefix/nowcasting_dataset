"""
This file has a few functions that are used to get GSP (Grid Supply Point) information

The info comes from National Grid ESO.

ESO - Electricity System Operator. General information can be found here
- https://data.nationalgrideso.com/system/gis-boundaries-for-gb-grid-supply-points

get_gsp_metadata_from_eso: gets the gsp metadata
get_gsp_shape_from_eso: gets the shape of the gsp regions

Peter Dudfield
2021-09-13
"""

import logging
import os
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

from nowcasting_dataset.data_sources.gsp.pvlive import get_list_of_gsp_ids
from nowcasting_dataset.geospatial import osgb_to_lat_lon

logger = logging.getLogger(__name__)

# When saving a file, the columns need to be less than 10 characters -
# - https://github.com/geopandas/geopandas/issues/1417
# - https://en.wikipedia.org/wiki/Shapefile#Limitations
rename_save_columns = {
    "centroid_x": "cen_x",
    "centroid_y": "cen_y",
    "centroid_lat": "cen_lat",
    "centroid_lon": "cen_lon",
}
rename_load_columns = {v: k for k, v in rename_save_columns.items()}


def get_gsp_metadata_from_eso(
    calculate_centroid: bool = True, load_local_file: bool = True, save_local_file: bool = False
) -> pd.DataFrame:
    """
    Get the metadata for the gsp, from ESO.

    Args:
        calculate_centroid: Load the shape file also, and calculate the Centroid
        load_local_file: Load from a local file, not from ESO
        save_local_file: Save to a local file, only need to do this is Data is updated.

    Returns: Dataframe of ESO Metadata

    """
    logger.debug("Getting GSP shape file")

    local_file = f"{os.path.dirname(os.path.realpath(__file__))}/eso_metadata.csv"

    if not os.path.isfile(local_file):
        logger.debug("There is no local file so going to get it from ESO, and save it afterwards")
        load_local_file = False
        save_local_file = True

    if load_local_file:
        logger.debug("loading local file for ESO metadata")
        metadata = pd.read_csv(local_file)
        # rename the columns to full name
        logger.debug("loading local file for ESO metadata:done")
    else:
        # we now get this from pvlive
        metadata = get_list_of_gsp_ids(return_dataframe=True, return_national=False)

        # drop duplicates
        metadata = metadata.drop_duplicates(subset=["gsp_id"])

        # drop any nans in the gsp is column, and drop National row
        metadata = metadata[metadata["gsp_id"].notnull()]
        metadata = metadata[metadata["gsp_id"] > 0]

        # add in region id
        metadata["region_id"] = metadata["gsp_id"]

    if save_local_file:
        # save file
        metadata.to_csv(local_file)

    if calculate_centroid:
        # get shape data from eso
        shape_data = get_gsp_shape_from_eso()

        # join data together
        metadata = gpd.GeoDataFrame(
            metadata.merge(shape_data, right_on="RegionID", left_on="region_id", how="left")
        )

        assert "centroid_lat" in metadata.keys(), (
            "GSP Metadata does not contain centroid_lat, "
            'try deleting the folder "nowcasting_dataset/data_sources/gsp/gsp_shape"'
        )

        assert "centroid_lon" in metadata.keys(), (
            "GSP Metadata does not contain centroid_lon, "
            'try deleting the folder "nowcasting_dataset/data_sources/gsp/gsp_shape"'
        )

    return metadata


def get_gsp_shape_from_eso(
    join_duplicates: bool = True, load_local_file: bool = True, save_local_file: bool = False
) -> gpd.GeoDataFrame:
    """
    Get the the gsp shape file from ESO (or a local file)

    Args:
        join_duplicates: If True, any RegionIDs which have multiple entries, will be joined
            together to give one entry.
        load_local_file: Load from a local file, not from ESO
        save_local_file: Save to a local file, only need to do this is Data is updated.

    Returns: Geo Pandas dataframe of GSP shape data
    """
    logger.debug("Loading GSP shape file")

    local_file = f"{os.path.dirname(os.path.realpath(__file__))}/gsp_shape"

    if not os.path.isdir(local_file):
        logger.debug("There is no local file so going to get it from ESO, and save it afterwards")
        load_local_file = False
        save_local_file = True

    if load_local_file:
        logger.debug("loading local file for GSP shape data")
        shape_gpd = gpd.read_file(local_file)
        # rename the columns to full name
        shape_gpd.rename(columns=rename_load_columns, inplace=True)
        logger.debug("loading local file for GSP shape data:done")
    else:
        # call ESO website. There is a possibility that this API will be replaced and its unclear
        # if this original API will stay operational.
        url = (
            "https://data.nationalgrideso.com/backend/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/"
            "resource/08534dae-5408-4e31-8639-b579c8f1c50b/download/gsp_regions_20220314.geojson"
        )

        with urlopen(url) as response:
            shape_gpd = gpd.read_file(response)

            # calculate the centroid before using - to_crs
            shape_gpd["centroid_x"] = shape_gpd["geometry"].centroid.x
            shape_gpd["centroid_y"] = shape_gpd["geometry"].centroid.y
            shape_gpd["centroid_lat"], shape_gpd["centroid_lon"] = osgb_to_lat_lon(
                x=shape_gpd["centroid_x"], y=shape_gpd["centroid_y"]
            )

            # Decided not project the shape data to WGS84, as we want to keep
            # all 'batch' data the same projection.
            # However when plotting it may be useful to project to WGS84
            # i.e shape_gpd = shape_gpd.to_crs(WGS84_CRS)

            # TODO is this right?
            # latest geo json does not have region id in it, so add this for the moment
            shape_gpd.sort_values("GSPs", inplace=True)
            shape_gpd.reset_index(inplace=True, drop=True)
            shape_gpd["RegionID"] = range(1, len(shape_gpd) + 1)

    if save_local_file:

        # rename the columns to less than 10 characters
        shape_gpd_to_save = shape_gpd.copy()
        shape_gpd_to_save.rename(columns=rename_save_columns, inplace=True)

        # save file
        shape_gpd_to_save.to_file(local_file)

    # sort
    shape_gpd = shape_gpd.sort_values(by=["RegionID"])

    # join duplicates, currently some GSP shapes are in split into two
    if join_duplicates:
        logger.debug("Removing duplicates by joining geometry together")

        shape_gpd_no_duplicates = shape_gpd.drop_duplicates(subset=["GSPs"]).copy()
        duplicated_raw = shape_gpd[shape_gpd["GSPs"].duplicated()]

        for _, duplicate in duplicated_raw.iterrows():
            # find index in data set with no duplicates
            index_other = shape_gpd_no_duplicates[
                shape_gpd_no_duplicates["GSPs"] == duplicate.GSPs
            ].index

            # join geometries together
            new_geometry = shape_gpd_no_duplicates.loc[index_other, "geometry"].union(
                duplicate.geometry
            )
            # set new geometry
            shape_gpd_no_duplicates.loc[index_other, "geometry"] = new_geometry

        shape_gpd = shape_gpd_no_duplicates

        # sort after removing duplicates
        shape_gpd.sort_values("GSPs", inplace=True)
        shape_gpd.reset_index(inplace=True, drop=True)
        shape_gpd["RegionID"] = range(1, len(shape_gpd) + 1)

    return shape_gpd
