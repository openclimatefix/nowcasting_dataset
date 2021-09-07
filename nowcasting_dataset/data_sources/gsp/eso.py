import json
import urllib
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd

from nowcasting_dataset.data_sources.gsp.pv_gsp_data_source import logger, WGS84_CRS


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