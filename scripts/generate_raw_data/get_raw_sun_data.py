"""
Compute raw sun data using pvlib

2021-09-01
Peter Dudfield

The data is about
 - 1MB for a 2 of days, for ~2000 sites and takes about ~1 minutes
 - 6MB for a 10 of days, for ~2000 sites and takes about ~1 minutes
 - 252MB for a 365 of days, for ~2000 sites and takes about ~11 minutes (on a macbook pro)

Decide to just go for one year of data
on 1st Jan 2019 and 2020, the biggest differences was in elevation was 1 degree,
More investigation has been done (link), and happy difference is less than 1 degree,
Therefore, its ok good to use 1 year of data, for all the years
"""

import logging
import os
from datetime import datetime

import pandas as pd
from pathy import Pathy

import nowcasting_dataset
from nowcasting_dataset.config import load_yaml_configuration
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from nowcasting_dataset.data_sources.sun.raw_data_load_save import (
    get_azimuth_and_elevation,
    save_to_zarr,
)
from nowcasting_dataset.geospatial import lat_lon_to_osgb

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

config_filename = Pathy(nowcasting_dataset.__file__).parent / "config" / "gcp.yaml"
config = load_yaml_configuration(config_filename)


# set up
sun_file_zarr = config.input_data.sun.sun_zarr_path

# set up variables
local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."
start_dt = datetime.fromisoformat("2019-01-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2020-01-01 00:00:00.000+00:00")
datestamps = pd.date_range(start=start_dt, end=end_dt, freq="5T")

# PV metadata
pv_metadatas = []
for pv_files in config.input_data.pv.pv_files_groups:
    metadata_filename = pv_files.pv_metadata_filename
    pv_metadata = pd.read_csv(metadata_filename, index_col="system_id")
    pv_metadata = pv_metadata.dropna(subset=["longitude", "latitude"])
    pv_metadata["location_x"], pv_metadata["location_y"] = lat_lon_to_osgb(
        pv_metadata["latitude"], pv_metadata["longitude"]
    )
    pv_metadatas.append(pv_metadata)

pv_metadata = pd.concat(pv_metadatas)
pv_x = pv_metadata["location_x"]
pv_y = pv_metadata["location_y"]

# GSP Metadata
gsp_metadata = get_gsp_metadata_from_eso()
gsp_metadata = gsp_metadata.dropna(subset=["centroid_lon", "centroid_lat"])
gsp_x = gsp_metadata["centroid_x"]
gsp_y = gsp_metadata["centroid_y"]

# join all sites together
x_centers = list(pv_x.values) + list(gsp_x.values)
y_centers = list(pv_y.values) + list(gsp_y.values)

# make d
azimuth, elevation = get_azimuth_and_elevation(
    x_centers=x_centers, y_centers=y_centers, datestamps=datestamps
)

# save it locally and in the cloud, just in case when saving in the cloud it fails
save_to_zarr(azimuth=azimuth, elevation=elevation, zarr_path="./sun.zarr")
save_to_zarr(azimuth=azimuth, elevation=elevation, zarr_path=sun_file_zarr)
# This has been uploaded to 'gs://solar-pv-nowcasting-data/Sun/v0'
