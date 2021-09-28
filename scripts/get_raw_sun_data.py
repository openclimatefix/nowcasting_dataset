############
# Pull raw sun data using pvlib
#
# 2021-09-01
# Peter Dudfield
#
# The data is about 21MB for a 2 of days, for ~3000 sites and takes about 2 minutes
############

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from datetime import datetime
from pathlib import Path
import pandas as pd
import os
import nowcasting_dataset
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from nowcasting_dataset.data_sources.sun.raw_data_load_save import (
    save_to_zarr,
    get_azimuth_and_elevation,
)

# set up
BUCKET = Path("solar-pv-nowcasting-data")
PV_PATH = BUCKET / "PV/PVOutput.org"
PV_METADATA_FILENAME = PV_PATH / "UK_PV_metadata.csv"

# set up variables
local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."
metadata_filename = f"gs://{PV_METADATA_FILENAME}"
start_dt = datetime.fromisoformat("2019-01-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2019-01-03 00:00:00.000+00:00")
datestamps = pd.date_range(start=start_dt, end=end_dt, freq="5T")

# PV metadata
pv_metadata = pd.read_csv(metadata_filename, index_col="system_id")
pv_longitudes = pv_metadata["longitude"]
pv_latitudes = pv_metadata["latitude"]

# GSP Metadata
gsp_metadata = get_gsp_metadata_from_eso()
# probably need to change this to centroid
gsp_lon = gsp_metadata["gsp_lon"]
gsp_lat = gsp_metadata["gsp_lat"]

# join all sites together
longitudes = list(pv_longitudes.values) + list(gsp_lon.values)
latitudes = list(pv_latitudes.values) + list(gsp_lat.values)

# make d
azimuth, elevation = get_azimuth_and_elevation(
    longitudes=longitudes, latitudes=latitudes, datestamps=datestamps
)


filename = f"./sun.zarr"
save_to_zarr(azimuth=azimuth, elevation=elevation, filename=filename)
