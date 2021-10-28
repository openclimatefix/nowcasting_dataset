############
# Look into the differences from year to year in elevation and azimuthal direction

# Looked from 2018-2020, for January, April, July and October,
# Found the different from year to year was less than 1 degree

############

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import nowcasting_dataset
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_metadata_from_eso
from nowcasting_dataset.data_sources.sun.raw_data_load_save import (
    get_azimuth_and_elevation,
    save_to_zarr,
)

# set up
BUCKET = Path("solar-pv-nowcasting-data")
PV_PATH = BUCKET / "PV/PVOutput.org"
PV_METADATA_FILENAME = PV_PATH / "UK_PV_metadata.csv"

# set up variables
local_path = os.path.dirname(nowcasting_dataset.__file__) + "/.."
metadata_filename = f"gs://{PV_METADATA_FILENAME}"

# PV metadata
pv_metadata = pd.read_csv(metadata_filename, index_col="system_id")
pv_metadata = pv_metadata.dropna(subset=["longitude", "latitude"])
pv_longitudes = pv_metadata["longitude"]
pv_latitudes = pv_metadata["latitude"]

# GSP Metadata
gsp_metadata = get_gsp_metadata_from_eso()
gsp_metadata = gsp_metadata.dropna(subset=["centroid_lon", "centroid_lat"])
# probably need to change this to centroid
gsp_lon = gsp_metadata["centroid_lon"]
gsp_lat = gsp_metadata["centroid_lat"]

# join all sites together
longitudes = list(pv_longitudes.values) + list(gsp_lon.values)
latitudes = list(pv_latitudes.values) + list(gsp_lat.values)

# make d
start_dt = datetime.fromisoformat("2019-01-01 00:00:00.000+00:00")
end_dt = datetime.fromisoformat("2019-01-02 00:00:00.000+00:00")

azimuths = {}
azimuths_sin = {}
azimuths_cos = {}
elevations = {}
months = [1, 4, 7, 10]
years = [2018, 2019, 2020]
for month in months:
    for year in years:
        print(year)
        print(month)
        start_dt = start_dt.replace(year=year, month=month)
        end_dt = end_dt.replace(year=year, month=month)
        datestamps = pd.date_range(start=start_dt, end=end_dt, freq="5T")

        azimuth, elevation = get_azimuth_and_elevation(
            longitudes=longitudes, latitudes=latitudes, datestamps=datestamps
        )

        azimuths[f"{year}_{month}"] = azimuth
        azimuths_sin[f"{year}_{month}"] = np.sin(np.deg2rad(azimuth))
        azimuths_cos[f"{year}_{month}"] = np.cos(np.deg2rad(azimuth))
        elevations[f"{year}_{month}"] = elevation

m_azimuths = []
m_azimuths_sin = []
m_azimuths_cos = []
m_elevations = []
for month in months:
    for year in years[1:]:
        print(year)
        print(month)

        m_azimuths.append(
            (np.abs(azimuths[f"{year}_{month}"].values - azimuths[f"2018_{month}"].values)).max()
        )
        m_azimuths_sin.append(
            (
                np.abs(
                    azimuths_sin[f"{year}_{month}"].values - azimuths_sin[f"2018_{month}"].values
                )
            ).max()
        )
        m_azimuths_cos.append(
            (
                np.abs(
                    azimuths_cos[f"{year}_{month}"].values - azimuths_cos[f"2018_{month}"].values
                )
            ).max()
        )
        m_elevations.append(
            (
                np.abs(elevations[f"{year}_{month}"].values - elevations[f"2018_{month}"].values)
            ).max()
        )


# for small radians, sin(x) ~ x, so sin(x)*180/pi ~ degrees
m_azimuths = np.array(m_azimuths_sin) * 180 / np.pi
# m_azimuths = np.array(m_azimuths_cos) * 180 / np.pi

print(f"Maximum azimuth difference is {max(m_azimuths)} degree")
print(f"Maximum elevation difference is {max(m_elevations)} degree")

# largest different in both azimuth and elevation < 1 degree --> Happy to use one yea data
